"""
Pure-PyTorch implementation of ``custom_rasterizer_kernel.rasterize_image``.

Why this exists
---------------
The upstream ``custom_rasterizer`` ships a CUDA C++ extension. On AMD GPUs
under Windows (native ROCm/HIP PyTorch builds, or ZLUDA) the prebuilt CUDA
wheels cannot be loaded (``DLL load failed`` / missing ``c10_cuda.dll``) and
PTX kernels may be unsupported. This module reproduces the exact semantics
of the original kernel (``lib/custom_rasterizer_kernel/rasterizer_gpu.cu``)
using only standard tensor ops, so it runs on any device PyTorch supports
(ROCm GPU, CUDA GPU, ZLUDA, or CPU) without compilation.

Semantics reproduced from the kernel
------------------------------------
* clip -> screen:  x = (X/W*0.5+0.5)*(width-1)+0.5,
                   y = (0.5+0.5*Y/W)*(height-1)+0.5,
                   z = Z/W*0.49999+0.5
* per-face bbox scan with pixel centres at (px+0.5, py+0.5), inclusive
  barycentric bounds [0, 1].
* z-buffer token = int(depth * 2^18) * MAXINT + (face_idx + 1), resolved with
  an atomic-min (here: ``scatter_reduce(..., 'amin')``).
* optional depth prior: fragments with depth < D*0.49999+0.5+trunc are culled.
* output ``findices`` (int32, 1-based, 0 = background) and perspective
  corrected ``barycentric`` (float32, HxWx3).
"""

import os
import torch

MAXINT = 2147483647
_BG_TOKEN = MAXINT * MAXINT + (MAXINT - 1)
_Z_SCALE = float(2 << 17)
_CANDIDATE_BUDGET = int(os.environ.get("HY3D_RASTER_CANDIDATE_BUDGET", 8388608))


def _signed_area2(ax, ay, bx, by, cx, cy):
    return (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)


def _to_screen(V, width, height):
    w = V[:, 3]
    x = (V[:, 0] / w * 0.5 + 0.5) * (width - 1) + 0.5
    y = (0.5 + 0.5 * V[:, 1] / w) * (height - 1) + 0.5
    z = V[:, 2] / w * 0.49999 + 0.5
    return x, y, z, w


@torch.no_grad()
def rasterize_image(V, F, D, width, height, occlusion_truncation=1e-6, use_depth_prior=0):
    """Drop-in replacement for ``custom_rasterizer_kernel.rasterize_image``.

    Args:
        V: (N, 4) float clip-space vertex positions.
        F: (M, 3) int triangle indices.
        D: (H, W) float depth prior (only used if ``use_depth_prior``).
        width, height: output resolution.
        occlusion_truncation: depth-prior truncation offset.
        use_depth_prior: 0/1.

    Returns:
        [findices (H, W) int32, barycentric (H, W, 3) float32]
    """
    device = V.device
    width = int(width)
    height = int(height)
    V = V.to(torch.float32).contiguous()
    F = F.to(torch.int64).contiguous()
    num_faces = F.shape[0]
    num_pix = width * height

    zbuf = torch.full((num_pix,), _BG_TOKEN, dtype=torch.int64, device=device)

    sx, sy, sz, sw = _to_screen(V, width, height)

    if num_faces > 0:
        x0, x1, x2 = sx[F[:, 0]], sx[F[:, 1]], sx[F[:, 2]]
        y0, y1, y2 = sy[F[:, 0]], sy[F[:, 1]], sy[F[:, 2]]
        z0, z1, z2 = sz[F[:, 0]], sz[F[:, 1]], sz[F[:, 2]]

        xmin = torch.minimum(x0, torch.minimum(x1, x2))
        xmax = torch.maximum(x0, torch.maximum(x1, x2))
        ymin = torch.minimum(y0, torch.minimum(y1, y2))
        ymax = torch.maximum(y0, torch.maximum(y1, y2))

        finite = torch.isfinite(xmin) & torch.isfinite(xmax) & torch.isfinite(ymin) & torch.isfinite(ymax)

        lim = float(max(width, height) + 2)
        xmin_c = torch.nan_to_num(xmin, 0.0).clamp(-lim, lim)
        xmax_c = torch.nan_to_num(xmax, 0.0).clamp(-lim, lim)
        ymin_c = torch.nan_to_num(ymin, 0.0).clamp(-lim, lim)
        ymax_c = torch.nan_to_num(ymax, 0.0).clamp(-lim, lim)

        bx0 = torch.trunc(xmin_c).to(torch.int64).clamp(0, width)
        bx1 = torch.ceil(xmax_c + 1.0).to(torch.int64).clamp(0, width)
        by0 = torch.trunc(ymin_c).to(torch.int64).clamp(0, height)
        by1 = torch.ceil(ymax_c + 1.0).to(torch.int64).clamp(0, height)
        bw = (bx1 - bx0).clamp_min(0)
        bh = (by1 - by0).clamp_min(0)
        area = torch.where(finite, bw * bh, torch.zeros_like(bw))

        if use_depth_prior:
            Dflat = D.to(device=device, dtype=torch.float32).reshape(-1)

        cum = torch.cumsum(area, 0)
        total = int(cum[-1].item())
        if total > 0:
            n_batches = max(1, (total + _CANDIDATE_BUDGET - 1) // _CANDIDATE_BUDGET)
            bounds = torch.arange(1, n_batches, device=device, dtype=torch.int64) * _CANDIDATE_BUDGET
            splits = torch.searchsorted(cum, bounds, right=False).tolist()
            edges = [0] + [s + 1 for s in splits] + [num_faces]

            uniq = []
            for e in edges:
                e = min(max(e, 0), num_faces)
                if not uniq or e > uniq[-1]:
                    uniq.append(e)
            if uniq[0] != 0:
                uniq.insert(0, 0)

            for s, e in zip(uniq[:-1], uniq[1:]):
                a = area[s:e]
                n = int(a.sum().item())
                if n == 0:
                    continue
                local_face = torch.repeat_interleave(torch.arange(e - s, device=device), a)
                start = torch.cumsum(a, 0) - a
                offs = torch.arange(n, device=device, dtype=torch.int64) - start[local_face]
                fid = local_face + s
                w_ = bw[fid]
                px = bx0[fid] + (offs % w_)
                py = by0[fid] + torch.div(offs, w_, rounding_mode="floor")
                del offs, start, local_face, w_

                cx = px.to(torch.float32) + 0.5
                cy = py.to(torch.float32) + 0.5

                ax, ay = x0[fid], y0[fid]
                bxv, byv = x1[fid], y1[fid]
                cxv, cyv = x2[fid], y2[fid]

                beta_tri = _signed_area2(ax, ay, cx, cy, cxv, cyv)
                gamma_tri = _signed_area2(ax, ay, bxv, byv, cx, cy)
                tri_area = _signed_area2(ax, ay, bxv, byv, cxv, cyv)
                nondeg = tri_area != 0
                tri_inv = 1.0 / torch.where(nondeg, tri_area, torch.ones_like(tri_area))
                beta = beta_tri * tri_inv
                gamma = gamma_tri * tri_inv
                alpha = 1.0 - beta - gamma
                del beta_tri, gamma_tri, tri_area, tri_inv, ax, ay, bxv, byv, cxv, cyv, cx, cy

                inside = (
                    nondeg
                    & (alpha >= 0)
                    & (alpha <= 1)
                    & (beta >= 0)
                    & (beta <= 1)
                    & (gamma >= 0)
                    & (gamma <= 1)
                )

                if not bool(inside.any()):
                    continue

                pix = py * width + px
                depth = alpha * z0[fid] + beta * z1[fid] + gamma * z2[fid]
                if use_depth_prior:
                    thres = Dflat[pix] * 0.49999 + 0.5 + occlusion_truncation
                else:
                    thres = torch.zeros_like(depth)
                zq = (depth * _Z_SCALE).to(torch.int32).to(torch.int64)

                keep = inside & ~(depth < thres) & (zq >= 0)
                if not bool(keep.any()):
                    continue
                token = zq * MAXINT + (fid + 1)
                zbuf.scatter_reduce_(0, pix[keep], token[keep], reduce="amin", include_self=True)
                del pix, depth, thres, zq, keep, token, alpha, beta, gamma, inside, fid, px, py

    f = zbuf % MAXINT
    bg = f == (MAXINT - 1)
    findices = torch.where(bg, torch.zeros_like(f), f).to(torch.int32)
    barycentric = torch.zeros((num_pix, 3), dtype=torch.float32, device=device)

    hit = torch.nonzero(findices.view(-1) > 0, as_tuple=False).squeeze(1)
    if hit.numel() > 0:
        fi = findices.view(-1)[hit].to(torch.int64) - 1
        tri = F[fi]
        vx = (hit % width).to(torch.float32) + 0.5
        vy = torch.div(hit, width, rounding_mode="floor").to(torch.float32) + 0.5
        X0, X1, X2 = sx[tri[:, 0]], sx[tri[:, 1]], sx[tri[:, 2]]
        Y0, Y1, Y2 = sy[tri[:, 0]], sy[tri[:, 1]], sy[tri[:, 2]]
        W0, W1, W2 = sw[tri[:, 0]], sw[tri[:, 1]], sw[tri[:, 2]]

        beta_tri = _signed_area2(X0, Y0, vx, vy, X2, Y2)
        gamma_tri = _signed_area2(X0, Y0, X1, Y1, vx, vy)
        tri_area = _signed_area2(X0, Y0, X1, Y1, X2, Y2)
        nondeg = tri_area != 0
        tri_inv = 1.0 / torch.where(nondeg, tri_area, torch.ones_like(tri_area))
        b1 = beta_tri * tri_inv
        b2 = gamma_tri * tri_inv
        b0 = 1.0 - b1 - b2
        neg = torch.full_like(b0, -1.0)
        b0 = torch.where(nondeg, b0, neg)
        b1 = torch.where(nondeg, b1, neg)
        b2 = torch.where(nondeg, b2, neg)

        b0 = b0 / W0
        b1 = b1 / W1
        b2 = b2 / W2
        inv = 1.0 / (b0 + b1 + b2)
        barycentric[hit] = torch.stack((b0 * inv, b1 * inv, b2 * inv), dim=-1)

    return [findices.view(height, width), barycentric.view(height, width, 3)]
