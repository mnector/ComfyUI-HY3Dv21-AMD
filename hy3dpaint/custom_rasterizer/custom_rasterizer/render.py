# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
from . import torch_backend

try:
    import custom_rasterizer_kernel as _cr_kernel
except Exception:
    _cr_kernel = None

_FORCE_TORCH_BACKEND = os.environ.get("HY3D_RASTER_BACKEND", "").lower() == "torch"
_FORCE_CPU_FALLBACK = os.environ.get("HY3D_RASTER_BACKEND", "").lower() == "cpu"


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    orig_device = pos.device
    v = pos[0]

    if not _FORCE_TORCH_BACKEND and _cr_kernel is not None:
        if _FORCE_CPU_FALLBACK:
            v_cpu = v.detach().to("cpu", dtype=torch.float32).contiguous()
            tri_cpu = tri.detach().to("cpu", dtype=torch.int32).contiguous()
            d_cpu = clamp_depth.detach().to("cpu", dtype=torch.float32).contiguous() if use_depth_prior else torch.zeros(0, dtype=torch.float32)
            findices, barycentric = _cr_kernel.rasterize_image(
                v_cpu, tri_cpu, d_cpu, int(resolution[1]), int(resolution[0]), 1e-6, int(use_depth_prior)
            )
            return findices.to(orig_device), barycentric.to(orig_device)

        try:
            findices, barycentric = _cr_kernel.rasterize_image(
                v, tri, clamp_depth, int(resolution[1]), int(resolution[0]), 1e-6, int(use_depth_prior)
            )
            return findices.to(orig_device), barycentric.to(orig_device)
        except Exception:
            try:
                v_cpu = v.detach().to("cpu", dtype=torch.float32).contiguous()
                tri_cpu = tri.detach().to("cpu", dtype=torch.int32).contiguous()
                d_cpu = clamp_depth.detach().to("cpu", dtype=torch.float32).contiguous() if use_depth_prior else torch.zeros(0, dtype=torch.float32)
                findices, barycentric = _cr_kernel.rasterize_image(
                    v_cpu, tri_cpu, d_cpu, int(resolution[1]), int(resolution[0]), 1e-6, int(use_depth_prior)
                )
                return findices.to(orig_device), barycentric.to(orig_device)
            except Exception:
                pass

    try:
        findices, barycentric = torch_backend.rasterize_image(
            v, tri, clamp_depth, int(resolution[1]), int(resolution[0]), 1e-6, int(use_depth_prior)
        )
        return findices.to(orig_device), barycentric.to(orig_device)
    except Exception:
        findices, barycentric = torch_backend.rasterize_image(
            v.detach().cpu(), tri.detach().cpu(), clamp_depth.detach().cpu(), int(resolution[1]), int(resolution[0]), 1e-6, int(use_depth_prior)
        )
        return findices.to(orig_device), barycentric.to(orig_device)


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
