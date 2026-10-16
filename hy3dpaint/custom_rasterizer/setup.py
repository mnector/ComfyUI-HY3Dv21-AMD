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
import sys
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if sys.platform == "win32":
    os.environ["CL"] = (
        os.environ.get("CL", "")
        + " /std:c++20 /O2 /DTORCH_EXTENSION_NAME=custom_rasterizer_kernel /DTORCH_API_INCLUDE_EXTENSION_H /wd4244 /wd4267 /wd4251"
    )


# On AMD (ROCm / ZLUDA) or when CUDA nvcc is unavailable, build CppExtension
# using the C++ rasterizer kernel (which handles GPU tensors via automatic CPU
# staging) to avoid PTX / c10_cuda.dll incompatibilities.
use_cuda_ext = (
    os.environ.get("FORCE_CUDA_RASTERIZER", "0") == "1"
    and torch.version.cuda is not None
    and getattr(torch.version, "hip", None) is None
)

if use_cuda_ext:
    cxx_flags = ["/O2", "/std:c++20", "-DWITH_CUDA_RASTERIZER"] if sys.platform == "win32" else ["-O3", "-std=c++20", "-DWITH_CUDA_RASTERIZER"]
    custom_rasterizer_module = CUDAExtension(
        "custom_rasterizer_kernel",
        [
            "lib/custom_rasterizer_kernel/rasterizer.cpp",
            "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
            "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": ["-O3", "-std=c++20", "-DWITH_CUDA_RASTERIZER"],
        },
    )
else:
    cxx_flags = ["/O2", "/std:c++20"] if sys.platform == "win32" else ["-O3", "-std=c++20"]
    custom_rasterizer_module = CppExtension(
        "custom_rasterizer_kernel",
        [
            "lib/custom_rasterizer_kernel/rasterizer.cpp",
            "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        ],
        extra_compile_args=cxx_flags,
    )

class SafeBuildExtension(BuildExtension):
    def build_extensions(self):
        if sys.platform == "win32":
            for ext in self.extensions:
                if isinstance(ext.extra_compile_args, dict) and "cxx" in ext.extra_compile_args:
                    ext.extra_compile_args["cxx"] = [
                        arg for arg in ext.extra_compile_args["cxx"]
                        if not arg.startswith("-fms-runtime-lib")
                    ]
                elif isinstance(ext.extra_compile_args, list):
                    ext.extra_compile_args = [
                        arg for arg in ext.extra_compile_args
                        if not arg.startswith("-fms-runtime-lib")
                    ]
        super().build_extensions()


setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": SafeBuildExtension.with_options( use_ninja=False )},
)

