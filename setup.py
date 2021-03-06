import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():
    extensions_dir = os.path.join("atss_core", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu + source_cuda

    extra_compile_args = {"cxx": []}
    define_macros = []

    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = ["-DCUDA_HAS_FP16=1",
                                  "-D__CUDA_NO_HALF_OPERATORS__",
                                  "-D__CUDA_NO_HALF_CONVERSIONS__",
                                  "-D__CUDA_NO_HALF2_OPERATORS__"]

    include_dirs = [extensions_dir]
    ext_modules = [CUDAExtension("atss_core._C",
                                 sources,
                                 include_dirs=include_dirs,
                                 define_macros=define_macros,
                                 extra_compile_args=extra_compile_args)]

    return ext_modules


setup(name="ATSS_minimal",
      version="1.0",
      packages=find_packages(exclude=("configs", "tests",)),
      ext_modules=get_extensions(),
      cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension})
