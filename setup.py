from setuptools import setup
from torch.utils import cpp_extension
import glob

# Automatically find all .cpp and .cu files
cpp_files = glob.glob("custom_ops/csrc/*.cpp")
cu_files = glob.glob("custom_ops/csrc/*.cu")

setup(
    name="custom_ops",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="custom_ops._C",
            sources=cpp_files + cu_files,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"]
            },
            include_dirs=["custom_ops/csrc"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=["custom_ops"]
)