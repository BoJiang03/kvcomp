from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get absolute path to include directory
include_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'include'))

def find_all_src_files(src_dir):
    """Recursively find all source files in the given directory."""
    src_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(('.cpp', '.cu')):
                src_files.append(os.path.join(root, file))
    return src_files

setup(
    name="kvcomp_cuda",
    ext_modules=[
        CUDAExtension(
            name="kvcomp_cuda",
            sources=find_all_src_files('src') + ["./export_kvcomp.cpp"],
            include_dirs=[include_path],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)