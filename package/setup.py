import torch
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

compute_arch = 'compute_61'
include_dir = 'ext_modules/include'
nn_src = 'ext_modules/src/nn/cpp'
nn_src_cuda = 'ext_modules/src/nn/cuda'
util_src = 'ext_modules/src/util/cpp'
util_src_cuda = 'ext_modules/src/util/cuda'
eigen_include_dir = '/usr/local/include/eigen3'


def extension(name,
              package,
              source_basename,
              cuda=True,
              include_dirs=[],
              cxx_compile_args=[],
              nvcc_compile_args=[]):
    '''Create a build extension. Use CUDA if available, otherwise C++ only'''
    if package == 'nn':
        prefix = nn_src
        prefix_cuda = nn_src_cuda
    elif package == 'util':
        prefix = util_src
        prefix_cuda = util_src_cuda

    if torch.cuda.is_available() and cuda:
        return CUDAExtension(
            name=name,
            sources=[
                osp.join(prefix, source_basename + '.cpp'),
                osp.join(prefix_cuda, source_basename + '.cu'),
            ],
            include_dirs=[include_dir] + include_dirs,
            extra_compile_args={
                'cxx': ['-fopenmp', '-O3'] + cxx_compile_args,
                'nvcc':
                ['--gpu-architecture=' + compute_arch] + nvcc_compile_args
            })
    else:
        return CppExtension(
            name=name,
            sources=[
                osp.join(prefix, source_basename + '.cpp'),
            ],
            include_dirs=[include_dir] + include_dirs,
            define_macros=[('__NO_CUDA__', None)],
            extra_compile_args={
                'cxx': ['-fopenmp', '-O3'] + cxx_compile_args,
                'nvcc': [] + nvcc_compile_args
            })


setup(
    name='Tangent Images',
    version='0.0.1',
    author='Marc Eder',
    author_email='meder@cs.unc.edu',
    description='A PyTorch module for tangent_images',
    ext_package='_tangent_images_ext',
    ext_modules=[
        extension('_resample', 'nn', 'resample_layer'),
        extension('_weighted_resample', 'nn', 'weighted_resample_layer'),
        extension('_uv_resample', 'nn', 'uv_resample_layer'),
        extension('_mesh', 'util', 'triangle_mesh', False, [eigen_include_dir],
                  ['-DCGAL_HEADER_ONLY'])
    ],
    packages=[
        'tangent_images',
        'tangent_images.nn',
        'tangent_images.util',
    ],
    package_dir={
        'tangent_images': 'layers',
        'tangent_images.nn': 'layers/nn',
        'tangent_images.util': 'layers/util',
    },
    cmdclass={'build_ext': BuildExtension},
)
