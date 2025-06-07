from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='inverseOMPV2',
    ext_modules=[cpp_extension.CppExtension(
        'inverseOMPV2',                # Nombre del módulo de extensión
        ['inverseOMPV2.cpp', 'step_cholesky_kernel.cu'],          # Archivos de código fuente
        include_dirs=['/usr/local/cuda-11.8/include',
                      cpp_extension.include_paths()],  # Directorios de inclusión de PyTorch
        library_dirs=['/usr/local/cuda/lib64'],
        extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'] #,'-gencode=arch=compute_52,code=sm_52'
                }
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},  # Configuración de la clase de construcción
)
