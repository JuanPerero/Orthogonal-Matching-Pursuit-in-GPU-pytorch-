from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, include_paths
import torch
import os
import subprocess

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

def get_cuda_version():
    """Detecta la versión de CUDA instalada"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version = line.split('release ')[1].split(',')[0]
                    return version
    except:
        pass
    return None

def find_cuda_home():
    """Encuentra CUDA_HOME automáticamente"""
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        return cuda_home
    
    # Buscar en ubicaciones comunes
    common_paths = [
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/cuda'
    ]
    
    cuda_version = get_cuda_version()
    if cuda_version:
        versioned_paths = [f'/usr/local/cuda-{cuda_version}']
        common_paths = versioned_paths + common_paths
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return '/usr/local/cuda'  # Default fallback

def get_gpu_arch():
    """Detecta la compute capability de la GPU actual"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        # Obtener la compute capability de la GPU actual
        device_props = torch.cuda.get_device_properties(0)
        major = device_props.major
        minor = device_props.minor
        
        print(f"GPU detectada: {device_props.name}")
        print(f"Compute capability: {major}.{minor}")
        
        return f"{major}{minor}"
    except:
        return None

def get_compatible_arch():
    """Obtiene las arquitecturas compatibles basadas en la GPU detectada"""
    gpu_arch = get_gpu_arch()
    
    if gpu_arch is None:
        print("No se pudo detectar GPU, usando arquitecturas por defecto")
        return [
            '-gencode=arch=compute_50,code=sm_50',
            '-gencode=arch=compute_60,code=sm_60',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86'
        ]
    
    # Mapeo de arquitecturas
    arch_map = {
        '50': ['-gencode=arch=compute_50,code=sm_50'],
        '52': ['-gencode=arch=compute_52,code=sm_52'],
        '60': ['-gencode=arch=compute_60,code=sm_60'],
        '61': ['-gencode=arch=compute_61,code=sm_61'],
        '70': ['-gencode=arch=compute_70,code=sm_70'],
        '75': ['-gencode=arch=compute_75,code=sm_75'],
        '80': ['-gencode=arch=compute_80,code=sm_80'],
        '86': ['-gencode=arch=compute_86,code=sm_86'],
        '87': ['-gencode=arch=compute_87,code=sm_87'],
        '89': ['-gencode=arch=compute_89,code=sm_89'],
        '90': ['-gencode=arch=compute_90,code=sm_90']
    }
    
    if gpu_arch in arch_map:
        print(f"Compilando para arquitectura específica: {gpu_arch}")
        return arch_map[gpu_arch]
    else:
        print(f"Arquitectura {gpu_arch} no reconocida, usando por defecto")
        return [f'-gencode=arch=compute_{gpu_arch},code=sm_{gpu_arch}']




def get_extension_modules():
    """Configura los módulos de extensión CUDA"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA no disponible. No se compilarán extensiones CUDA.")
        return []
    
    cuda_home = find_cuda_home()
    cuda_version = get_cuda_version()
    
    print(f"Usando CUDA desde: {cuda_home}")
    if cuda_version:
        print(f"Versión CUDA detectada: {cuda_version}")
    
    # Configurar paths
    include_dirs = [
        os.path.join(cuda_home, 'include'),
        include_paths()
    ]
    
    library_dirs = [os.path.join(cuda_home, 'lib64')]
    
    # Detectar compute capability automáticamente
    extra_compile_args = {
        'cxx': ['-O3', '-std=c++17', '-fopenmp'],  # Añadir OpenMP
        'nvcc': ['-O3', '--use_fast_math']
    }
    
    # Añadir bibliotecas necesarias
    libraries = ['gomp']  # Biblioteca OpenMP para GCC
    
    # Añadir arquitecturas comunes
    nvcc_args = extra_compile_args['nvcc']
    nvcc_args.extend(get_compatible_arch())
    
    ext_modules = [
        CUDAExtension(
            name='ompingpu.cuda.inverse_omp',
            sources=[
                'ompingpu/cuda/inverse_omp.cpp',
                'ompingpu/cuda/step_cholesky_kernel.cu'
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,  # Añadir bibliotecas
            extra_compile_args=extra_compile_args,
            language='c++'
        )
    ]
    
    return ext_modules

# Leer README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Leer requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ompingpu",
    version="0.2.0",
    author="Juan Perero",
    author_email="tu@email.com",
    description="Orthogonal Matching Pursuit optimizado para GPU con PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/OMPinGPU",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Intended Audience :: Developers", 
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    ext_modules=get_extension_modules(),
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    zip_safe=False,
    keywords=['omp', 'orthogonal matching pursuit', 'gpu', 'cuda', 'pytorch', 'sparse coding'],
)
