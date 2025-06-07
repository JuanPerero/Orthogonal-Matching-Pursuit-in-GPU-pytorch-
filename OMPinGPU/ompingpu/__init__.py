"""
OMPinGPU - Orthogonal Matching Pursuit optimizado para GPU
========================================================

Esta librería implementa el algoritmo Orthogonal Matching Pursuit (OMP)
optimizado para GPUs usando PyTorch y CUDA.

Funciones principales:
- omp_v4: Implementación principal del algoritmo OMP
- step_cholesky: Kernel CUDA para factorización de Cholesky incremental
"""

__version__ = "0.1.0"
__author__ = "Tu Nombre"

import torch

# Verificar disponibilidad de CUDA
CUDA_AVAILABLE = torch.cuda.is_available()

# Importar funciones principales
from .omp import omp_v4
from .omp import omp_batch
from .omp import onlyinverse

# Intentar importar módulos CUDA
try:
    from .cuda import inverse_omp
    CUDA_EXTENSIONS_AVAILABLE = True
    print("OMPinGPU: Extensiones CUDA cargadas correctamente")
except ImportError as e:
    CUDA_EXTENSIONS_AVAILABLE = False
    print(f"OMPinGPU: Extensiones CUDA no disponibles - {e}")
    print("OMPinGPU: Funcionando solo con PyTorch nativo")

# Exportar funciones públicas
__all__ = [
    'omp_v4',
    'omp_batch',
    'onlyinverse',
    'CUDA_AVAILABLE', 
    'CUDA_EXTENSIONS_AVAILABLE'
]

def get_info():
    """Información sobre la instalación"""
    info = {
        'version': __version__,
        'pytorch_version': torch.__version__,
        'cuda_available': CUDA_AVAILABLE,
        'cuda_extensions_available': CUDA_EXTENSIONS_AVAILABLE,
    }
    
    if CUDA_AVAILABLE:
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        
    return info