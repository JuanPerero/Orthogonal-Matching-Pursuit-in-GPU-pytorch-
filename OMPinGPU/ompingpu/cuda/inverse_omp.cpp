#include "inverse_omp.h"
#include <cuda_runtime.h>
#include <stdexcept>

// Declarar la función CUDA
//template <typename T>
//void step_cholesky_kernel(T* d_A, T* d_B, T* d_R, int N, int M, int fdim);

torch::Tensor step_cholesky(torch::Tensor A, torch::Tensor B) {
 
    // Validaciones de entrada
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::runtime_error("Los tensores deben estar en dispositivo CUDA");
    }
    
    if (A.dim() != 2 || B.dim() != 3) {
        throw std::runtime_error("A debe ser 2D y B debe ser 3D");
    }

    int M = A.size(0);      // Cantidad de elementos a procesar
    int N = A.size(1);      // Dimensión del renglón a generar
    int full_dim = B.size(1); // Dimensión de la matriz completa
    
    // Validar dimensiones
    if (B.size(2) != full_dim) {
        throw std::runtime_error("B debe ser una matriz cuadrada en las últimas dos dimensiones");
    }
    
    if (M != B.size(0)) {
        throw std::runtime_error("A y B deben tener el mismo batch size");
    }
    
    // Crear tensor de salida
    torch::Tensor R = torch::zeros({M, N}, A.options());

    if (M < 1 || N < 1) {
        throw std::runtime_error("Las dimensiones deben ser positivas");
    }

    // Dispatch según el tipo de datos
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "step_cholesky", ([&] {
        step_cholesky_kernel<scalar_t>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            R.data_ptr<scalar_t>(),
            N, M, full_dim
        );
    }));

    return R;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kernels CUDA optimizados para OMP";
    m.def("step_cholesky", &step_cholesky, 
          "Paso de factorización de Cholesky optimizado para OMP",
          py::arg("A"), py::arg("B"));
}