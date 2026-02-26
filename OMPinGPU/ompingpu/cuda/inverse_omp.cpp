#include "inverse_omp.h"
#include <cuda_runtime.h>
#include <stdexcept>


torch::Tensor step_cholesky(torch::Tensor A, torch::Tensor B) {
 
    TORCH_CHECK(A.is_cuda(), "A debe estar en GPU");
    TORCH_CHECK(B.is_cuda(), "B debe estar en GPU");
    TORCH_CHECK(R.is_cuda(), "R debe estar en GPU");
    TORCH_CHECK(A.is_contiguous(), "A debe ser contiguo");
    TORCH_CHECK(B.is_contiguous(), "B debe ser contiguo");
    TORCH_CHECK(R.is_contiguous(), "R debe ser contiguo");
      
    if (A.dim() != 2 || B.dim() != 3) {
        throw std::runtime_error("A debe ser 2D y B debe ser 3D");
    }
    
    int M = A.size(0);      // Cantidad de elementos a procesar
    int N = A.size(1);      // Dimensión del renglón a generar
    int full_dim = B.size(1); // Dimensión de la matriz completa
    
    // Validar dimensiones
    TORCH_CHECK(B.size(2) == full_dim, "B debe ser una matriz cuadrada en las últimas dos dimensiones");
    TORCH_CHECK(M == B.size(0), "A y B deben tener el mismo batch size");
    
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



void step_cholesky_w_forward(torch::Tensor L, torch::Tensor B) {
    TORCH_CHECK(L.is_cuda(), "L debe estar en GPU");
    TORCH_CHECK(b.is_cuda(), "b debe estar en GPU");
    TORCH_CHECK(L.is_contiguous(), "L debe ser contiguo");
    TORCH_CHECK(b.is_contiguous(), "b debe ser contiguo");

    int M = L.size(0);      // Cantidad de señales a procesar
    int full_dim = L.size(1);      // Dimensión completa de la matriz L 
    int step = B.size(1); // Dimensión de la matriz de correlaciones

    // Crear tensor de salida
    //torch::Tensor W = torch::zeros({M, step}, L.options());
    AT_DISPATCH_FLOATING_TYPES(L.scalar_type(), "step_cholesky_w_forward",
            ([&] {
                step_cholesky_w_forward_kernel<scalar_t>(
                    L.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    //W.data_ptr<scalar_t>(),
                    M, step, full_dim
                );
            })
        );       
}

// ----------------------------             ###############
// ----------------------------             ###############

void step_fb_coeficients(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D) {
    // A -> matrix L
    // B -> DTX seccionada  -> se puede encontrar el step en base a su tamaño
    // C -> Forward previos
    // D -> Matriz de coeficientes gamma seccionada, para guardar el resultado de los coeficientes calculados en este paso. Se puede usar como buffer de salida, o crear un nuevo tensor para el resultado.

    int M = A.size(0);      // Cantidad de señales a procesar
    int fdim = A.size(1);      // Dimensión completa de la matriz L
    int step_iter = B.size(1); // Paso del algoritmo, con la cantidad de elementos en DTX se puede saber

    //printf("\n Debug: Cantidad de señales M: %d \n", M);
    //printf("\n Debug: Dimension completa: %d \n", fdim);
    //printf("\n Debug: Tamaño de B: %d \n", step_iter);
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "step_fb_coeficients",
                ([&] {
                    step_fb_coeficients_kernel<scalar_t>(
                        A.data_ptr<scalar_t>(),
                        B.data_ptr<scalar_t>(),
                        C.data_ptr<scalar_t>(),
                        D.data_ptr<scalar_t>(),
                        M, step_iter, fdim
                    );
                })
            );
}

// NO DEBERIA TENER UN cuda.synchronize()??

// ----------------------------             ###############
// ----------------------------             ###############
// ----------------------------             ###############


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kernels CUDA optimizados para OMP";
    m.def("step_cholesky", &step_cholesky, 
          "Paso de factorización de Cholesky optimizado para OMP",
          py::arg("A"), py::arg("B"));
    m.def("step_cholesky_w_forward", &step_cholesky_w_forward,
          "Actualización de W usando método de forward para Cholesky",
          py::arg("A"), py::arg("B"));
    m.def("step_fb_coeficients", &step_fb_coeficients,
          "Cálculo de coeficientes usando método forward-backward",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"));
}