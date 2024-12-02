#include "inverseOMPV2.h"
#include <cuda_runtime.h>


/* Hipotesis
----------------------------------------------------------------------------------------------------------------
No necesito completamente la matriz L junto con la matriz Linv (inversa) para encontrar el ultimo renglon de la inversa.

Entonces, para calcular el ultimo renglon de la inversa, solo necesito:
- El ultimo renglon de la matriz L
- Y la matriz inversa construida hasta el momento
*/



// Declarar la función CUDA
template <typename Types>
void step_cholesky_kernel(Types * d_A, Types * d_B, Types * d_R, int N, int M, int fdim);

torch::Tensor step_cholesky(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()){
        throw std::runtime_error("Tensors must be on CUDA device");
    }

    int M = A.size(0); // Cantidad de elementos a procesar
    int N = A.size(1); // Dimension del renglon a generar
    int full_dim = B.size(1); // Dimension de la matriz completa
    
    torch::Tensor R = torch::zeros({M,N}).to(A.options());

    if (M < 1 || N < 1){
        throw std::runtime_error("Una de las matrices de entrada es cero.");
    }

    if (A.dtype() == torch::kDouble){
        double *d_A = A.data_ptr<double>();
        double *d_B = B.data_ptr<double>();
        double *d_R = R.data_ptr<double>();
        step_cholesky_kernel<double>(d_A, d_B, d_R, N, M, full_dim);  
    }
    else if (A.dtype() == torch::kFloat32){
        float *d_A = A.data_ptr<float>();
        float *d_B = B.data_ptr<float>();
        float *d_R = R.data_ptr<float>();
        step_cholesky_kernel<float>(d_A, d_B, d_R, N, M, full_dim);  
    }
    else{
        throw std::runtime_error("The tensor must be float32 or float64");
    }
    return R;
}


// Instanciación explícita de las plantillas para los tipos soportados
//template void step_cholesky_kernel<float>(float * d_A, float * d_B, float * d_R, int N, int M);
//template void step_cholesky_kernel<double>(double * d_A, double * d_B, double * d_R, int N, int M);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step_cholesky", &step_cholesky, "Step Cholesky (CUDA)");
}


