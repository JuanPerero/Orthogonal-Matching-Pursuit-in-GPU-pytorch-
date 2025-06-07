#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>

#define imin(a,b) (a<b?a:b)

template <typename Types>
__global__ void MultiplicationKernel(Types * d_A, Types * d_B, Types * d_R, int N, int M, int fdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    
    if (idx < M){  
        Types* a = d_A + idx * N;           // Direcciona el array a
        Types* b = d_B + idx * fdim * fdim; // Direcciona la matriz B
        Types* r = d_R + idx * N;           // Direcciona el array r

        int aux_ind1 = (N-1) * fdim;        //Direcciona el renglon de B a computar
        int aux_ind2 = aux_ind1 + (N-1);    // Direcciona el ultimo elemento de B a computar

        b[aux_ind2] = 1/a[N-1];
        r[N-1] = 1/a[N-1];
        
        for(int k = 0; k < N-1; k++){
            int aux3 = aux_ind1 + k;
            b[aux3] = 0;
            for(int i = k; i < N-1; i++){
                b[aux3] -= a[i] * b[i * fdim + k]; 
            }
            b[aux3] /= a[N-1];
            r[k] = b[aux3];
        }       
    }
}

template <typename Types>
void step_cholesky_kernel(Types * d_A, Types * d_B, Types * d_R, int N, int M, int fdim) {   
    if (M <= 0) {
        printf("Error: M debe ser > 0\n");
        return;
    }

    int threadsPerBlock = 256;
    threadsPerBlock = imin(M, threadsPerBlock);
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;   
   
    // Llamar al kernel
    MultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_R, N, M, fdim);

    // Verificar errores del kernel INMEDIATAMENTE
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error al lanzar kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Sincronizar el dispositivo
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error en sincronización: %s\n", cudaGetErrorString(err));
        return;
    }   
}

// Instanciación explícita de las plantillas para los tipos soportados
template void step_cholesky_kernel<float>(float * d_A, float * d_B, float * d_R, int N, int M, int fdim);
template void step_cholesky_kernel<double>(double * d_A, double * d_B, double * d_R, int N, int M, int fdim);