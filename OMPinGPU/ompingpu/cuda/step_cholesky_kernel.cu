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
    #ifdef DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error en sincronización: %s\n", cudaGetErrorString(err));
        return;
    }   
}

// Instanciación explícita de las plantillas para los tipos soportados
template void step_cholesky_kernel<float>(float * d_A, float * d_B, float * d_R, int N, int M, int fdim);
template void step_cholesky_kernel<double>(double * d_A, double * d_B, double * d_R, int N, int M, int fdim);







// ------------------------------------------------------------------------------------
// --------------           Forward-Backward para cálculo de coeficientes en OMP incremental  
// ------------------------------------------------------------------------------------
template <typename Types>
__global__ void IncrementalForwBack(Types* d_L, Types* d_b, Types* d_forw_step, Types* d_x, int M, int step, int fdim)
{   
    /*
    L es la matriz triangular inferior
    b es el vector del lado derecho - samples/señales Y en el omp
    forw_step es el almacenamiento de los valores forward hasta el momento
    x es el resultado retornado. 
    M es el tamaño total de la matriz L (cantidad de pasos en omp)
    step es el paso actual del omp
    fdim es la dimensión total de la matriz L
    Muchas de las variables se pasan con punteros, es decir, que son direcciones de memoria.
    */

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (batch_idx >= M) return;

    Types* L = d_L + batch_idx * fdim * fdim;
    Types* b = d_b + batch_idx * step;
    Types* forw_step = d_forw_step + batch_idx * fdim;  // Unidimensional con tamaño fdim, aunque sean ceros
    Types* x = d_x + batch_idx * fdim;                  // Unidimensional con tamaño fdim, aunque sean ceros   

    // ---------- Forward substitution ----------
    if (step == 0)
        forw_step[0] = b[0] / L[0];
    else{
        Types sum = 0.0;
        for (int j = 0; j < step; j++)
            sum += L[step*fdim + j] * forw_step[j];
            forw_step[step] = (b[step] - sum) / L[step*fdim + step];
    }

    // ---------- Backward substitution ----------   
    for (int i = step; i >= 0; i--)
    {      
        Types sum = 0.0;
        for (int j = step; j > i; j--){
            sum += L[j*fdim + i] * x[j]; 
        }
        x[i] = (forw_step[i] - sum) / L[i*fdim + i];
    }

}

// Lanzador para batch_forward_substitution_kernel
template <typename Types>
void step_fb_coeficients_kernel(
    Types* d_L, Types* d_b, Types* d_forws, Types* d_g, int B, int k, int fdim)
{
    int threadsPerBlock = 256;
    int threads = (B < threadsPerBlock) ? B : threadsPerBlock;
    int blocks = (B + threads - 1) / threads;

    k -= 1;
    IncrementalForwBack<Types><<<blocks, threads>>>(d_L, d_b, d_forws, d_g, B, k, fdim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error al lanzar batch_forward_substitution_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    #ifdef DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error en sincronización: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Instanciación explícita para float y double
template void step_fb_coeficients_kernel<float>(float* d_L, float* d_b, float* d_forws, float* d_g, int B, int k, int fdim);
template void step_fb_coeficients_kernel<double>(double* d_L, double* d_b, double* d_forws, double* d_g, int B, int k, int fdim);



// ------------------------------------------------------------------------------------
// --------------           Forward para actualización de W en Cholesky incremental     
// ------------------------------------------------------------------------------------

template <typename Types>
__global__ void batch_forward_substitution_kernel(
    Types* d_L,                  
    Types* __restrict__ d_b,  
    int M, int step, int fdim)
{   
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= M) return;
    
    Types* L = d_L + batch_idx * fdim * fdim;
    Types* b = d_b + batch_idx * step;   
    
    #pragma unroll
    for (int i = 0; i < step; ++i) {
        Types sum = 0;
        for (int j = 0; j < i; ++j)
            sum += L[i * fdim + j] * L[step * fdim + j]; 
        L[step*fdim + i] = (b[i] - sum) / L[i * fdim + i];
    }
}



template <typename Types>
void step_cholesky_w_forward_kernel(
    Types* d_L, Types* d_b, int M, int step, int fdim)
{
    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    batch_forward_substitution_kernel<Types><<<blocks, threads>>>(d_L, d_b, M, step, fdim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error al lanzar kernel: %s\n", cudaGetErrorString(err));
        return;
    }
    #ifdef DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error en sincronización: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Instanciación explícita de la plantilla para tipos soportados
template void step_cholesky_w_forward_kernel<float>(float* d_L, float* d_b, int M, int step, int fdim);
template void step_cholesky_w_forward_kernel<double>(double* d_L, double* d_b, int M, int step, int fdim);