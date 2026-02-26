#include <cuda_runtime.h>
#include <torch/extension.h>

#define imin(a,b) (a<b?a:b)

template <typename Types>
__global__ void MultiplicationKernel(Types * d_A, Types * d_B, Types * d_R, int N, int M, int fdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M){

        Types* a = d_A + idx * N;           // Direcciona el array a
        Types* b = d_B + idx * fdim * fdim; // Direcciona la matriz B
        Types* r = d_R + idx * N;           // Direcciona el array r

        int aux_ind1 = (N-1) * fdim;  //Direcciona el renglon de B a computar
        int aux_ind2 = aux_ind1 + (N-1);  // Direcciona el ultimo elemento de B a computar

        //if(idx==0){
            //printf("\nStart: %d - %d\n", aux_ind1, aux_ind2);
        //}


        b[aux_ind2] = 1/a[N-1];
        r[N-1] = 1/a[N-1];
        
        for(int k = 0; k < N-1; k++){
            int aux3 = aux_ind1 + k;
            b[aux3] = 0;
            for(int i = k; i < N-1; i++){
                b[aux3] -= a[i] * b[i * fdim + k]; 
                //if(idx==0){
                //    printf("\nStart: %d - %d - %d\n", aux3, i * fdim + k, i);
                //}
            }
            b[aux3] /= a[N-1];
            r[k] = b[aux3];
        }
    }

}

template <typename Types>
void step_cholesky_kernel(Types * d_A, Types * d_B, Types * d_R, int N, int M, int fdim) {
    int threadsPerBlock = 256;
    threadsPerBlock = imin(M, threadsPerBlock);
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
    
    // Llamar al kernel
    MultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_R, N, M, fdim);

    // Sincronizar el dispositivo
    cudaDeviceSynchronize();
}


template __global__ void MultiplicationKernel(float * d_A, float * d_B, float * d_R, int N, int M, int fdim);
template __global__ void MultiplicationKernel(double * d_A, double * d_B, double * d_R, int N, int M, int fdim);

// Instanciación explícita de las plantillas para los tipos soportados
template void step_cholesky_kernel(float * d_A, float * d_B, float * d_R, int N, int M, int fdim);
template void step_cholesky_kernel(double * d_A, double * d_B, double * d_R, int N, int M, int fdim);




template <typename Types>
__global__ void ForwBackKernel(Types * d_L, Types * d_y, Types * d_x, Types * d_fordw, int N, int M, int fdim) {
    // d_L = direccion en memoria de la matriz L creada
    // d_y = direccion en memoria de las samples
    // d_x = direccion en memoria de las soluciones
    // d_fordw = direccion en memoria de las soluciones intermedias (forward)
    // N = tamaño del sistema
    // M = cantidad de sistemas a resolver
    // fdim = dimensiones de las filas de la matriz L
    //                                              M es la cantidad de samples, señales o sistemas a resolver?

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < M){   
        Types* L_int = d_L + idx * N;            // Direcciona el array particular de "L"    
        Types* b_int = d_y + idx * fdim * fdim;  // Direcciona el array particular de la señal a procesar "b"
        Types* ford_int = d_fordw + idx * N;           // Direcciona el array de los forwards previos
        
        // ----------------------------------------------------------
        //                  hay que acomodar el direccionamiento
        // ----------------------------------------------------------


        // ------------------------------------------------------------------------------------
        //                  hay que pasar a esta funcion el numero de la iteracion calculada
        int it_omp;
        float aux_sum = 0;

        int desp_row = it_omp * fdim;

        // Forward incremental substitution       
        for(it_f = 0; it_f<it_omp; it_f++){
            aux_sum = L_int[desp_row + it_f] * b_int[it_f];
        }
        ford_int[it_omp] = (b_int[it_omp] - aux_sum)/L_int[desp_row + it_omp]

        // Backward substitution
        //x = np.zeros(i+1) --> debe cargarse en d_x
        for(int j=0;j<=it_omp;j++){
            if(j==0){
                d_x[it_omp] = ford_int[it_omp] / L_int[desp_row + it_omp];
            }
            else{
                aux_sum = 0;
                for (int k = it_omp - j + 1; k <= it_omp; ++k) {
                    aux_sum += L_int[k * fdim + (it_omp - j)] * b_int[k];
                }
                b_int[it_omp - j] = (ford_int[it_omp - j] - aux_sum) / L_int[desp_row + (it_omp - j)];
            }
        }  
}


template __global__ void ForwBackKernel(float * d_A, float * d_B, float * d_R, int N, int M, int fdim);
template __global__ void ForwBackKernel(double * d_A, double * d_B, double * d_R, int N, int M, int fdim);