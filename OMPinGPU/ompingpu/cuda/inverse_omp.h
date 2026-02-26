#ifndef INVERSE_OMP_H
#define INVERSE_OMP_H

#include <torch/extension.h>

/**
 * Realiza un paso de factorización de Cholesky inversa optimizada para OMP
 * 
 * @param A Tensor 2D con los nuevos renglones de la matriz L
 * @param B Tensor 3D con las matrices inversas actuales
 * @return Tensor con las matrices inversas actualizadas
 */
torch::Tensor step_cholesky(torch::Tensor A, torch::Tensor B);

void step_cholesky_w_forward(torch::Tensor L, torch::Tensor B);

void step_fb_coeficients(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D);

// Declaración de la función CUDA (implementada en step_cholesky_kernel.cu)
template <typename T>
void step_cholesky_kernel(T* d_A, T* d_B, T* d_R, int N, int M, int fdim);

template <typename T>
void step_cholesky_w_forward_kernel(T* d_L, T* d_b, int M, int step, int fdim);

template <typename T>
void step_fb_coeficients_kernel(T* d_L, T* d_b, T* d_forws, T* d_g, int B, int k, int fdim);

#endif // INVERSE_OMP_H