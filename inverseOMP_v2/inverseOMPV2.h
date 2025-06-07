#ifndef INVERSEOMPV2_H
#define INVERSEOMPV2_H

#include <torch/extension.h>

torch::Tensor step_cholesky(torch::Tensor A, torch::Tensor B);

#endif // INVERSEOMP_H
