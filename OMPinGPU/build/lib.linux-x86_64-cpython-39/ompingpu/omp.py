"""
Implementación del algoritmo Orthogonal Matching Pursuit (OMP) optimizado para GPU
"""

import torch as tc
import warnings

try:
    from .cuda import inverse_omp
    CUDA_EXTENSIONS = True
except ImportError:
    CUDA_EXTENSIONS = False
    warnings.warn("Extensiones CUDA no disponibles. Usando implementación PyTorch pura.")


def omp_v4(X, y, XTX=None, n_nonzero_coefs=None, tol=1e-2, device=None):
    """
    Orthogonal Matching Pursuit (OMP) optimizado para GPU.
    
    Parameters:
    -----------
    X : torch.Tensor
        Matriz de diccionario de forma (n_samples, n_features)
    y : torch.Tensor  
        Señales objetivo de forma (n_samples, n_signals)
    XTX : torch.Tensor, optional
        Matriz de Gram pre-computada X.T @ X
    n_nonzero_coefs : int
        Número máximo de coeficientes no cero
    tol : float, default=1e-2
        Tolerancia para convergencia
    device : str, optional
        Dispositivo donde ejecutar ('cuda' o 'cpu')
        
    Returns:
    --------
    torch.Tensor
        Coeficientes sparse de forma (n_signals, n_features)
    """
    
    # Validación de entrada
    if not isinstance(X, tc.Tensor) or not isinstance(y, tc.Tensor):
        raise TypeError("X e y deben ser torch.Tensor")
    
    if X.dim() != 2 or y.dim() != 2:
        raise ValueError("X debe ser 2D y y debe ser 2D")
        
    if X.size(0) != y.size(0):
        raise ValueError("X e y deben tener el mismo número de filas")
    
    if X.device != y.device:
            raise ValueError(f"X e y deben estar en el mismo dispositivo (X: {X.device}, y: {y.device})")

    # Determinar el dispositivo a usar
    if device is None:
        device = X.device
    
    if XTX is None:
        XTX = X.T @ X    
    else:
        if not XTX.size(0) == X.size(1) and XTX.size(1) == X.size(1):
            print("XTX debe ser una matriz cuadrada del tamaño igual a la cantidad de atomos del diccionario")
        XTX = XTX.to(device)
    
    B = y.shape[1]  # cantidad de señales
    DTX = y.T @ X   # Correlación
    residuo = y.clone()
    
    limit_position = X.size(1)
    # Inicialización
    sets = limit_position * y.new_ones(n_nonzero_coefs, B, dtype=tc.int64)
    gamma = y.new_zeros(B, n_nonzero_coefs)

    Linv = y.new_zeros(B, n_nonzero_coefs, n_nonzero_coefs)
    Linv[:, 0, 0] = 1
    
    bool_ctrl = tc.ones((B, X.shape[1]), dtype=tc.bool, device=device)
    end_n = tc.ones(B, dtype=tc.bool, device=device)
    
    for k in range(n_nonzero_coefs):
        correlation = X.T @ residuo
        sets[k] = correlation.abs().argmax(0)
        # Primero verifico si en la mascara estan activados los elementos seleccionados
        end_n = end_n.logical_and(bool_ctrl.gather(1, sets[k].unsqueeze(1)).squeeze())      
        # Marco la mascara de elementos seleccionados
        bool_ctrl.scatter_(1, sets[k].unsqueeze(1), False)
        if tc.any(~end_n):
            sets[k,~end_n] = limit_position
        if end_n.sum()==0:
            break
        if k:          
            w =  tc.bmm(Linv[end_n,:k,:k],XTX[sets[k, end_n].unsqueeze(1).expand(-1, k), sets.T[end_n, :k],None])
            L = tc.concatenate((w,tc.sqrt(1 - tc.bmm(w.permute(0,2,1),w))),1).squeeze(2)         
            inverse_omp.step_cholesky(L,Linv) # Esto no toma en cuenta el end_n, necesitaria implementar el retornoy asignarlo al Linv
        gamma[end_n,:k+1] = tc.gather(DTX, 1, sets.T[end_n,:k+1]) 
        gamma[end_n,:k+1,None] = tc.bmm(tc.bmm(Linv[end_n,:k+1,:k+1].transpose(1,2), Linv[end_n,:k+1,:k+1]), gamma[end_n,:k+1,None])              
        residuo[:,end_n] = y[:,end_n] - tc.bmm(gamma[end_n,None,:k+1],X.T[sets.T[end_n, :k+1], :]).permute(1,2,0)[0]
   
    # Limpiar memoria
    if 'L' in locals():
        del L
    if 'Linv' in locals():
        del Linv
    tc.cuda.empty_cache()
    indx = tc.arange(y.shape[1],device=X.device).repeat(n_nonzero_coefs,1).T.flatten()
    indx = tc.vstack((indx.to("cuda"),sets.T.flatten()))
    dense_tensor = tc.zeros(y.shape[1], X.size(1)+1).to(device) # Mas uno es necesario por la condicion de corte (ultima posicion+1)
    dense_tensor[indx[0], indx[1]] = gamma.flatten()
    # Eliminar la ultima columna del retorno
    dense_tensor = dense_tensor[:,:-1]   
    return dense_tensor


def omp_batch(X, Y, n_nonzero_coefs, batch_size=20000, **kwargs):
    """
    Procesa múltiples señales en lotes para manejar datasets grandes.
    
    Parameters:
    -----------
    X : torch.Tensor
        Matriz de diccionario
    Y : torch.Tensor
        Múltiples señales objetivo
    n_nonzero_coefs : int
        Número de coeficientes no cero
    batch_size : int
        Tamaño del lote
    **kwargs : dict
        Argumentos adicionales para omp_v4
        
    Returns:
    --------
    torch.Tensor
        Coeficientes para todas las señales
    """
    n_signals = Y.size(1)
    results = []
    
    for i in range(0, n_signals, batch_size):
        end_idx = min(i + batch_size, n_signals)
        y_batch = Y[:, i:end_idx]
        
        result_batch = omp_v4(X, y_batch, n_nonzero_coefs=n_nonzero_coefs, **kwargs)
        results.append(result_batch)
        
        # Limpiar memoria después de cada lote
        tc.cuda.empty_cache()
    
    return tc.cat(results, dim=0)


def onlyinverse(L1,L2):
    inverse_omp.step_cholesky(L1,L2)
    return 
