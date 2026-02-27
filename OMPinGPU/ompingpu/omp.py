"""
Implementación del algoritmo Orthogonal Matching Pursuit (OMP) optimizado para GPU
"""

import torch as tc
import warnings

try:
    from .cuda.inverse_omp import step_cholesky, step_cholesky_w_forward, step_fb_coeficients
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
    ctrl_end = end_n.sum()
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
            if ctrl_end != end_n.sum():
                Linv = Linv[end_n]
                ctrl_end = end_n.sum()
            w =  tc.bmm(Linv[end_n,:k,:k],XTX[sets[k, end_n].unsqueeze(1).expand(-1, k), sets.T[end_n, :k],None])
            L = tc.concatenate((w,tc.sqrt(1 - tc.bmm(w.permute(0,2,1),w))),1).squeeze(2)         
            step_cholesky(L,Linv) # Correccion realizada para considerar muestras finalizadas y modificar directamente Linv, lo que no sucedia con linv[end_n]
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


def omp_v5_inv(X, y, XTX=None, n_nonzero_coefs=None, tol=1e-2, device=None):
    """
    OMP v5 - Versión IDÉNTICA a v4 con optimizaciones de memoria.
    Parameters:
    -----------
    X : torch.Tensor (n_samples, n_features)
    y : torch.Tensor (n_samples, n_signals)
    XTX : torch.Tensor, optional (n_features, n_features)
    n_nonzero_coefs : int
    tol : float, default=1e-2
    device : str, optional
        
    Returns:
    --------
    torch.Tensor (n_signals, n_features)
    """
    # ===== PRE-CÓMPUTO =====
    n_samples, n_features = X.shape
    n_signals = y.shape[1]
    
    if XTX is None:
        XTX = X.T @ X    
    
    
    # ===== INICIALIZACIÓN CON BUFFERS REUTILIZABLES =====
    DTX = y.T @ X  # (n_signals, n_atoms)
    residuo = y.clone()
    limit_position = n_features
    sets = tc.full((n_nonzero_coefs, n_signals), limit_position, dtype=tc.int64, device=device)
    gamma = tc.zeros(n_signals, n_nonzero_coefs, device=device, dtype=y.dtype)
    Linv = tc.zeros(n_signals, n_nonzero_coefs, n_nonzero_coefs, device=device, dtype=y.dtype)
    Linv[:, 0, 0] = 1
    # Máscaras
    bool_ctrl = tc.ones(n_signals, n_features, dtype=tc.bool, device=device)
    active_signals = tc.ones(n_signals, dtype=tc.bool, device=device)
    # ===== VARIABLES DE CONTROL (como v4) =====
    ctrl_end = active_signals.sum()
    end_fixed = active_signals.clone()
    # ===== LOOP PRINCIPAL =====
    for k in range(n_nonzero_coefs):
        if not active_signals.any():
            break
        correlation = X.T @ residuo
        best_atoms = correlation.abs().argmax(dim=0)  # (n_signals,)
        sets[k] = best_atoms
        active_signals = active_signals.logical_and(
            bool_ctrl.gather(1, sets[k].unsqueeze(1)).squeeze()
        )
        bool_ctrl.scatter_(1, sets[k].unsqueeze(1), False)
        limit_position = n_features                     # Creo que esto esta de mas
        if tc.any(~active_signals):
            sets[k, ~active_signals] = limit_position
        if active_signals.sum() == 0:
            break
        if k > 0:
            if ctrl_end != active_signals.sum():
                Linv = Linv[active_signals[end_fixed]]
                ctrl_end = active_signals.sum()
                end_fixed = active_signals.clone()
            gram_block = XTX[
                sets[k, active_signals].unsqueeze(1).expand(-1, k),
                sets.T[active_signals, :k],  
                None
            ]
            # === Cholesky update ===
            w = tc.bmm(Linv[:, :k, :k], gram_block)  # (n_active, k, 1)
            
            diag = tc.sqrt(tc.clamp(1 - (w**2).sum(1, keepdim=True), min=1e-10))                  

            L = tc.concatenate((w, diag), 1).squeeze(2)  # (n_active, k+1)
            step_cholesky(L, Linv) 
        gamma[active_signals, :k+1] = tc.gather(DTX, 1, sets.T[active_signals, :k+1])
        gamma[active_signals, :k+1, None] = tc.bmm(
            tc.bmm(Linv[:, :k+1, :k+1].transpose(1, 2), Linv[:, :k+1, :k+1]),
            gamma[active_signals, :k+1, None]
        )
        residuo[:, active_signals] = y[:, active_signals] - tc.bmm(
            gamma[active_signals, None, :k+1],
            X.T[sets.T[active_signals, :k+1], :]
        ).permute(1, 2, 0)[0]          
    # ===== LIMPIAR MEMORIA =====
    del Linv
    tc.cuda.empty_cache()
    indx = tc.arange(n_signals, device=device).repeat(n_nonzero_coefs, 1).T.flatten()
    indx = tc.vstack((indx, sets.T.flatten())) 
    output = tc.zeros(n_signals, n_features + 1, dtype=y.dtype, device=device)
    output[indx[0], indx[1]] = gamma.flatten()
    output = output[:, :-1]
    return output




def omp_v5_fb(X, y, XTX=None, n_nonzero_coefs=None, tol=1e-2, device=None):
    # ===== PRE-CÓMPUTO =====
    n_samples, n_features = X.shape
    n_signals = y.shape[1]
    
    if XTX is None:
        XTX = X.T @ X    
    
    
    # ===== INICIALIZACIÓN CON BUFFERS REUTILIZABLES =====
    DTX = y.T @ X  # (n_signals, n_atoms)
    residuo = y.clone()
    limit_position = n_features
    sets = tc.full((n_nonzero_coefs, n_signals), limit_position, dtype=tc.int64, device=device)
    gamma = tc.zeros(n_signals, n_nonzero_coefs, device=device, dtype=y.dtype)
    g_return = gamma.clone()
    
    L_batch = tc.zeros(n_signals, n_nonzero_coefs, n_nonzero_coefs, device=device, dtype=y.dtype)
    L_batch[:, 0, 0] = 1

    Fordward_buffer = tc.zeros(n_signals, n_nonzero_coefs, device=device, dtype=y.dtype)    # Controlar si es de este tamaño
    
    # Máscaras
    bool_ctrl = tc.ones(n_signals, n_features, dtype=tc.bool, device=device)
    active_signals = tc.ones(n_signals, dtype=tc.bool, device=device)
    # ===== VARIABLES DE CONTROL (como v4) =====
    ctrl_end = active_signals.sum()
    end_fixed = active_signals.clone()
    # ===== LOOP PRINCIPAL =====
    for k in range(n_nonzero_coefs):
        if not active_signals.any():
            break
        correlation = X.T @ residuo
        best_atoms = correlation.abs().argmax(dim=0)  # (n_signals,)
        sets[k] = best_atoms
        active_signals = active_signals.logical_and(
            bool_ctrl.gather(1, sets[k].unsqueeze(1)).squeeze()
        )
        bool_ctrl.scatter_(1, sets[k].unsqueeze(1), False)
        limit_position = n_features
        if tc.any(~active_signals):
            sets[k, ~active_signals] = limit_position
        if active_signals.sum() == 0:
            break
        if k > 0:
            if ctrl_end != active_signals.sum():
                L_batch = L_batch[active_signals[end_fixed]]
                Fordward_buffer = Fordward_buffer[active_signals[end_fixed]]
                g_return[~end_fixed] = gamma[~active_signals[end_fixed]]

                ctrl_end = active_signals.sum()
                end_fixed = active_signals.clone()
            gram_block = XTX[
                sets[k, active_signals].unsqueeze(1).expand(-1, k),
                sets.T[active_signals, :k]]  # gram_block (n_active, k)
            
            # === Cholesky update with forward method ===          
            #w = tc.bmm(Linv[:, :k, :k], gram_block)  # (n_active, k, 1)
            step_cholesky_w_forward(L_batch, gram_block)
            w = L_batch[:, :k+1, :k+1]
            diag = tc.sqrt(tc.clamp(1 - (w**2).sum(2, keepdim=True), min=1e-10))
            L = tc.concatenate((w, diag), 1).squeeze(2)  # (n_active, k+1)
            
            
        # Adaptar al formato Forward-Backward
        step_fb_coeficients(L_batch, DTX[active_signals, :k+1], Fordward_buffer, gamma)  #gamma[active_signals, :k+1]) 
        residuo[:, active_signals] = y[:, active_signals] - tc.bmm(
            gamma[active_signals, None, :k+1],
            X.T[sets.T[active_signals, :k+1], :]
        ).permute(1, 2, 0)[0]

    tc.cuda.empty_cache()
    indx = tc.arange(n_signals, device=device).repeat(n_nonzero_coefs, 1).T.flatten()
    indx = tc.vstack((indx, sets.T.flatten())) 
    output = tc.zeros(n_signals, n_features + 1, dtype=y.dtype, device=device)
    output[indx[0], indx[1]] = g_return.flatten() # gamma.flatten()
    output = output[:, :-1]
    return output





def omp_batch(X, Y, n_nonzero_coefs, batch_size=20000, method="inv", **kwargs):
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

    if method == "inv":
        omp_v5_func = omp_v5_inv
    elif method == "fb":
        omp_v5_func = omp_v5_fb  # Aquí se podría implementar una versión forward-backward si se desea
    else:
        raise ValueError("Método no reconocido. Use 'inv' o 'fb'.")

    # Validación de entrada
    if not isinstance(X, tc.Tensor):
        raise TypeError("X debe ser torch.Tensor")
    if not isinstance(Y, tc.Tensor):
        raise TypeError("Y debe ser torch.Tensor")
    if X.dim() != 2:
        raise ValueError("X debe ser 2D")
    if Y.dim() != 2:
        raise ValueError("Y debe ser 2D")
    if X.size(0) != Y.size(0):
        raise ValueError("X e Y deben tener el mismo número de filas")
    if X.device != Y.device:
        raise ValueError(f"X e Y deben estar en el mismo dispositivo (X: {X.device}, Y: {Y.device})")
    if not isinstance(n_nonzero_coefs, int) or n_nonzero_coefs <= 0:
        raise ValueError("n_nonzero_coefs debe ser un entero positivo")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size debe ser un entero positivo")
    if Y.size(1) == 0:
        raise ValueError("Y debe contener al menos una señal (columnas)")
    
    n_signals = Y.size(1)
    n_features = X.size(1)
    device = Y.device

    # Preasignar tensor de salida
    output = tc.zeros(n_signals, n_features, device=device, dtype=Y.dtype)
    for i in range(0, n_signals, batch_size):
        end_idx = min(i + batch_size, n_signals)
        y_batch = Y[:, i:end_idx]
        result_batch = omp_v5_func(X, y_batch, n_nonzero_coefs=n_nonzero_coefs, device=device, **kwargs)
        output[i:end_idx] = result_batch
        tc.cuda.empty_cache()
    if tc.isnan(output).any():
        print("Warning: NaN values detected in the output. Consider increasing the tolerance or checking the input data.")
    return output

def onlyinverse(L1,L2):
    inverse_omp.step_cholesky(L1,L2)
    return 


### Lista de modificaciones realizadas:
# - Se agregó validación de entrada para asegurar que X e y sean tensores de pytorch
# - Control de valores NaN en la salida, con una advertencia para el usuario
# - Planteo de procesamiento con la funcion fordward-backward

