import OMPinGPU.inverseOMP as inverseOMP
import torch as tc


def omp_v4(X, y, XTX=None, n_nonzero_coefs=None, tol=1e-2, inverse_cholesky=True):
    if XTX is None:
        XTX = X.T @ X    
    B = y.shape[1] # cantidad de señales, muestras o elementos de y
    DTX = y.T @ X  # Correlacion pero en dimension 17k x 128
    residuo = y.clone()
    sets = X.size(1)*y.new_ones(n_nonzero_coefs, B, dtype=tc.int64)
    gamma = y.new_zeros(B,n_nonzero_coefs)

    Linv = y.new_zeros(B,n_nonzero_coefs,n_nonzero_coefs)
    Linv[:,0,0] = 1

    bool_ctrl = tc.ones((B, X.shape[1]), dtype=tc.bool, device=y.device)
    end_n = tc.ones(B, dtype=tc.bool, device=y.device)

    for k in range(n_nonzero_coefs):
        correlation = X.T @ residuo
        sets[k] = correlation.abs().argmax(0)
        # Primero verifico si en la mascara estan activados los elementos seleccionados
        end_n = end_n.logical_and(bool_ctrl.gather(1, sets[k].unsqueeze(1)).squeeze())      
        # Marco la mascara de elementos seleccionados
        bool_ctrl.scatter_(1, sets[k].unsqueeze(1), False)
        if tc.any(~end_n):
            sets[k,~end_n] = 256
        if end_n.sum()==0:
            break
        if k:
            w =  tc.bmm(Linv[end_n,:k,:k],XTX[sets[k, end_n].unsqueeze(1).expand(-1, k), sets.T[end_n, :k],None])
            L = tc.concatenate((w,tc.sqrt(1 - tc.bmm(w.permute(0,2,1),w))),1).squeeze(2)
            inverseOMP.step_cholesky(L,Linv) # Esto no toma en cuenta el end_n, necesitaria implementar el retornoy asignarlo al Linv
        gamma[end_n,:k+1] = tc.gather(DTX, 1, sets.T[end_n,:k+1]) 
        gamma[end_n,:k+1,None] = tc.bmm(tc.bmm(Linv[end_n,:k+1,:k+1].transpose(1,2), Linv[end_n,:k+1,:k+1]), gamma[end_n,:k+1,None])              
        residuo[:,end_n] = y[:,end_n] - tc.bmm(gamma[end_n,None,:k+1],X.T[sets.T[end_n, :k+1], :]).permute(1,2,0)[0]
    del L
    del Linv
    tc.cuda.empty_cache()

    indx = tc.arange(y.shape[1],device=X.device).repeat(n_nonzero_coefs,1).T.flatten()
    indx = tc.vstack((indx.to("cuda"),sets.T.flatten()))
    retorno = tc.sparse_coo_tensor(indx,gamma.flatten(),size=(y.shape[1],X.size(1)+1)).to_dense()

    # Eliminar la ultima columna del retorno
    retorno = retorno[:,:-1]

    return retorno
