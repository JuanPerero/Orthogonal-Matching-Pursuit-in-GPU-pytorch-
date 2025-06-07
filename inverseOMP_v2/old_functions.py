import inverseOMP





def omp_v1(X, y, XTX, n_nonzero_coefs=None, tol=1e-2, inverse_cholesky=True):
    B = y.shape[1] # cantidad de señales, muestras o elementos de y
    DTX = y.T @ X  # Correlacion pero en dimension 17k x 128
    residuo = y.clone()
    sets = X.size(1)*y.new_ones(n_nonzero_coefs, B, dtype=torch.int64)
    D_mybest = y.new_zeros(B, n_nonzero_coefs, 1) # Construccion del diccionario de atomos selectos correlacionado con un atomo
    
    gamma = y.new_zeros(B,n_nonzero_coefs)
    Di =  y.new_zeros(B,n_nonzero_coefs,X.shape[0]) ## --
    
    L = y.new_zeros(B,n_nonzero_coefs,n_nonzero_coefs)
    L[:,0,0] = 1
    Linv = L.clone()

    bool_ctrl = torch.ones((B, X.shape[1]), dtype=torch.bool, device=y.device)
    end_n = torch.ones(B, dtype=torch.bool, device=y.device)

    for k in range(n_nonzero_coefs):
        correlation = X.T @ residuo
        sets[k] = correlation.abs().argmax(0)
        # Primero verifico si en la mascara estan activados los elementos seleccionados
        end_n = end_n.logical_and(bool_ctrl.gather(1, sets[k].unsqueeze(1)).squeeze())      
        # Marco la mascara de elementos seleccionados
        bool_ctrl.scatter_(1, sets[k].unsqueeze(1), False)
        if k:
            D_mybest[end_n, :k, 0] = torch.gather(XTX[sets[k]], 1, sets.T[end_n,:k])
            w =  torch.bmm(Linv[end_n,:k,:k], D_mybest[end_n,:k])                       ## D_best = XTX[sets[k, end_n].unsqueeze(1).expand(-1, k), sets.T[end_n, :k]]
            L[end_n,k,:k] = w[:,:,0]
            L[end_n,k, k] = torch.sqrt(1 - tc.bmm(w.permute(0,2,1),w).squeeze())
            inverseOMP.step_cholesky(L,Linv,k+1)
        Di[end_n, k, :] = torch.gather(X.T, 0, sets[k, end_n, None].expand(-1, X.shape[0]))
        gamma[end_n,:k+1] = torch.gather(DTX, 1, sets.T[end_n,:k+1]) 
        LTL = torch.bmm(Linv[end_n,:k+1,:k+1].transpose(1,2), Linv[end_n,:k+1,:k+1])            ##### Se puede mejorar por ser iterativo las operaciones Linv[:,:k+1].transpose(1,2), Linv[:,:k+1]
        gamma[end_n,:k+1,None] = torch.bmm(LTL, gamma[end_n,:k+1,None])                     ##### ACA creo que tampoco hace falta hacer tooooooda la multiplicacion
        residuo = y - torch.bmm(gamma[:,None,:k+1],Di[:,:k+1]).permute(1,2,0)[0]
    # X.T[None],sets[k, end_n, None].expand(-1, X.shape[0])
    del L
    del LTL
    del Linv
    del D_mybest
    del Di
    torch.cuda.empty_cache()

    indx = torch.arange(y.shape[1],device=X.device).repeat(n_nonzero_coefs,1).T.flatten()
    indx = torch.vstack((indx.to("cuda"),sets.T.flatten()))
    retorno = torch.sparse_coo_tensor(indx,gamma.flatten(),size=(y.shape[1],X.size(1)+1)).to_dense()

    #retorno = y.new_zeros((y.shape[1], X.size(1)+1))
    #retorno.scatter_(1, sets.T, gamma)

    # Eliminar la ultima columna del retorno
    retorno = retorno[:,:-1]

    return retorno






def omp_v3_testing(X, y, XTX=None, n_nonzero_coefs=None, tol=1e-2, inverse_cholesky=True):
    if XTX is None:
        XTX = X.T @ X    

    B = y.shape[1] # cantidad de señales, muestras o elementos de y
    DTX = y.T @ X  # Correlacion pero en dimension 17k x 128
    residuo = y.clone()
    sets = X.size(1)*y.new_ones(n_nonzero_coefs, B, dtype=torch.int64)
    gamma = y.new_zeros(B,n_nonzero_coefs)

    Laux = y.new_zeros(B,n_nonzero_coefs,n_nonzero_coefs)
    Laux[:,0,0] = 1

    Linv = y.new_zeros(B,n_nonzero_coefs,n_nonzero_coefs)
    Linv[:,0,0] = 1

    #D_mybest = y.new_zeros(B, n_nonzero_coefs, 1) # Construccion del diccionario de atomos selectos correlacionado con un atomo
    #Di =  y.new_zeros(B,n_nonzero_coefs,X.shape[0]) ## --
    

    bool_ctrl = torch.ones((B, X.shape[1]), dtype=torch.bool, device=y.device)
    end_n = torch.ones(B, dtype=torch.bool, device=y.device)

    for k in range(n_nonzero_coefs):
        correlation = X.T @ residuo
        sets[k] = correlation.abs().argmax(0)
        # Primero verifico si en la mascara estan activados los elementos seleccionados
        end_n = end_n.logical_and(bool_ctrl.gather(1, sets[k].unsqueeze(1)).squeeze())      
        # Marco la mascara de elementos seleccionados
        bool_ctrl.scatter_(1, sets[k].unsqueeze(1), False)
        if torch.any(~end_n):
            sets[k,~end_n] = 256
        if end_n.sum()==0:
            break
        if k:
            ###########
            #D_mybest[end_n, :k, 0] = torch.gather(XTX[sets[k]], 1, sets.T[end_n,:k])
            #w =  torch.bmm(Linv[end_n,:k,:k], D_mybest[end_n,:k])
            ## -----
            ## Dbest = XTX[sets[k, end_n].unsqueeze(1).expand(-1, k), sets.T[end_n, :k],None]
            w =  torch.bmm(Linv[end_n,:k,:k],XTX[sets[k, end_n].unsqueeze(1).expand(-1, k), sets.T[end_n, :k],None])
            ###########
            #L[end_n,k,:k] = w[:,:,0]
            #L[end_n,k, k] = torch.sqrt(1 - tc.bmm(w.permute(0,2,1),w).squeeze())
            L = torch.concatenate((w,torch.sqrt(1 - tc.bmm(w.permute(0,2,1),w))),1).squeeze(2)

            Laux[end_n,k,:k+1] = torch.concatenate((w,torch.sqrt(1 - tc.bmm(w.permute(0,2,1),w))),1).squeeze(2)
            #torch.linalg.inv(Laux[0,:k+1,:k+1])
            # L2 = torch.concatenate((w,torch.sqrt(1 - tc.bmm(w.permute(0,2,1),w))),1)
            
            #inverseOMP.step_cholesky(L,Linv,k+1)          
            inverseOMPV2.step_cholesky(L,Linv) # Esto no toma en cuenta el end_n, necesitaria implementar el retornoy asignarlo al Linv

            #inverseOMPV2.step_cholesky(L[:,:k+1,:k+1],Linv[:,:k,:k])

            #L2 = L[:,:k+1,:k+1][0,-1,:].unsqueeze(0).clone()
            #Linv2 = L[:,:k+1,:k+1][0,:,:].clone()

        gamma[end_n,:k+1] = torch.gather(DTX, 1, sets.T[end_n,:k+1]) 
        ####
        #LTL = torch.bmm(Linv[end_n,:k+1,:k+1].transpose(1,2), Linv[end_n,:k+1,:k+1])            ##### Se puede mejorar por ser iterativo las operaciones Linv[:,:k+1].transpose(1,2), Linv[:,:k+1]
        #gamma[end_n,:k+1,None] = torch.bmm(LTL, gamma[end_n,:k+1,None])                     ##### ACA creo que tampoco hace falta hacer tooooooda la multiplicacion
        
        gamma[end_n,:k+1,None] = torch.bmm(torch.bmm(Linv[end_n,:k+1,:k+1].transpose(1,2), Linv[end_n,:k+1,:k+1]), gamma[end_n,:k+1,None])              


        # X.T[None],sets[k, end_n, None].expand(-1, X.shape[0])
        # X.T[None].expand(y.shape[1],-1,-1)
        # sets.T[end_n,:k+1, None]  torch.Size([17511, 2, 1])
        ## sets.T[end_n,:k+1, None].expand(-1,-1, X.shape[0])
        ######## torch.gather(X.T[None].expand(end_n.sum(),-1,-1), 1, sets.T[end_n,:k+1, None].expand(-1,-1, X.shape[0]))
        ######## X.T[sets.T[end_n, :k+1], :] ## ALTERNATIVA - Para calcular Di - equivalente a Di[:,:k+1]
        
        
        ##Di[end_n, k, :] = torch.gather(X.T, 0, sets[k, end_n, None].expand(-1, X.shape[0]))
        ##residuo = y - torch.bmm(gamma[:,None,:k+1],Di[:,:k+1]).permute(1,2,0)[0]

        residuo[:,end_n] = y[:,end_n] - torch.bmm(gamma[end_n,None,:k+1],X.T[sets.T[end_n, :k+1], :]).permute(1,2,0)[0]
        
   
    del L
    #del LTL
    del Linv
    #del D_mybest
    #del Di
    torch.cuda.empty_cache()

    indx = torch.arange(y.shape[1],device=X.device).repeat(n_nonzero_coefs,1).T.flatten()
    indx = torch.vstack((indx.to("cuda"),sets.T.flatten()))
    retorno = torch.sparse_coo_tensor(indx,gamma.flatten(),size=(y.shape[1],X.size(1)+1)).to_dense()

    #retorno = y.new_zeros((y.shape[1], X.size(1)+1))
    #retorno.scatter_(1, sets.T, gamma)

    # Eliminar la ultima columna del retorno
    retorno = retorno[:,:-1]
    return retorno