import sys
import math
import subprocess

from utils import *
import math
import os
import scipy.stats as st 
import multiprocessing as mp
from cvxopt import solvers, matrix, spmatrix, mul,spdiag
from collections import Counter
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
# import torch

# from numba import njit,prange

def sample_pre_process(Xarr,Y_full,bins,mean_dp,var_dp,num_chunks,seed,pheno):
	
    n,d = Xarr.shape
    var = var_dp*np.ones(n,dtype=np.float32)

    Y_uniq, prob_y = prob_dist_Y(Y_full,bins,mean_dp,var) #np.repeat(y_var,n))

    print(f"Original Y: # of bins={bins} and # of unique values={Y_uniq.shape}", flush=True)

    Y_hat,diff_yy = get_diff_YY(Y_full,Y_uniq,bins,mean_dp,var)
    print(f"Y_hat size: {Y_hat.shape}, dimension of Y-Y_hat diff matrix: {diff_yy.shape}", flush=True)
    
    sz_y,sz_yhat = diff_yy.shape

    rng = np.random.RandomState(seed)

    kmeans = KMeans(n_clusters=num_chunks, random_state=seed)  
    kmeans.fit(mean_dp.reshape(-1, 1))
    labels =  kmeans.labels_
    
    clustered_indices = {i: np.where(labels == i)[0] for i in range(np.max(labels) + 1)}
    # for cluster_id, indices in clustered_indices.items():
    #     print(f"Cluster {cluster_id} has {len(indices)} elements.")
    chunks = [clustered_indices[i] for i in range(len(clustered_indices))]
    
    print(f'Number of clusters: {len(chunks)}', flush=True)

    K = computeGRM(Xarr)
    
    print(f'GRM size : {K.shape}', flush=True)
    diag = np.diag(K)

    # Zero out near-zero values
    threshold = 1e-6
    K[np.abs(K) < threshold] = 0.0

    
    B = []
    A = []
    block_matrix = [[None for _ in range(num_chunks)] for _ in range(num_chunks)]

    for i in range(num_chunks):
        for j in range(i, num_chunks):             # Only compute upper triangle + diagonal
            ixgrid = np.ix_(chunks[i],chunks[j])
            di = diag[chunks[i]]  # shape: (ni,)
            dj = diag[chunks[j]]  # shape: (nj,)

            # Normalize Ki → Ycorr
            Ycorr = K[ixgrid] / np.sqrt(di)[:, None]
            Ycorr = Ycorr / np.sqrt(dj)[None, :]
            Ycorr = np.clip(Ycorr, -1.0, 1.0)
            zero_diag = (i == j)                  # diagonal chunk, pass zero_diag=True
            Aij = compute_A_for_chunk(K[ixgrid], Ycorr, diff_yy, Y_uniq, var_dp,
                                                  mean_dp[chunks[i]], mean_dp[chunks[j]], d, zero_diag)
            # if i == j:
            #     # diagonal chunk, pass zero_diag=True
            #     A_temp.append(compute_A_for_chunk(K[ixgrid], Ycorr, diff_yy, Y_uniq, var_dp,
            #                                       mean_dp[chunks[i]], mean_dp[chunks[j]], d, zero_diag=True))
            # else:
            #     # off-diagonal chunk, zero_diag=False
            #     A_temp.append(compute_A_for_chunk(K[ixgrid], Ycorr, diff_yy, Y_uniq, var_dp,
            #                                       mean_dp[chunks[i]], mean_dp[chunks[j]], d, zero_diag=False))

            block_matrix[i][j] = Aij

            if i != j:
                block_matrix[j][i] = Aij.T 
       
        # A.append(np.hstack(A_temp))
        B.append(compute_B_for_chunk(Xarr[chunks[i],:],prob_y[chunks[i],:],diff_yy))

    # A = np.concatenate(A)
    A_temp = [np.hstack(row) for row in block_matrix]
    A = np.vstack(A_temp)
    B = np.concatenate(B)

    A = A/(n**2 *var[0])
    B = B/(n**2 *var[0])
    
    print(f"B shape: {B.shape}, A shape: {A.shape}")
    np.save(f"A_{A.shape}_{pheno}",A)
    np.save(f"B_{B.shape}_{pheno}",B)

    eigval, eigvec = eigsh((2.0)*A , k=1, which='SA')
    rho= max(-1.0*min(eigval.real),0) + 10**-3                                #np.abs(min(eigval.real))*2

    print(min(eigval.real),rho, flush=True)
    Q2 = rho* np.eye(A.shape[0],dtype=np.float32)
    Q1 = (2.0)*A + Q2

    print(np.min(Q1),np.max(Q1),flush=True)
    return Q1,Q2,B,Y_uniq,Y_hat,chunks
    

"""Estimate private probability distribution using Normal distribution. For continuous Y, a discrete subset is created of size bins"""
def prob_dist_Y(Yarr,bins,mean,variance):

    sd = np.sqrt(variance[0]) 
    low,up = st.t.interval(0.99, df=len(Yarr)-1, loc=np.mean(mean),  scale=sd)
    low = max(Yarr.min(),low)
    up = min(Yarr.max(),up)
    
    # low = Yarr.min()
    # up = Yarr.max()
    print(low,up,flush=True)
    bin_edges = np.linspace(low,up,num=bins+1)
    Y_uniq = (bin_edges[:-1] + bin_edges[1:]) / 2

    n = len(variance)
    prob_y = np.zeros((n,len(Y_uniq)),dtype=np.float32)
    inter = (Y_uniq.max()-Y_uniq.min())/(len(Y_uniq)-1)

    for i in range(n):
        mean_dp = mean[i] 
        var_dp = variance[i]
        prob = stats.norm(mean_dp,np.sqrt(var_dp))
        prob_y[i,:] = prob.pdf(Y_uniq)*inter
        prob_y[i,:] = prob_y[i,:]/np.sum(prob_y[i,:])
    print(prob_y.shape)
    return Y_uniq,prob_y


"""Creates set Y_hat which is an array of "bins" values evenly spaced between min and max of Yarr. Computes Y_uniq-Y_hat matrix"""
def get_diff_YY(Yarr,Y_uniq,bins,mean,var=None):

    # if var is not None:
    #     sd = np.sqrt(var)
    # else:
    #     sd = (Yarr.min()-Yarr.max())/4.0
    sd = np.sqrt(var[0]) 
    low,up = st.t.interval(0.99, df=len(Yarr)-1, loc=np.mean(mean),  scale=sd)
    low = max(Yarr.min(),low)
    up = min(Yarr.max(),up)
    # low = Yarr.min()
    # up = Yarr.max()
    print(low,up,flush=True)
    
    bin_edges = np.linspace(low,up,num=bins+1)
    Y_hat = (bin_edges[:-1] + bin_edges[1:]) / 2
    d_yy = np.zeros((len(Y_uniq),len(Y_hat)),dtype=np.float32)
    
    for i,y in enumerate(Y_uniq):
        d_yy[i,:] =  y - Y_hat
    
    return Y_hat,d_yy

def computeGRM(Xstd):
    n, d = Xstd.shape
    chunk_size = 50000
    K = np.zeros((n, n), dtype=np.float32)

    for i in range(0, d, chunk_size):
        end = min(i + chunk_size, d)
        X_chunk = Xstd[:, i:end].astype(np.float32)  # Ensure float32 to save memory
        K += X_chunk @ X_chunk.T  # Efficient NumPy matrix multiplication

    # Optional normalization
    # K /= d

    return K


"""Create QP objective matrices assuming independence of array elements"""
def cov_obj_mat(Xstd,prob_xy,d_yy):
    
    sz_y,sz_yhat = d_yy.shape
    n,d = Xstd.shape
    print(f"Xstd shape: {Xstd.shape}")

    B = np.sum(np.square(Xstd),axis=1)
    C = (Xstd.T @ prob_xy) #  m x n  @  n x |y|
    C = C.T @ C            # |y| x m @  m x |y|
    E = np.zeros((sz_y,sz_y))
    for i in range(Xstd.shape[0]):
        E+= B[i]* (prob_xy[i,:,None] @ prob_xy[i,None,:])

    C = C-E
    C = np.repeat(np.repeat(C.T,repeats=sz_yhat, axis=1), repeats=sz_yhat, axis=0)
    
    d_vec = np.ravel(d_yy)
    D = d_vec[:,np.newaxis] @ d_vec[:,np.newaxis].T
    
    A = C * D # element-wise
    
    return A

def compute_A_for_chunk(K,Ycorr,d_yy,Y_uniq,var,mean_i,mean_j,d,zero_diag):
    
    sz_y,sz_yhat = d_yy.shape
    ni,nj = Ycorr.shape
    inter = (Y_uniq.max()-Y_uniq.min())/(len(Y_uniq)-1)
    
    C = np.zeros((sz_y,sz_yhat),dtype=np.float32)
    U = np.repeat(Y_uniq,sz_y).reshape(sz_y,sz_y)
    print(f"Created K matrix: {K.shape}, Ycorr: {Ycorr.shape}",flush=True)
    
    print(mp.cpu_count(),flush=True)
    pool = mp.Pool(processes=mp.cpu_count()-5,maxtasksperchild=5)
    
    args_list = [
        (i, K[i, :], Ycorr[i, :], U, mean_i[i], mean_j, var, inter,zero_diag) 
        for i in range(ni)
    ]
    # results = pool.map(compute_partial_C_for_row,[(i,K[i,:],Ycorr[i,:],U,mean_i[i],mean_j,var,inter) for i in range(ni)])
    results = pool.map(compute_partial_C_for_row, args_list)
    
    pool.close()
    pool.join()
    
    C = sum(results)
    # C = C/(n**2 *var[0])
    
    C = np.repeat(np.repeat(C.T,repeats=sz_yhat, axis=1), repeats=sz_yhat, axis=0)
    d_vec = np.ravel(d_yy)
    D = d_vec[:,np.newaxis] @ d_vec[:,np.newaxis].T
    A = C * D # element-wise

    return A
    
    
def compute_partial_C_for_row(args):
    i, K_row, corr_ij, U, mean_i, mean_j, var, inter, zero_diag = args
    # i,K,corr_ij,U,mean_x,mean_y,var,inter = args

    sz_y,sz_yhat = U.shape
    C = np.zeros((sz_y,sz_yhat),dtype=np.float32)
    if zero_diag:
        if i < len(K_row):
            K_row = K_row.copy()  
            K_row[i] = 0.0 

    for j in range(len(K_row)): 
        if K_row[j] == 0.0:
            continue
        H = method2(corr_ij[j],var,mean_i,mean_j[j],U,inter)
        # H = joint_prob_dist(corr_ij[j],U,var,mean_i,mean_j[j],inter)
        C += (K_row[j] * H)
    
    return C


def method2(corr,var,meanx,meany,U,inter):
    
    loc1 = meanx + corr * (U-meany).T
    scale1 = (1 - np.power(corr, 2)) * var
    denom1 = np.sqrt(2 * np.pi * scale1)
    num1 = np.exp(-((U - loc1) ** 2) / (2 * scale1))
    H = (num1 / denom1) * inter
    
    denom2 = np.sqrt(2 * np.pi * var)
    num2 = np.exp(-((U-meany).T ** 2) / (2 * var))
    G = (num2 / denom2) * inter
    
    return H * G

# def method2(corr,varx,vary,meanx,meany,U,inter):
#     loc1 = meanx + np.sqrt(varx / vary) * corr * (U-meany).T
#     scale1 = (1 - np.power(corr, 2)) * varx
#     denom1 = np.sqrt(2 * np.pi * scale1)
#     num1 = np.exp(-((U - loc1) ** 2) / (2 * scale1))
#     H = (num1 / denom1) * inter
    
#     denom2 = np.sqrt(2 * np.pi * vary)
#     num2 = np.exp(-((U-meany).T ** 2) / (2 * vary))
#     G = (num2 / denom2) * inter
    
#     return H * G



def compute_B_for_chunk(Xstd,prob_y,d_yy):

    sz_y,sz_yhat = d_yy.shape
    
    B = np.sum(np.square(Xstd),axis=1)
    B = B.T @ prob_y            # 1 x n_r @  n_r x |y|
    B = np.repeat(B,repeats=sz_yhat)
    d_vec = np.ravel(d_yy)
    D = d_vec * d_vec
    B = B * D 

    return B

"""Solve given QP optimization using DCA and cvxopt solver"""
def opt_dca(A,B,E,sz_y,sz_yhat,eps,max_iter,optTol,num_chunks):

    nvars = sz_y * sz_yhat
    tot_sz = nvars * num_chunks
    Aopt,bopt,Gopt,hopt = create_const_mat(nvars,sz_y,sz_yhat,num_chunks,eps)

    scale = np.linalg.norm(B,2)
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    qopt = matrix(B)  # Check that this is 1 x m^2
    
    print(f"here {scale}",flush=True)

    print(np.min(A),np.max(A),A.dtype,eps,flush=True)
    
    solvers.options['show_progress'] = True
    
    sol = solvers.lp(qopt,Gopt,hopt,Aopt,bopt)
    if sol['status'] == "optimal":
        prevW = np.array(sol['x'])
    else:
        prevW = np.zeros((nvars*num_chunks,1),dtype=np.float32)

    Popt = matrix(A)
    for itr in range(max_iter):
        H =  (E @ prevW )
        newB = B[:,None] - H
        qopt = matrix(np.ravel(newB))
        print("Popt max:", np.max(Popt), "min:", np.min(Popt),flush=True)
        print("qopt max:", np.max(qopt), "min:", np.min(qopt))
        print("Gopt shape:", Gopt.size,flush=True)
        
        # solvers.options['show_progress'] = True
        
        sol = solvers.qp(Popt,qopt,Gopt,hopt,Aopt,bopt)
        
        if np.linalg.norm(prevW,2) >1 :
            err =np.linalg.norm(sol['x']-prevW,2)**2/np.linalg.norm(prevW,2)**2
        else: 
            err =np.linalg.norm(sol['x']-prevW,2)**2 
        # err =np.linalg.norm(sol['x']-prevW,2)/(1+np.linalg.norm(prevW,2))
        prev_fval = 0.5* (prevW.T @ (A-E)@ prevW) + B.T @ prevW
        fval = 0.5* (sol['x'].T @ (A-E)@ sol['x']) + B.T @ sol['x']
        
        print(f"Iteration: {itr+1}, Iterate difference: {err}",flush=True)
        print(f"Solution: Status - {sol['status']}, Primal obj value- {sol['primal objective']}",flush=True)
        print(f"Function minimum: {fval}",flush=True)
        
        if (err<=optTol or np.abs(fval-prev_fval) <= optTol ):
            break
        prevW = np.array(sol['x'])
    
    return sol

"""Returns the constraint matrices required for the QP as per cvxopt solver"""
def create_const_mat(nvars,sz_y,sz_yhat,num_chunks,eps):

    Aopt = spmatrix(np.ones(nvars*num_chunks), np.repeat(np.arange(sz_y*num_chunks),sz_yhat), np.arange(nvars*num_chunks), (sz_y*num_chunks, nvars*num_chunks))
    bopt = matrix(1.0,(sz_y*num_chunks,1))
    
    print(f"Rank of A: {np.linalg.matrix_rank(np.array(matrix(Aopt)))}")

    tot_sz = sz_y*nvars*num_chunks
    h = matrix(0.0, (tot_sz,1)) 

    val1 = np.ones(tot_sz)
    row1 = np.arange(tot_sz)
    col1 = np.repeat(np.arange(nvars*num_chunks), sz_y)
    val2 = -np.exp(eps) * np.ones(tot_sz)
    col2 = np.tile(np.ravel(np.reshape(np.arange(nvars), (sz_y,sz_yhat)).transpose()), sz_y)
    temp = col2
    for i in range(num_chunks-1):
        col2 = np.concatenate((col2,temp+(i+1)*nvars))
    # np.tile(np.ravel(np.reshape(np.arange(nvars*3), (sz_y*3,sz_yhat)).transpose()),sz_y*3)
    
    idx = col1 == col2
    val1[idx] = -1
    val2[idx] = 0

    G = spmatrix(val1, row1, col1, (tot_sz, nvars*num_chunks)) + spmatrix(val2, row1, col2, (tot_sz, nvars*num_chunks)) 

    # print(G.size)
    # identity_matrix = spmatrix(1.0, range(G.size[0]), np.repeat(range(G.size[1]),sz_y))  
    # G = G + 1e-6 * identity_matrix
    # print(f"Rank of G: {np.linalg.matrix_rank(np.array(G))}")
    # np.linalg.matrix_rank(G_np)
    print(Aopt.size,bopt.size,G.size,h.size,flush=True)
    return Aopt,bopt,G,h

"""Finds the mapping between Y_uniq and Y_hat based on sol, and saves the numpy array in file"""
def save_QP_Yhat(sol,Y_full,Y_uniq,Y_hat,seed,chunks):

    num_chunks = len(chunks)
    sz_y = Y_uniq.shape[0]
    sz_yhat = Y_hat.shape[0]
    tot_sz = sz_y*num_chunks
    
    # W = sol 
    W = np.array(sol['x'])
    W = np.reshape(W,(tot_sz,sz_yhat))
    rng = np.random.RandomState(seed)

    n= Y_full.shape[0]
    priv_y = np.zeros(n,dtype=np.float32)

    for i, chunk in enumerate(chunks):
    
        mat = W[i*sz_y:(i+1)*sz_y,:]
        bin_indices = find_indices(Y_full[chunk], Y_uniq)

        temp = Y_full[chunk]
        for j in range(len(chunk)):
            if np.isnan(temp[j]):
                priv_y[chunk[j]] = np.nan
            else:
                idx = bin_indices[j]
                priv_y[chunk[j]] = Y_hat[rng.choice(sz_yhat,1,p=mat[idx])[0]]
    
    print(f"Y_hat: unique values: {np.unique(priv_y)}")
    
    
    return priv_y
    
    
"""Run preprocessing steps for the QP i.e. estimare prior dist, load saved matrices for objective. """
# (Xarr,Y_full,bins,dest,mean_dp,var_dp,num_chunks,seed)
def load_pre_process(Y_full,bins,mean_dp,var_dp,num_chunks,seed,pheno):

    	
    n, = Y_full.shape
    var = var_dp*np.ones(n,dtype=np.float32)

    Y_uniq, prob_y = prob_dist_Y(Y_full,bins,mean_dp,var) #np.repeat(y_var,n))

    print(f"Original Y: # of bins={bins} and # of unique values={Y_uniq.shape}", flush=True)

    Y_hat,diff_yy = get_diff_YY(Y_full,Y_uniq,bins,mean_dp,var)
    print(f"Y_hat size: {Y_hat.shape}, dimension of Y-Y_hat diff matrix: {diff_yy.shape}", flush=True)
    
    sz_y,sz_yhat = diff_yy.shape

    rng = np.random.RandomState(seed)

    # num_chunks = 7 
    kmeans = KMeans(n_clusters=num_chunks, random_state=seed)  
    kmeans.fit(mean_dp.reshape(-1, 1))
    labels =  kmeans.labels_
    
    clustered_indices = {i: np.where(labels == i)[0] for i in range(np.max(labels) + 1)}
    # for cluster_id, indices in clustered_indices.items():
    #     print(f"Cluster {cluster_id} has {len(indices)} elements.")
    chunks = [clustered_indices[i] for i in range(len(clustered_indices))]
    
    print(f'Number of clusters: {len(chunks)}', flush=True)
    
#     dim = num_chunks * sz_y**2
#     A = np.load(f"A_({dim}, {dim})_{pheno}.npy")
#     B = np.load(f"B_({dim},)_{pheno}.npy")

#     eigval, eigvec = eigsh((2.0)*A , k=1, which='SA')
#     rho= max(-1.0*min(eigval.real),0) + 10**-3                                #np.abs(min(eigval.real))*2

#     print(min(eigval.real),rho, flush=True)
#     Q2 = rho* np.eye(A.shape[0],dtype=np.float32)
#     Q1 = (2.0)*A + Q2

#     print(np.min(Q1),np.max(Q1),flush=True)
    # np.save(dest/f"Q1_({dim},{dim}).npy",Q1)
    return Y_uniq,Y_hat,chunks
 

def find_indices(arr, bin_centers):
    
    diff_matrix = np.abs(arr[:, None] - bin_centers[None, :])
    indices = np.argmin(diff_matrix, axis=1)
    return indices


"""Estimate joint probability distribution of bivariate normal distribution. """
def joint_prob_dist(corr,U,var,meanx,meany,inter):
    
#     num = np.square(u)/var_x + np.square(v)/var_y - ((2*corr)*u*v/(np.sqrt(var_x*var_y)))
    denom = 2*np.pi*var*np.sqrt(1-corr**2)
    num = (U-meanx)**2 + (U.T-meany)**2 - (2*corr*U*U.T)
    num = np.exp(-num/(2*var*(1-corr**2)))
    pdf = num /denom
    return pdf * inter**2 