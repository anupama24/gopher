import os
import sys
import math
import scipy.stats as st
import multiprocessing as mp
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.special import logsumexp
from cvxopt import solvers, matrix, spmatrix

# --- Local imports ---
from utils import *
from rr_lp_utils import *

def sample_pre_process(Xarr, Y_full, bins, mean_dp, var_dp, overall_mean_dp,overall_var,num_chunks, seed, pheno):
    """
    Prepare all pre-processing steps and matrices for the Multi-QP mechanism.
        1. Compute privatized phenotype probability distribution (Y_uniq, prob_y)
        2. Compute genetic relationship matrix (GRM)
        3. Build quadratic programming objective matrices A and B

    """
    print(f"Preprocessing for phenotype '{pheno}'", flush=True)
    n, d = Xarr.shape
    
    # --- Step 1: Estimate phenotype distribution ---
    Y_uniq, prob_y = prob_dist_Y(Y_full, bins, mean_dp, var_dp,overall_mean_dp,overall_var)
    print(f"Original Y: bins={bins}, unique={Y_uniq.shape}", flush=True)

    Y_hat, diff_yy = get_diff_YY(Y_full, Y_uniq, bins,overall_mean_dp,overall_var)
    print(f"Y_hat size={Y_hat.shape}, diff_yy={diff_yy.shape}", flush=True)
    
    sz_y,sz_yhat = diff_yy.shape

    rng = np.random.RandomState(seed)

    # --- Step 2: Cluster samples into phenotype groups ---
    kmeans = KMeans(n_clusters=num_chunks, random_state=seed)
    kmeans.fit(mean_dp.reshape(-1, 1))
    labels = kmeans.labels_
    clustered_indices = {i: np.where(labels == i)[0] for i in range(np.max(labels) + 1)}
    chunks = [clustered_indices[i] for i in range(len(clustered_indices))]
    print(f"Formed {len(chunks)} clusters", flush=True)
    
    # --- Step 3: Compute Genetic Relationship Matrix (GRM) ---
    K = computeGRM(Xarr)
    print(f"GRM: min={np.min(K):.4f}, max={np.max(K):.4f}, shape={K.shape}", flush=True)

    # --- Step 4: Construct block matrices A and B ---
    block_matrix = [[None for _ in range(num_chunks)] for _ in range(num_chunks)]
    B_blocks = []

    for i in range(num_chunks):
        for j in range(i, num_chunks):
            ixgrid = np.ix_(chunks[i], chunks[j])
            Aij = compute_A_for_chunk(
                K[ixgrid], diff_yy, Y_uniq, var_dp[chunks[i]], var_dp[chunks[j]],
                mean_dp[chunks[i]], mean_dp[chunks[j]],d, zero_diag=(i == j))
            block_matrix[i][j] = Aij
            if i != j:
                block_matrix[j][i] = Aij.T

        B_blocks.append(compute_B_for_chunk(Xarr[chunks[i], :],
                                            prob_y[chunks[i], :], diff_yy))



    # Stack block matrices
    A = np.vstack([np.hstack(row) for row in block_matrix])
    B = np.concatenate(B_blocks)

    A /= (n**2 * overall_var)
    B /= (n**2 * overall_var)

    print(f"A shape={A.shape}, B shape={B.shape}", flush=True)
    # np.save(f"A_{A.shape}_{pheno}", A)
    # np.save(f"B_{B.shape}_{pheno}", B)

    # --- Step 5: Stabilize A matrix ---
    # eigval, _ = eigsh(2.0 * A, k=1, which='SA')
    eigval, _ = eigh(2.0 * A, subset_by_index=[0, 0])  # smallest eigenvalue only
    eigval = float(eigval[0])
    rho = max(-min(eigval.real), 0) + 1e-3
    Q2 = rho * np.eye(A.shape[0], dtype=np.float32)
    Q1 = 2.0 * A + Q2
    print(f"Stabilization rho={rho:.5f}, Q1 range=({np.min(Q1):.5e}, {np.max(Q1):.5e})", flush=True)

    return Q1, Q2, B, Y_uniq, Y_hat, chunks

def computeGRM(Xstd):
    """Compute the Genetic Relationship Matrix (GRM) from standardized genotypes."""
    n, d = Xstd.shape
    K = np.zeros((n, n))
    chunk_size = 50000
    print(f"Computing GRM using chunks of size {chunk_size}", flush=True)

    for i in range(0, d, chunk_size):
        end = min(i + chunk_size, d)
        X_chunk = Xstd[:, i:end]
        K += X_chunk @ X_chunk.T
        
    # Optional normalization
    K /= d
    return K


def prob_dist_Y(Yarr, bins, mean, variance,overall_mean_dp,overall_var):
    """Estimate private probability distribution for continuous phenotype Y."""
    sd = np.sqrt(overall_var)
    low, up = st.t.interval(0.99, df=len(Yarr)-1, loc=overall_mean_dp, scale=sd)
    low, up = max(Yarr.min(), low), min(Yarr.max(), up)
    print(f"Clipped range: [{low:.3f}, {up:.3f}]", flush=True)
    
    bin_edges = np.linspace(low, up, num=bins+1)
    Y_uniq = (bin_edges[:-1] + bin_edges[1:]) / 2

    n = len(variance)
    prob_y = np.zeros((n, len(Y_uniq)), dtype=np.float32)
    inter = (Y_uniq.max() - Y_uniq.min()) / (len(Y_uniq) - 1)

    for i in range(n):
        dist = st.norm(mean[i], np.sqrt(variance[i]))
        prob_y[i, :] = dist.pdf(Y_uniq) * inter
        prob_y[i, :] /= np.sum(prob_y[i, :])
    return Y_uniq, prob_y


def get_diff_YY(Yarr, Y_uniq, bins, mean, var):
    """Construct matrix of differences between Y_uniq and discretized Y_hat."""
    sd = np.sqrt(var)
    low, up = st.t.interval(0.99, df=len(Yarr)-1, loc=mean, scale=sd)
    low, up = max(Yarr.min(), low), min(Yarr.max(), up)

    bin_edges = np.linspace(low, up, num=bins+1)
    Y_hat = (bin_edges[:-1] + bin_edges[1:]) / 2
    d_yy = np.array([[y - yhat for yhat in Y_hat] for y in Y_uniq], dtype=np.float32)
    return Y_hat, d_yy


def compute_A_for_chunk(K, d_yy, Y_uniq, var_i,var_j, mean_i, mean_j, d, zero_diag):
    """
    Compute a block of the A matrix corresponding to one (i,j) chunk pair.

    """
    sz_y, sz_yhat = d_yy.shape
    ni, nj = K.shape
    inter = (Y_uniq.max() - Y_uniq.min()) / (len(Y_uniq) - 1)
    Ycorr = np.clip(K.copy(), -1.0, 1.0)
    K *= d

    U = np.repeat(Y_uniq, sz_y).reshape(sz_y, sz_y)
    print(f"Computing A block {ni}×{nj}", flush=True)
    
    pool = mp.Pool(processes=mp.cpu_count()-4,maxtasksperchild=5)
    
    args_list = [(i, K[i, :], Ycorr[i, :], U, mean_i[i], mean_j, var_i[i],var_j, inter, zero_diag)
                     for i in range(ni)]
    
    results = pool.map(compute_partial_C_for_row, args_list)
    
    pool.close()
    pool.join()
    
    C = sum(results)
        
    C = np.repeat(np.repeat(C.T, sz_yhat, axis=1), sz_yhat, axis=0)
    d_vec = np.ravel(d_yy)
    D = d_vec[:,np.newaxis] @ d_vec[:,np.newaxis].T

    A = C * D
    return A
    

def compute_partial_C_for_row(args):
    """Helper for parallel A computation."""
    i, K_row, corr_ij, U, mean_i, mean_j, var_i,var_j, inter, zero_diag = args
    sz_y, sz_yhat = U.shape
    C = np.zeros((sz_y, sz_yhat), dtype=np.float32)

    for j in range(len(K_row)):
        if zero_diag and i == j:
            continue
        log_H = joint_log_prob_dist(corr_ij[j], U, var_i,var_j[j], mean_i, mean_j[j], inter)
        H = np.exp(log_H).astype(np.float32)
        C += K_row[j] * H
    return C


   
def compute_B_for_chunk(Xstd, prob_y, d_yy):
    """
    Compute block B_i for cluster i.
    """
    sz_y, sz_yhat = d_yy.shape
    B = np.sum(np.square(Xstd), axis=1)
    B = B.T @ prob_y            # 1 x n_r @  n_r x |y|

    B = np.repeat(B, repeats=sz_yhat)
    D = np.square(np.ravel(d_yy))
    return B * D

"""Estimate joint probability distribution of bivariate normal distribution. """
def joint_prob_dist(corr,U,var,meanx,meany,inter):
    
#     num = np.square(u)/var_x + np.square(v)/var_y - ((2*corr)*u*v/(np.sqrt(var_x*var_y)))
    denom = 2*np.pi*var*np.sqrt(1-corr**2)
    num = (U-meanx)**2 + (U.T-meany)**2 - (2*corr*U*U.T)
    num = np.exp(-num/(2*var*(1-corr**2)))
    pdf = num /denom
    return pdf * inter**2 

"""
Computes log of joint PDF of a bivariate normal distribution.
Returns log(PDF) + log(inter^2), for later log-space accumulation.
"""
def joint_log_prob_dist(corr, U, varx,vary, meanx, meany, inter):
    tol = 1e-10  # numerical stability
    dx = U - meanx
    dy = U.T - meany

    det = varx * vary * (1 - corr**2 + tol)
    exponent = (
        dx**2 / varx +
        dy**2 / vary -
        2 * corr * dx * dy / np.sqrt(varx * vary)
    )
    log_num = -0.5 * exponent
    log_denom = np.log(2 * np.pi) + 0.5 * np.log(det)
    log_pdf = log_num - log_denom + 2 * np.log(inter)

    return log_pdf

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
    
    os.environ["OPENBLAS_NUM_THREADS"] = "16"
    os.environ["OMP_NUM_THREADS"] = "16"

    print("Set OpenBLAS to:", os.environ.get("OPENBLAS_NUM_THREADS"),flush=True)
    
    for itr in range(max_iter):
        H = E @ prevW
        newB = B[:, None] - H
        qopt = matrix(np.ravel(newB))
        solvers.options['show_progress'] = False
    
        sol = solvers.qp(Popt, qopt, Gopt, hopt, Aopt, bopt)
        # if sol['status'] != 'optimal':
        #     print(f"Warning: Solver failed at iter {itr}")
            # break

        currW = np.array(sol['x'])
        err = np.linalg.norm(currW - prevW, 2)**2 / (np.linalg.norm(prevW, 2)**2 + 1e-10)
        fval = 0.5 * (currW.T @ (A - E) @ currW) + B.T @ currW

        print(f"Iteration {itr+1}: err={err:.3e}, obj={fval}",flush=True)

        prev_fval = 0.5* (prevW.T @ (A-E)@ prevW) + B.T @ prevW
        if (err <= optTol or np.abs(fval-prev_fval) <= optTol ):
            break

        prevW = currW
    
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
    """Find nearest bin indices for array elements."""
    diff_matrix = np.abs(arr[:, None] - bin_centers[None, :])
    return np.argmin(diff_matrix, axis=1)


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

