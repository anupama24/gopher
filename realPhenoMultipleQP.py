###################################################################
## Implementation of the QP method to generate private phenotype ##
## This file tests for different values of privacy budget        ##
###################################################################


import argparse
import os, fnmatch
import sys
import math
import subprocess


# from utils import *
from multi_qp_utils import *
from pgen_reader import *
from functions import *
import gc
from scipy.sparse.linalg import eigsh
# from lmm import *
#sys.path.append(".")

""" Configure command line arguments """
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--geno_file', default=None, help='genotype file tag')
    parser.add_argument('--pheno_file', default=None, help='path to phenotype file')
    parser.add_argument('--pheno', default="Sim_Y_100", type=str, help='Name of phenotype')
    parser.add_argument('-el', '--eps_list', default=None,help='delimited eps list input', type=str)
    parser.add_argument('--seed', default=37, type=int, help='Seed for random generator')
    parser.add_argument('--bins', default=100, type=int, help='Dicrete bins for continuous input')
    parser.add_argument('--eps', default=0.1, type=float, help='Privacy budget for prior')
    parser.add_argument('--score_file', default=None, help='path to GWAS score file')
    parser.add_argument('--lp_file', default=None, help='path to LP pheno file')
    parser.add_argument('--sam', default=10000, type=int, help='Sample size')
    # parser.add_argument('--h2', default=0.8, type=float, help='h2')
    parser.add_argument('--optTot', default=10e-6, type=int, help='Error tolerance for DCA')
    parser.add_argument('--itr', default=50, type=int, help='Number of iterations for DCA')
    
    return parser.parse_args()
    
"""Subsample data and store variant ids in file"""
def subsample(file_prefix,variantNum,pvarOutFile,seed):

    rng = np.random.RandomState(seed)
    
    if variantNum is not None:
        pvar_file = f"{file_prefix}.pvar"
        pvar_df = load_pvar(pvar_file)
        variant_ids = pvar_df['id'].tolist()
        M = len(variant_ids)
        if M < variantNum:
            v_idxs=list(range(0, M))
        else:
            v_ids = rng.randint(M-1,size=variantNum)
        
        new_pvar_df = pd.DataFrame(data=np.array(variant_ids,dtype=object)[v_ids], columns=['id'])
        new_pvar_df.to_csv(pvarOutFile, sep="\t", index=False,header=False)

"""Estimate the variance of y"""
def est_var_y(y,n,eps):
    
    # n=y.shape[0]
    var = np.var(y)
    return var+ laplace_noise((y.max()-y.min())**2/n, eps, 1)   
 

def main():

    args = parse_args()

    print("here",flush=True)
    phenoFile= args.pheno_file
    scoreFile = args.score_file
    samLPFile= args.lp_file

    if args.eps_list is not None:
        eps_all = [float(item) for item in args.eps_list.split(',')]
    else: 
        eps_all = [1.0,2.0,3.0,4.0,5.0,6.0]
        
    pheno_name = args.pheno
    seed = args.seed
    bins = args.bins
    eps1 = args.eps
    sam =  args.sam
    

    print(' '.join(f'{k}={v}' for k, v in vars(args).items()),flush=True)
    
    print(f"Running Multi QP mechanism on phenotype: {pheno_name}",flush=True)
    

    genoFile = args.geno_file
    pgen_file = f"{genoFile}.pgen"
    pvar_file = f"{genoFile}.pvar"
    psam_file = f"{genoFile}.psam"

    pheno_df = load_phenotype(phenoFile,sample_subset=None)
    pheno_df.index = pheno_df.index.astype(str)
    
    arr = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    Y_max = np.nanmax(arr)
    Y_min = np.nanmin(arr)
    
    if samLPFile is not None:
        lp_df = load_phenotype(samLPFile,sample_subset=None)
        lp_df.index = lp_df.index.astype(str)
        id_set = set(lp_df['IID'])

        is_in_set = pheno_df['IID'].isin(id_set)
        row_numbers =  np.where(~is_in_set)[0]
        print(len(row_numbers))

        pheno_df = pheno_df[~pheno_df['IID'].isin(id_set)]
    
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    print(f"Size of phenotype vector: {Y_full.shape}",flush=True)
    # n= Y_full.shape[0]
    
    id_set = set(pheno_df['IID'])
    
    X_df = load_genotypes(pgen_file, pvar_file, psam_file, np.int8)
    X_df = X_df.loc[pheno_df.index]

    print(X_df.shape)
    # X_snp = X_snp.iloc[row_numbers].to_numpy(dtype=np.int8)
    X_snp = X_df.to_numpy(dtype=np.int8)
    del X_df
    gc.collect()
    print(X_snp.shape)

    # Xstd= np.zeros(X_snp.shape,dtype=np.float32)
    X_snp = X_snp.astype(np.float32)  # convert once to save memory

    chunk_size=1000
    
    n_rows, n_cols = X_snp.shape
    # standardized_array = np.zeros_like(X,dtype=np.float32)

    for col_start in range(0, n_cols, chunk_size):
            
            col_end = min(col_start + chunk_size, n_cols)
            chunk = X_snp[:, col_start:col_end]
            mean = np.mean(chunk, axis=0)
            std = np.std(chunk, axis=0)
            # mean = np.mean(X_snp[:, col_start:col_end], axis=0)
            # std = np.std(X_snp[:, col_start:col_end], axis=0)
            std[std == 0] = 1.0
            for row_start in range(0, n_rows, chunk_size):
                row_end = min(row_start + chunk_size, n_rows)
            
                # X_chunk = X[row_start:row_end, col_start:col_end]
    
                # Standardize the chunk
                X_snp[row_start:row_end, col_start:col_end] = ((chunk[row_start:row_end] - mean) / std)

    # del X_snp

    # del X
    gc.collect()
    # print(Xstd.shape,Xstd.dtype,flush=True)
    
    print(f"Standardized array: {X_snp.shape},{X_snp.dtype}",flush=True)
    Y = Y_full[~np.isnan(Y_full)]
    n_full = Y_full.shape[0]
    n= Y.shape[0]
    
    # scoreFile=dest/"results/NonPriv_PRS_100_20000_h=0.8.sscore"
    
    score_df =  pd.read_csv(scoreFile, sep='\t',index_col='#FID')
    # print(score_df.head())
    score_df = score_df[score_df['IID'].isin(id_set)]
    score_df = score_df.loc[pheno_df['IID']]  
    
    pheno_df['group'] = pd.qcut(pheno_df[pheno_name].rank(method='first'), q=10, labels=False) + 1

    # Group Y into bins and compute mean and variance
    y_grouped = pheno_df.groupby('group')[pheno_name]
    y_group_mean = y_grouped.mean()
    y_group_var = y_grouped.var()

    group_counts = y_grouped.count()
    group_max = Y_max.max()
    group_min = Y_min.min()
    sensitivity = (group_max - group_min) / group_counts

    # Add Laplace noise
    dp_group_mean = y_group_mean + laplace_noise(sensitivity, eps1*0.1, size=len(y_group_mean))
    sensitivity = ((group_max - group_min)**2) / n
    dp_group_var = y_group_var + np.abs(laplace_noise(sensitivity, eps1*0.9, size=len(y_group_var)))

    # overall_var_dp = np.var(Y_full) + laplace_noise(sensitivity, eps1*0.9, 1)

    # Global mean
    overall_mean_dp = (group_counts * dp_group_mean).sum() / n
    overall_var_dp = (group_counts * dp_group_var).sum() / n
    between_group_var = ((group_counts * (dp_group_mean - overall_mean_dp)**2).sum()) / n
    # Total variance
    overall_var_dp = overall_var_dp + between_group_var
    print(overall_var_dp)
    
    # dp_group_var = dp_group_var.apply(lambda x: overall_var_dp if x <= 1e-12 else x)
    
    pheno_df['dp_group_mean'] = pheno_df['group'].map(dp_group_mean)
    pheno_df['dp_group_var'] = pheno_df['group'].map(dp_group_var)


    score_df = score_df.merge(
        pheno_df[['IID', 'dp_group_mean','dp_group_var']],
        on='IID',
        how='left'
    )

    score_df['mean_score'] = score_df['SCORE1_SUM'] * score_df['dp_group_var'] + score_df['dp_group_mean']
    mean_dp = score_df['mean_score'].values
    overall_var_dp = np.var(Y_full) + laplace_noise(sensitivity, eps1*0.9, 1)
    y_var = overall_var_dp.astype('float32')[0]
    # mean_dp = score_df['SCORE1_SUM'].values
    print(mean_dp.shape)

    # y_var, sigma2_g, sigma2_e = estimate_variance(Y,varLMMfile,n_full, eps1*0.9,"ev")
    # overall_mean_dp = np.mean(Y)
    # y_var = est_var_y(Y,n_full, eps1)
    # sigma2_g = h2 * y_var
    # sigma2_e = y_var - sigma2_g
    # y_var = y_var.astype('float32')[0]
    # sigma2_g = sigma2_g.astype('float32')[0]
    # sigma2_e = sigma2_e.astype('float32')[0]
    # print(f"Phenotype max: {Y.max()}, min: {Y.min()}, mean: {np.mean(Y)}, variance: {y_var}, sigma2_g: {sigma2_g},sigma2_e: {sigma2_e}", flush=True)

    # y_var = overall_var_dp
    print(f"Phenotype max: {Y.max()}, min: {Y.min()}, DP mean: {overall_mean_dp}, variance: {y_var}", flush=True)
    n,d = X_snp.shape
    
    num_chunks = 7
    # sample_pre_process(Xarr,Y_full,bins,mean_dp,var_dp,num_chunks,seed)
    Q1,Q2,B,Y_uniq,Y_hat,chunks = sample_pre_process(X_snp,Y,bins,mean_dp,y_var,num_chunks,seed,pheno_name)
    # Q1,Q2,B,Y_uniq,Y_hat,chunks = load_pre_process(Xstd,Y,bins,out_path,mean_dp,y_var,num_chunks,seed)

    sz_y = Y_uniq.shape[0]
    sz_yhat = Y_hat.shape[0]
    optTot = args.optTot #10e-6
    max_iter = args.itr #50

#     for eps_itr in eps_all:

#         eps_temp =eps_itr  - eps1
#         print(f"Running for eps={eps_itr}",flush=True)
#         sol = opt_dca(Q1,B,Q2, sz_y,sz_yhat, eps_temp ,max_iter,optTot,num_chunks)

#         priv_Y = save_QP_Yhat(sol,Y_full,Y_uniq,Y_hat,seed,chunks)
        
#         outFile = f"MultiQP_Priv_sample_{sam}_eps_{eps_itr}_{pheno_name}.txt"
        
#         if os.path.isfile(outFile):
#             pc_df= pd.read_csv(outFile, sep='\t', index_col=0)
#             pc_df.index = pc_df.index.astype(str)
#             is_in_set = pc_df['IID'].isin(id_set)
#             row_numbers =  np.where(~is_in_set)[0]
#             # pc_df = pc_df[pc_df.columns]
#             # pc_df['IID'] = pc_df.index.copy()   
#             print(pc_df.head(),flush=True)
            
#         else:
#             pheno_df = load_phenotype(phenoFile,sample_subset=None)
#             pheno_df.index = pheno_df.index.astype(str)
            
#             pc_df = pd.DataFrame(index=pheno_df.index.copy())
#             pc_df.insert(0,'FID','')
#             pc_df.insert(1,'IID','')
#             pc_df['FID'] = pheno_df.index.copy()
#             pc_df['IID'] = pheno_df.index.copy()   
#             pc_df = pc_df.set_index("FID")
#             pc_df[pheno_name] = np.nan
        
#         # pheno_df.reset_index(drop=True, inplace=True)
        
#         if samLPFile is not None:
#             pc_df[pheno_name].update(lp_df[pheno_name])
#             # pc_df[pheno_name] =  multi_Y_priv
#             pc_df.loc[pheno_df.iloc[row_numbers].index.astype(str), pheno_name] = priv_Y
#         else:
#             pc_df[pheno_name] = priv_Y
            
#         print(pc_df.columns,flush=True)
#         print(len(pc_df),flush=True)
        
#         pc_df.to_csv(outFile, sep="\t", na_rep='NA',index=True)
#         print(f"Multiple QP for {eps_itr} done",flush=True)


#     print(f"GOPHER-QP mechanism for {pheno_name} done",flush=True)


	 

if __name__ == '__main__':
    main()


# sol = opt_dca((1.0/np.square(N)*y_var)*A, (1.0/np.square(N)*y_var)*B,(1.0/np.square(N)*y_var)*E,sz_y,sz_yhat, eps2,max_iter,optTot)
