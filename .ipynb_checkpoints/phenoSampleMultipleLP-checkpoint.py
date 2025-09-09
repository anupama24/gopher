import argparse
import os, fnmatch
import sys
import math
from pathlib import Path

from pgen_reader import *
from functions import *
from utils import *
from rr_lp_utils import *
# from unbiased_lp_utils import *
#from tqdm.notebook import tqdm

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
    return parser.parse_args()

"""Estimate the variance of y"""
def est_var_y(y,n,eps):
    
    # n=y.shape[0]
    var = np.var(y)
    return var+ laplace_noise((y.max()-y.min())**2/n, eps, 1)   
    
def main():

    # print("here",flush=True)
    args = parse_args()
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
    # h2 = args.h2

    
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()),flush=True)
    
    print(f"Running multiple LP mechanism on phenotype: {pheno_name}",flush=True)

    simFile = args.geno_file
    pgen_file = f"{simFile}.pgen"
    pvar_file = f"{simFile}.pvar"
    psam_file = f"{simFile}.psam"
    
    pheno_df = load_phenotype(phenoFile,sample_subset=None)
    pheno_df.index = pheno_df.index.astype(str)
    
    arr =pheno_df[pheno_name].to_numpy(dtype=np.float32)
    Y_max=np.nanmax(arr)
    Y_min=np.nanmin(arr)
    
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
    n= Y_full.shape[0]
    
    id_set = set(pheno_df['IID'])
    # scoreFile=dest/"results/NonPriv_PRS_100_20000_h=0.8.sscore"
    
    Xmean_df =  pd.read_csv(scoreFile, sep='\t',index_col='#FID')
    # print(Xmean_df.head())
    Xmean_df = Xmean_df[Xmean_df['IID'].isin(id_set)]
    Xmean_df = Xmean_df.loc[pheno_df['IID']]  
    
    pheno_df['group'] = pd.qcut(pheno_df[pheno_name].rank(method='first'), q=10, labels=False) + 1

    # Group Y into bins and compute mean and variance
    y_grouped = pheno_df.groupby('group')[pheno_name]
    y_group_mean = y_grouped.mean()
    y_group_var = y_grouped.var()

    group_counts = y_grouped.count()
    group_max = Y_max
    group_min = Y_min
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
    # dp_group_var = dp_group_var.apply(lambda x: overall_var_dp if x <= 1e-12 else x)
    
    pheno_df['dp_group_mean'] = pheno_df['group'].map(dp_group_mean)
    pheno_df['dp_group_var'] = pheno_df['group'].map(dp_group_var)


    Xmean_df = Xmean_df.merge(
        pheno_df[['IID', 'dp_group_mean','dp_group_var']],
        on='IID',
        how='left'
    )
    

    Xmean_df['mean_score'] = Xmean_df['SCORE1_SUM'] * Xmean_df['dp_group_var']  + Xmean_df['dp_group_mean']
    
#     Xmean_df['PRS_percentile'] = pd.qcut(Xmean_df['SCORE1_SUM'], q=50, labels=False) + 1
#     # Xmean_df['PRS_percentile'] = pd.cut(Xmean_df['SCORE1_SUM'], bins=100, labels=False) + 1
    
#     # Compute mean per percentile group
#     # mean_by_percentile = Xmean_df.groupby('PRS_percentile')['SCORE1_SUM'].mean()
#     # # Then, map the mean values back to the original df
#     # Xmean_df['mean_score'] = Xmean_df['PRS_percentile'].map(mean_by_percentile)

#     # Xmean_df['mean_score']=Xmean_df['SCORE1_SUM']
#     # mask = (Xmean_df['PRS_percentile'] >= 20) & (Xmean_df['PRS_percentile'] <= 80)
#     # Xmean_df.loc[mask, 'mean_score'] = Xmean_df.loc[mask, 'PRS_percentile'].map(mean_by_percentile)

# #     overall_mean_dp = np.mean(Y_full) + laplace_noise((Y_full.max()-Y_full.min())/n, eps1*0.1,1)
# #     overall_mean_dp = overall_mean_dp.astype('float32')[0]
    
# #     var1 = est_var_y(Y_full,n,eps1*0.9)
# #     var1 = var1.astype('float32')[0]
    
#     Xmean_df['Y'] = Y_full

#     # Group Y by PRS_percentile and compute mean and variance
#     y_grouped = Xmean_df.groupby('PRS_percentile')['Y']
#     y_group_mean = y_grouped.mean()
#     y_group_var = y_grouped.var()

#     group_counts = y_grouped.count()
#     group_max = y_grouped.max()
#     group_min = y_grouped.min()
#     sensitivity = (group_max - group_min) / group_counts
#     # sensitivity = (Y_full.max() - Y_full.min()) / group_counts  # Per-group sensitivity for mean

#     # Add Laplace noise to group means and variances
#     dp_group_mean = y_group_mean + laplace_noise(sensitivity, eps1*0.1, size=len(y_group_mean))
#     sensitivity = ((group_max - group_min)**2) / group_counts
#     # sensitivity = ((Y_full.max() - Y_full.min())**2) / group_counts 
#     dp_group_var = y_group_var + laplace_noise(sensitivity, eps1*0.9, size=len(y_group_var))

#     Xmean_df['dp_group_var'] = Xmean_df['PRS_percentile'].map(dp_group_var)
#     Xmean_df['dp_group_mean'] = Xmean_df['PRS_percentile'].map(dp_group_mean)
    
    

#     # Global mean
#     overall_mean_dp = (group_counts * dp_group_mean).sum() / n
#     overall_var_dp = (group_counts * dp_group_var).sum() / n

#     between_group_var = ((group_counts * (dp_group_mean - overall_mean_dp)**2).sum()) / n

#     # Total variance
#     overall_var_dp = overall_var_dp + between_group_var
#     Xmean_df['mean_score'] = Xmean_df['SCORE1_SUM'] * Xmean_df['dp_group_var'] + Xmean_df['dp_group_mean']

    # mask = (Xmean_df['PRS_percentile'] >= 5) & (Xmean_df['PRS_percentile'] <= 45)
    # Xmean_df.loc[mask, 'mean_score'] = Xmean_df.loc[mask, 'SCORE1_SUM']*overall_var_dp + overall_mean_dp
    
    mean_dp = Xmean_df['mean_score'].to_numpy() 


    var1 = overall_var_dp
    var_dp = var1* np.ones(len(mean_dp),dtype=np.float32)
    
    print(Xmean_df.head())
    print(mean_dp[0:5],overall_mean_dp,mean_dp.shape)
    print(eps1,var1, var_dp.shape,var_dp[0:5])
    
    for eps_itr in eps_all:
        eps1 = 0.0
        multi_Y_priv = pool_rr_on_bins(Y_full,Y_full,bins,eps_itr,eps1,mean_dp,var_dp,seed)
        # np.save(out_path/ f"multiLP_sample_{sam}_Var_{h2}_NonPriv.npy",multi_Y_priv)
    
        outFile = f"MultiLP_Priv_10bins_{sam}_eps_{eps_itr}_{pheno_name}.txt"
        
        # save_pheno_gwas(multi_Y_priv,phenoFile,pheno_name,outFile)
        
        if os.path.isfile(outFile):
            pc_df= pd.read_csv(outFile, sep='\t', index_col=0)
            pc_df.index = pc_df.index.astype(str)
            is_in_set = pc_df['IID'].isin(id_set)
            row_numbers =  np.where(~is_in_set)[0]
            # pc_df = pc_df[pc_df.columns]
            # pc_df['IID'] = pc_df.index.copy()   
            print(pc_df.head(),flush=True)
            
        else:
            pheno_df = load_phenotype(phenoFile,sample_subset=None)
            pheno_df.index = pheno_df.index.astype(str)
            
            pc_df = pd.DataFrame(index=pheno_df.index.copy())
            pc_df.insert(0,'FID','')
            pc_df.insert(1,'IID','')
            pc_df['FID'] = pheno_df.index.copy()
            pc_df['IID'] = pheno_df.index.copy()   
            pc_df = pc_df.set_index("FID")
            pc_df[pheno_name] = np.nan
        
        # pheno_df.reset_index(drop=True, inplace=True)
        
        if samLPFile is not None:
            pc_df[pheno_name].update(lp_df[pheno_name])
            # pc_df[pheno_name] =  multi_Y_priv
            pc_df.loc[pheno_df.iloc[row_numbers].index.astype(str), pheno_name] = multi_Y_priv
        else:
            pc_df[pheno_name] = multi_Y_priv
            
        print(pc_df.columns,flush=True)
        print(len(pc_df),flush=True)
        
        pc_df.to_csv(outFile, sep="\t", na_rep='NA',index=True)

    print(f"GOPHER-LP mechanism for {pheno_name} done",flush=True)
#         print(f"Multiple LP for {eps_itr} done",flush=True)


if __name__ == '__main__':
    main()


