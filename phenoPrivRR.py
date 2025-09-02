###################################################################
## Implementation of the RR method to generate private phenotype ##
## This file tests for different values of privacy budget        ##
###################################################################


import argparse
import os, fnmatch
import sys
import math

# from utils import *
from rr_lp_utils import *

from pgen_reader import *
from functions import *
#sys.path.append(".")

""" Configure command line arguments """
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--src', default=None, help='path to merged genotypes and phenotype folder')
    parser.add_argument('--dest', default=None, help='path to destination folder')
    parser.add_argument('--pheno_file', default=None, help='path to phenotype file')
    parser.add_argument('--pheno', default="LDL-direct", type=str, help='Name of phenotype')
    parser.add_argument('-el', '--eps_list', default=None,help='delimited eps list input', type=str)
    parser.add_argument('--seed', default=37, type=int, help='Seed for random generator')
    parser.add_argument('--bins', default=100, type=int, help='Dicrete bins for continuous input')
    # parser.add_argument('--h2', default=0.8, type=float, help='h2')
    # parser.add_argument('--cov_file', default=None, help='path to covariate file')
    # parser.add_argument('--cov_priv', default=1.0, type=float, help='total privacy budget for covariates')
    return parser.parse_args()
    

def privRRSaveCovData(covFile,eps,covOutFile,bins,seed):

    W_df = pd.read_csv(covFile, sep='\t', index_col=0)
    age_full  = W_df['AGE'].to_numpy(dtype=np.int32)
    age = age_full[~np.isnan(age_full)]
    # priv_age = np.zeros(age.shape[0],dtype=np.int32)
    
    sex_full = W_df['SEX'].to_numpy(dtype=np.int8)
    sex = sex_full[~np.isnan(sex_full)]
    # ids = np.where((np.isclose(sex_full,-9))| (np.isclose(sex_full,-1)))[0] #~np.isnan(Y)
    # mask = np.ones(sex_full.shape, bool)
    # mask[ids] = False
    # sex = sex_full[mask]
    # sex = sex[~np.isnan(sex)]
    priv_sex = np.zeros(sex.shape[0],dtype=np.int8)

    eps1 = (3.0/4.0) * eps
    eps2 = (1.0/4.0) * eps

    priv_age = RR(age_full,eps1,bins,seed,None,dtype=np.int32) 
    sex_new = rr_baseline(sex,3,eps2,seed)
    
    
    j=0
    for i in range(sex_full.shape[0]):
        if (np.isclose(sex_full[i],-9) or np.isclose(sex_full[i],-1) or np.isnan(sex_full[i])) or j>=sex.shape[0]:
            priv_sex[i] = np.nan
        else:
            priv_sex[i] = sex_new[j]
            j+=1
                
    savePrivCovData(covFile,priv_age,priv_sex,covOutFile)

def find_indices(arr, bin_centers):
    
    diff_matrix = np.abs(arr[:, None] - bin_centers[None, :])
    indices = np.argmin(diff_matrix, axis=1)
    return indices


"""Applies RR mechanism to phenotype Yarr. Saves the private phenotype in file named "outFile" """
def RR(Yarr,eps,bins,seed,outFile=None,dtype=None):

    Y=Yarr[~np.isnan(Yarr)]
    print(f"Max of phenotype: {Y.max()}, min: {Y.min()}", flush=True)
    
    # sd = (Y.min()-Y.max())/4.0
    # low,up = st.t.interval(0.99, df=len(Yarr)-1, loc=Yarr.mean(),  scale=sd)
    # low = max(Y.min(),low)
    # up = min(Y.max(),up)

    low = Yarr.min()
    up = Yarr.max()
    print(low,up)

    bin_edges = np.linspace(low,up,bins+1)
    Y_uniq = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(Y_uniq.shape)
    
    # Y_uniq = np.unique(np.linspace(low,up,num=bins+1,dtype=dtype)[:-1])
    print(f"eps: {eps}")
    print(f"Original Y: # of bins={bins} and # of unique values={Y_uniq.shape}",flush=True)

    sz_y = Y_uniq.shape[0]
    rng = np.random.RandomState(seed)
  
    n= Yarr.shape[0]
    priv_y = np.zeros(n,dtype=np.float32)

    # bin_indices = find_indices(Yarr, Y_uniq)
    # bin_indices[np.isnan(Yarr)] = -1
    
    for i in range(n):
        if np.isnan(Yarr[i]):
            priv_y[i] = np.nan
        else:
            idx = np.abs(Y_uniq-Yarr[i]).argmin()
            if idx == len(Y_uniq):
                idx=idx-1
            new_label = Y_uniq[idx]
            priv_y[i] =  compute_RR_bins(new_label,Y_uniq,eps,rng)
        
            # # print(Y_uniq[bin_indices[i]-1],Yarr[i],flush=True)
            # idx = bin_indices[i]
            # priv_y[i] = compute_RR_bins(Y_uniq[idx],Y_uniq,eps,rng)
                
    if outFile is not None:
        np.save(outFile,priv_y)
    
    return priv_y

    

def main():
    
    args = parse_args()

    if args.dest is None:
        dest = Path.cwd()
    else:
        dest=Path(args.dest)
        
    
    ##Path to output folder
    # out_path= dest/"results/RR"
    out_path= dest
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # if args.pheno_file is None:
    #     phenoFile =  dest / "ukb_simulate_pheno_lp.txt"
    # else:
    #     phenoFile= dest / args.pheno_file
    
    phenoFile= args.pheno_file
    if args.eps_list is not None:
        eps_all = [float(item) for item in args.eps_list.split(',')]
    else: 
        eps_all = [1.0,2.0,3.0,4.0,5.0,6.0]
        
    pheno_name = args.pheno
    seed = args.seed
    bins = args.bins
#     eps_cov = args.cov_priv
#     h2 = args.h2

#     phenoCovFile= args.cov_file
    
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()),flush=True)
    
    print(f"Running RR mechanism on phenotype: {pheno_name}",flush=True)
    
    
    pheno_df = load_phenotype(phenoFile,sample_subset=None)
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    n=Y_full.shape[0]
    for eps_itr in eps_all:

        priv_Y = RR(Y_full,eps_itr,bins,seed,None,None)

        outFile = out_path/ f"RR_sample_{n}_eps_{eps_itr}.txt"
        save_pheno_gwas(priv_Y,phenoFile,pheno_name,outFile)

    print(f"Randomized-Response mechanism for {pheno_name} done",flush=True)
    
    # if phenoCovFile is not None:
    #     covOutFile = src / f"phenoRRCovR2PC.txt"
    #     if not os.path.isfile(covOutFile):
    #             privRRSaveCovData(phenoCovFile,eps_cov,covOutFile,bins,seed)
    
    # print("Done")

	 

if __name__ == '__main__':
    main()
