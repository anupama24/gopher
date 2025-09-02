##########################################################################
## Implementation of the baseline methods to generate private phenotype ##
## Test for different values of privacy budget                          ##
########################################################################## 


import argparse
import os, fnmatch
import sys
import math

from utils import *
from pgen_reader import *
from functions import *
#sys.path.append(".")

""" Configure command line arguments """
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--src', default=None, help='path to merged genotypes and phenotype folder')
    parser.add_argument('--dest', default=None, help='path to destination folder')
    parser.add_argument('--pheno_file', default=None, help='path to phenotype file')
    parser.add_argument('--pheno', default=None, type=str, help='Name of phenotype')
    parser.add_argument('-el', '--eps_list', default=None,help='delimited eps list input', type=str)
    # parser.add_argument('--h2', default=0.8, type=float, help='h2')
    # parser.add_argument('--cov_file', default=None, help='path to covariate file')
    # parser.add_argument('--cov_priv', default=1.0, type=float, help='total privacy budget for covariates')
    return parser.parse_args()


def main():
    
    args = parse_args()
    if args.dest is None:
        dest = Path.cwd()
    else:
        dest=Path(args.dest)
        
    # src = Path(args.src)
    
    ##Path to output folder
    out_path = dest
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # if args.pheno_file is None:
    #     phenoFile =  Path.cwd() / "test_data/ukb_simulate_pheno_lp.txt"
    # else:
    phenoFile= args.pheno_file

    if args.eps_list is not None:
        eps_all = [float(item) for item in args.eps_list.split(',')]
    else: 
        eps_all = [1.0,2.0,3.0,4.0,5.0,6.0]
        
    pheno_name = args.pheno
#     eps_cov = args.cov_priv
#     h2 = args.h2
#     phenoCovFile= args.cov_file
    
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()),flush=True)
    print(f"Running baseline method-Laplace mechanism on phenotype: {pheno_name}",flush=True)

    
    pheno_df = load_phenotype(phenoFile,sample_subset=None)
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    Y = Y_full[~np.isnan(Y_full)]
    n=Y.shape[0]

    Y_new = np.zeros(n,dtype=np.float32)
    print(f"Sensitivity= {Y.max()}-{Y.min()}", flush=True)
    
    for eps_itr in eps_all:

        sensitivity = Y.max() - Y.min()
        Y_new = Y + laplace_noise(sensitivity, eps_itr,n)
        
        priv_Y = np.zeros(Y_full.shape[0],dtype=np.float32)
        j= 0
        for i in range(Y_full.shape[0]):
            if (np.isclose(Y_full[i],-9) or np.isnan(Y_full[i]) or j>=n):
                priv_Y[i] = np.nan
            else:
                if j<n and Y_new[j] <0:
                    Y_new[j] = Y.min()
                priv_Y[i] = Y_new[j]
                j+=1

        outFile = out_path/ f"Lap_sample_{n}_eps_{eps_itr}_{pheno_name}.txt"
        save_pheno_gwas(priv_Y,phenoFile,pheno_name,outFile)
            

    print(f"Laplace mechanism for {pheno_name} done",flush=True)
    

#     if phenoCovFile is not None:
#         covOutFile = src / f"phenoLapCovR2PC_eps_{eps_itr}_{h2}.txt"
#         if not os.path.isfile(covOutFile):
#             privSaveCovData(phenoCovFile,eps_cov,covOutFile)
    
    
    
# def privSaveCovData(covFile,eps,covOutFile):

#     W_df = pd.read_csv(covFile, sep='\t', index_col=0)
#     age_full  = W_df['AGE'].to_numpy(dtype=np.int32)
#     age = age_full[~np.isnan(age_full)]
#     priv_age = np.zeros(age_full.shape[0],dtype=np.int32)
    
#     sex_full = W_df['SEX'].to_numpy(dtype=np.int8)
#     sex = sex_full[~np.isnan(sex_full)]
#     priv_sex = np.zeros(sex_full.shape[0],dtype=np.int8)

#     eps1 = (1/2) * eps
#     eps2 = (1/2) * eps

#     age_new = (np.rint(age + laplace_noise((age.max()-age.min()), eps1,age.shape[0]))).astype(int)
#     sex_new = sex + laplace_noise((sex.max()-sex.min()), eps2,sex.shape[0])
    
#     j= 0
#     for i in range(age_full.shape[0]):
#         if (np.isclose(age_full[i],-9) or np.isnan(age_full[i]) or j>=age.shape[0]):
#             priv_age[i] = np.nan
#         else:
#             if j<age.shape[0] and age_new[j] <0:
#                 age_new[j] = age.min()
#             priv_age[i] = age_new[j]
#             j+=1
#     j=0
#     for i in range(sex_full.shape[0]):
#         if (np.isclose(sex_full[i],-9) or np.isnan(sex_full[i]) or j>=sex.shape[0]):
#             priv_sex[i] = np.nan
#         else:
#             if sex_new[j] <0.5:
#                 priv_sex[i] = 1
#             else:
#                 priv_sex[i] = 2
#             j+=1

#     savePrivCovData(covFile,priv_age,priv_sex,covOutFile)

	 

if __name__ == '__main__':
    main()
