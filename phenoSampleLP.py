import argparse
import os, fnmatch
import sys
import math
from pathlib import Path

# from utils import *
from pgen_reader import *
from functions import *
from utils import *
from rr_lp_utils import *
# from unbiased_lp_utils import *
#from tqdm.notebook import tqdm

""" Configure command line arguments """
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--src', default=None, help='path to merged genotypes folder')
    parser.add_argument('--dest', default=None, help='path to destination folder')
    parser.add_argument('--pheno_file', default=None, help='path to phenotype file')
    parser.add_argument('--pheno', default="Sim_Y_100", type=str, help='Name of phenotype')
    parser.add_argument('-el', '--eps_list', default=None,help='delimited eps list input', type=str)
    parser.add_argument('--seed', default=1234, type=int, help='Seed for random generator')
    parser.add_argument('--bins', default=100, type=int, help='Dicrete bins for continuous input')
    parser.add_argument('--eps', default=0.1, type=int, help='Privacy budget for prior')
    parser.add_argument('--sam', default=10000, type=int, help='Sample size')
    # parser.add_argument('--h2', default=0.8, type=float, help='h2')
    
    return parser.parse_args()

    
def main():

    args = parse_args()

    if args.dest is None:
        dest = Path.cwd()
    else:
        dest=Path(args.dest)
        
    # src = Path(args.src)
    
    ##Path to output folder
    out_path= dest#/"LP"
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
    seed = args.seed
    bins = args.bins
    eps1 = args.eps
    sam =  args.sam
    # h2 = args.h2
    
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()),flush=True)
    
    print(f"Running LP mechanism on phenotype: {pheno_name}",flush=True)

    
    pheno_df = load_phenotype(phenoFile,sample_subset=None)
    # pheno_df.index = pheno_df.index.astype(str)
    # pheno_df = pheno_df.reindex(X_df.index)
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    print(Y_full.shape)

    for eps_itr in eps_all:
    # eps_itr = 2.0
    
        Y_priv = RR_on_bins(Y_full,eps_itr,eps1,bins,seed)
        outFile = out_path/ f"LP_sample_{sam}_eps_{eps_itr}_{pheno_name}.txt"
        
        save_pheno_gwas(Y_priv,phenoFile,pheno_name,outFile)
            

    print(f"GOPHER-LP mechanism for {pheno_name} done",flush=True)
    
    
if __name__ == '__main__':
    main()


