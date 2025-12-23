from utils import *
import argparse
import os
from pathlib import Path

from pgen_reader import *
from functions import *
from rr_lp_utils import *

# ---------------------------------------------------------------------
#  Command-line arguments
# ---------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Estimate DP mean and variance")

    parser.add_argument('--pheno_file', required=True, help='Path to phenotype file')
    parser.add_argument('--pheno', default="Sim_Y_100", type=str, help='Phenotype name to privatize')
    parser.add_argument('--eps', default=0.1, type=float, help='Privacy budget for prior (default: 0.1)')
    return parser.parse_args()


def est_DPMean_Var(Yarr, eps):
    
    Y = Yarr[~np.isnan(Yarr)]
    n = Y.shape[0]
    # print(f"Max(Y)={Y.max():.3f}, Min(Y)={Y.min():.3f}", flush=True)

    mean_dp = np.mean(Y) + laplace_noise((Y.max() - Y.min()) / n, eps * 0.1, 1)
    var = np.var(Y) + np.abs(laplace_noise((Y.max() - Y.min()) ** 2 / n, 0.9 * eps, 1))
    # print(f"DP mean={mean_dp.astype('float32')[0]:.3f}, DP var={var.astype('float32')[0]:.3f}", flush=True)
    return mean_dp, var

if __name__ == "__main__":

    args = parse_args()
    pheno_file = Path(args.pheno_file)
    pheno_name = args.pheno
    eps = args.eps
    
    pheno_df = load_phenotype(pheno_file, sample_subset=None)
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    # print(f"Loaded phenotype '{pheno_name}' with shape: {Y_full.shape}", flush=True)
    # print(np.mean(Y_full),np.var(Y_full))
    mean,var = est_DPMean_Var(Y_full, eps)

    print(f"{mean.astype('float32')[0]} {var.astype('float32')[0]}")
