##########################################################################
## Implementation of Laplace and Randomized Response (RR) mechanisms     ##
## for generating private phenotypes under differential privacy.         ##
## Supports testing across multiple epsilon values.                      ##
##########################################################################


import argparse
import os, fnmatch
import sys
import math

from utils import *
from pgen_reader import *
from functions import *
from rr_lp_utils import *

# ----------------------------- #
#   Command-line configuration  #
# ----------------------------- #
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate private phenotypes using Laplace or Randomized Response mechanisms.")
    parser.add_argument("--dest", default=None, help="Path to destination folder for outputs")
    parser.add_argument("--pheno_file", required=True, help="Phenotype file")
    parser.add_argument("--pheno", required=True, help="Name of phenotype column")
    parser.add_argument("--mech", choices=["laplace", "rr", "both"], default="both", help="Mechanism to apply")
    parser.add_argument("-el", "--eps_list", default="1.0,2.0,3.0,4.0,5.0", help="Comma-separated list of epsilon values")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for random number generator")
    parser.add_argument("--bins", type=int, default=100, help="Discrete bins for continuous inputs (for RR)")
    parser.add_argument("--h2", type=float, default=None, help="Heritability (optional)")
    
    return parser.parse_args()


def find_indices(arr, bin_centers):
    
    diff_matrix = np.abs(arr[:, None] - bin_centers[None, :])
    indices = np.argmin(diff_matrix, axis=1)
    return indices

# ----------------------------- #
#      Randomized Response      #
# ----------------------------- #
def apply_RR(Y_arr, eps, bins, seed):
    
    """Apply Randomized Response (RR) mechanism to phenotype array."""
    Y = Y_arr[~np.isnan(Y_arr)]
    print(f"RR: phenotype range: min={Y.min()}, max={Y.max()}", flush=True)

    low, high = Y_arr.min(), Y_arr.max()
    bin_edges = np.linspace(low, high, bins + 1)
    Y_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rng = np.random.RandomState(seed)

    priv_y = np.full(Y_arr.shape, np.nan, dtype=np.float32)

    for i, val in enumerate(Y_arr):
        if np.isnan(val):
            continue
        idx = np.abs(Y_bins - val).argmin()
        new_val = compute_RR_bins(Y_bins[idx], Y_bins, eps, rng)
        priv_y[i] = new_val

    return priv_y

# ----------------------------- #
#       Laplace Mechanism       #
# ----------------------------- #
def apply_laplace(Y_arr, eps, seed):
    
    """Apply Laplace mechanism to phenotype array."""
    Y = Y_arr[~np.isnan(Y_arr)]
    print(f"Laplace: phenotype range: min={Y.min()}, max={Y.max()}", flush=True)

    sensitivity = Y.max() - Y.min()
    rng = np.random.RandomState(seed)

    noise = laplace_noise(sensitivity, eps, len(Y))
    Y_private = Y + noise

    priv_Y = np.full(Y_arr.shape, np.nan, dtype=np.float32)
    j = 0
    for i, val in enumerate(Y_arr):
        if np.isnan(val) or np.isclose(val, -9) or j >= len(Y):
            continue
        priv_Y[i] = max(Y_private[j], Y.min())
        j += 1

    return priv_Y
 

# ----------------------------- #
#              Main             #
# ----------------------------- #
def main():
    args = parse_args()

    dest = Path(args.dest) if args.dest else Path.cwd()
    dest.mkdir(parents=True, exist_ok=True)

    # Create mechanism-specific subfolders only if needed
    lap_dir = dest / "Laplace"
    rr_dir = dest / "RR"

    if args.mech in ["laplace", "both"]:
        lap_dir.mkdir(parents=True, exist_ok=True)
    if args.mech in ["rr", "both"]:
        rr_dir.mkdir(parents=True, exist_ok=True)
        
    eps_all = [float(eps) for eps in args.eps_list.split(",")]
    pheno_file = Path(args.pheno_file)
    pheno_name = args.pheno
    h2 = args.h2
    
    print("\n===== Configuration =====")
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()),flush=True)
    print("==========================\n")

    # Load phenotype
    pheno_df = load_phenotype(pheno_file, sample_subset=None)
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    n = len(Y_full)

    # ---- Laplace Mechanism ----
    if args.mech in ["laplace", "both"]:
        print(f"Running Laplace mechanism for phenotype '{pheno_name}'...")
        for eps in eps_all:
            priv_Y = apply_laplace(Y_full, eps, args.seed)
            # Add h2 to filename only if provided
            if h2 is not None:
                out_file = lap_dir / f"Lap_sample_{n}_eps_{eps}_{pheno_name}_h2_{h2}.txt"
            else:
                out_file = lap_dir / f"Lap_sample_{n}_eps_{eps}_{pheno_name}.txt"

            save_pheno_gwas(priv_Y, pheno_file, pheno_name, out_file)
        print("Laplace mechanism completed.\n")

    # ---- Randomized Response ----
    if args.mech in ["rr", "both"]:
        print(f"Running Randomized Response mechanism for phenotype '{pheno_name}'...")
        for eps in eps_all:
            priv_Y = apply_RR(Y_full, eps, args.bins, args.seed)
            
            if h2 is not None:
                out_file = rr_dir / f"RR_sample_{n}_eps_{eps}_{pheno_name}_h2_{h2}.txt"
            else:
                out_file = rr_dir / f"RR_sample_{n}_eps_{eps}_{pheno_name}.txt"
                
            save_pheno_gwas(priv_Y, pheno_file, pheno_name, out_file)
        print("Randomized Response mechanism completed.\n")

    print("Done.")


if __name__ == '__main__':
    main()
