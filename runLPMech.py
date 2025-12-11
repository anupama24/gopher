########################################################################
#    Implementation of the GOPHER-LP (RR-on-bins) mechanism for        #
#    generating private phenotypes with varying privacy budgets (ε).   #
########################################################################


"""Usage Example:
--------------
python runLPMech.py \
    --pheno_file data/simulated_phenotypes_unrelated_samples.txt \
    --pheno Sim_Y_100 \
    --eps_list 0.5,1.0,2.0 \
    --dest results/LP \
    --bins 100 \
    --seed 1234 \
    --sam 10000 \
    --eps 0.1
"""

import argparse
import os
from pathlib import Path

from pgen_reader import *
from functions import *
from utils import *
from rr_lp_utils import *

# ---------------------------------------------------------------------
#  Command-line arguments
# ---------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GOPHER-LP mechanism for private phenotypes")

    parser.add_argument('--dest', default=None, help='Path to destination folder (default: current directory)')
    parser.add_argument('--pheno_file', required=True, help='Path to phenotype file')
    parser.add_argument('--pheno', default="Sim_Y_100", type=str, help='Phenotype name to privatize')
    parser.add_argument('-el', '--eps_list', default=None, help='Comma-separated list of ε values (e.g., 0.5,1,2)')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed for reproducibility')
    parser.add_argument('--bins', default=100, type=int, help='Number of discrete bins for continuous phenotype')
    parser.add_argument('--eps', default=0.1, type=float, help='Privacy budget for prior (default: 0.1)')
    parser.add_argument('--sam', default=None, type=int, help='Sample size')
    parser.add_argument('--h2', default=None, type=float, help='Optional heritability for output naming')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup destination folder
    dest = Path(args.dest) if args.dest else Path.cwd()
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest #/ "LP"
    # out_path.mkdir(parents=True, exist_ok=True)

    # Handle epsilon list
    if args.eps_list:
        eps_all = [float(item) for item in args.eps_list.split(',')]
    else:
        eps_all = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Extract parameters
    pheno_file = Path(args.pheno_file)
    pheno_name = args.pheno
    seed = args.seed
    bins = args.bins
    eps1 = args.eps
    h2 = args.h2


    # Log configuration
    print("\n===== Configuration =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    # Load phenotype
    pheno_df = load_phenotype(pheno_file, sample_subset=None)
    Y_full = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    print(f"Loaded phenotype '{pheno_name}' with shape: {Y_full.shape}", flush=True)

    if args.sam:
        sam = args.sam
    else:
        sam= Y_full.shape[0]
    # Apply LP mechanism for each epsilon
    print(f"\nRunning GOPHER-LP mechanism on phenotype: {pheno_name}", flush=True)

    for eps_itr in eps_all:
        # Apply privacy mechanism
        Y_priv = RR_on_bins(Y_full, eps_itr, eps1, bins, seed)

        # Define output file name
        out_file = out_path / f"LP_sample_{sam}_eps_{eps_itr}_{pheno_name}.txt"
        if h2 is not None:
            out_file = out_path / f"LP_sample_{sam}_eps_{eps_itr}_{pheno_name}_h2_{h2}.txt"
        else:
            out_file = out_path / f"LP_sample_{sam}_eps_{eps_itr}_{pheno_name}.txt"
        
        # Save privatized phenotype
        save_pheno_gwas(Y_priv, pheno_file, pheno_name, out_file)
        
    print(f"\n GOPHER-LP mechanism completed successfully for '{pheno_name}'.")
    
     

if __name__ == '__main__':
    main()




    


