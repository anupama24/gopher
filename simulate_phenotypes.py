"""
Simulate phenotypes from genotype data using pgen_reader.

Usage:
    python simulate_phenotypes.py --geno-prefix ukb_qc_thinned_unrelated_samples --h2 0.8
"""
import argparse
import sys
import math
import os
from pathlib import Path

from pgen_reader import *
from utils import *

# -----------------------------------------------------------------------------
# Command-line arguments (optional)
# -----------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description="Simulate phenotypes from PGEN data")
parser.add_argument("--data-path", type=str, default=str(Path.cwd()),
                    help="Path to genotype files (default: current directory)")
parser.add_argument("--geno-prefix", type=str, default="ukb_qc_thinned_unrelated_samples",
                    help="Prefix of genotype files (.pgen/.pvar/.psam)")
parser.add_argument("--pheno-file", type=str, default=None,
                    help="Output phenotype file (default: <data-path>/simulated_phenotypes.txt)")
parser.add_argument("--h2", type=float, default=0.8,
                    help="Heritability for phenotype simulation")
parser.add_argument("--seed", type=int, default=1234,
                    help="Random seed for reproducibility")
parser.add_argument("--chunk-size", type=int, default=1000,
                    help="Chunk size for standardization")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup paths
# -----------------------------------------------------------------------------
data_path = Path(args.data_path)
geno_file = args.geno_prefix
pheno_file = Path(args.pheno_file) if args.pheno_file else data_path / "simulated_phenotypes.txt"
h2 = args.h2
seed = args.seed
chunk_size = args.chunk_size

# -----------------------------------------------------------------------------
# Setup file paths
# -----------------------------------------------------------------------------
pgen_file = data_path / f"{geno_file}.pgen"
pvar_file = data_path / f"{geno_file}.pvar"
psam_file = data_path / f"{geno_file}.psam"

for f in [pgen_file, pvar_file, psam_file]:
    if not f.exists():
        raise FileNotFoundError(f"File not found: {f}")

# -----------------------------------------------------------------------------
# Load genotype data
# -----------------------------------------------------------------------------
print("Loading genotype data...")
X = load_genotypes(str(pgen_file), str(pvar_file), str(psam_file), np.int8).to_numpy(dtype=np.int8)
print(f"Genotype matrix shape: {X.shape}")

# -----------------------------------------------------------------------------
# Standardize genotypes in chunks
# -----------------------------------------------------------------------------
Xstd = np.zeros(X.shape, dtype=np.float32)

def standardize_with_chunks(chunk_size=1000):
    n_rows, n_cols = X.shape
    for col_start in range(0, n_cols, chunk_size):
        col_end = min(col_start + chunk_size, n_cols)
        mean = np.mean(X[:, col_start:col_end], axis=0)
        std = np.std(X[:, col_start:col_end], axis=0)
        std[std == 0] = 1.0
        # Xstd[:, col_start:col_end] = (X[:, col_start:col_end] - mean) / std
        for row_start in range(0, n_rows, chunk_size):
                row_end = min(row_start + chunk_size, n_rows)
                Xstd[row_start:row_end, col_start:col_end] = (X[row_start:row_end, col_start:col_end] - mean) / std

standardize_with_chunks(chunk_size=chunk_size)

# -----------------------------------------------------------------------------
# Simulate phenotypes
# -----------------------------------------------------------------------------
rng = np.random.RandomState(seed)

n, d = Xstd.shape
print(f"Standardized genotype matrix: {n} samples x {d} variants")

causal_variant_counts = [100, 1000, 10000, 100000]
var = 1.0

for varNum in causal_variant_counts:
    pi = varNum / d
    sigma = np.sqrt(h2 / varNum)
    beta = np.zeros(d, dtype=np.float32)
    causal_mask = rng.rand(d) < pi
    beta[causal_mask] = rng.normal(0.0, sigma, size=np.sum(causal_mask))
    
    print(f"{np.count_nonzero(beta)} causal variants assigned (target={varNum})")

    psam_df = load_psam(str(psam_file))
    sample_ids = psam_df.index.astype(str).tolist()

    sigma2_e = var - h2
    e = rng.normal(0.0, np.sqrt(sigma2_e), n)
    sim_y = Xstd @ beta[:, None] + e[:, None]

    pheno_name = f"Sim_Y_{varNum}"
    print(f"Simulated phenotype using {varNum} causal SNPs, length: {sim_y.shape},mean: {np.mean(sim_y)} var {np.var(sim_y)}",flush=True)
    
    if os.path.isfile(pheno_file):
        pheno_df = pd.read_csv(pheno_file, sep='\t', index_col=0)
        pheno_df.index = pheno_df.index.astype(str)
    else:
        pheno_df = pd.DataFrame()
        pheno_df.insert(0, "FID", sample_ids)
        pheno_df.insert(1, "IID", sample_ids)
        pheno_df = pheno_df.set_index("FID")

    pheno_df[pheno_name] = sim_y
    pheno_df.to_csv(pheno_file, sep='\t', na_rep='NA',index=True)

print(f"\n Simulated phenotype file {pheno_file} generated successfully.")
