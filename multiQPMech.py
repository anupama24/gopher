########################################################################
#          Implementation of the GOPHER-MultiQP mechanism for          #
#    generating private phenotypes with varying privacy budgets (ε).   #
#                                                                      #
# It supports both real and simulated phenotypes, optionally excluding #
# previously privatized samples(`--lp_file`) and reusing PRS scores    #
# for phenotypic mean and variance.                                    #
########################################################################


"""Example:
--------
python multiQPMech.py \
  --geno_file ukb22828_c1_b0_v3 \
  --pheno_file data/phenotypes.txt \
  --score_file data/scores.sscore \
  --pheno Sim_Y_100 \
  --eps_list 0.5,1.0,2.0 \
  --sam 10000 \
  --h2 0.8 \
  --tag real \
  --seed 37
"""
import argparse
import os
import math
from pathlib import Path
from scipy import stats
from scipy.sparse.linalg import eigsh
import gc
from multi_qp_utils import *
from pgen_reader import *
from functions import *
from utils import *

# ---------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run GOPHER-MultiQP mechanism on phenotype data")
    
    parser.add_argument('--geno_file', required=True, help='Base tag for genotype files (without .pgen/.pvar/.psam)')
    
    parser.add_argument('--pheno_file', required=True, help='Path to phenotype file')
    parser.add_argument('--pheno', default="Sim_Y_100", help='Name of phenotype column')
    parser.add_argument('-el', '--eps_list', type=str, help='Comma-separated list of ε values (e.g. 1.0,2.0,3.0)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--bins', type=int, default=100, help='Number of bins for continuous phenotype')
    parser.add_argument('--eps', type=float, default=0.1, help='Privacy budget for prior')
    parser.add_argument('--score_file', required=True, help='Path to PRS score file')
    parser.add_argument('--lp_file', default=None, help='Optional path to existing privatized phenotype file used in PRS (for exclusion)')
    parser.add_argument('--sam', type=int, default=10000, help='Sample size')
    parser.add_argument('--tag', type=str, default=None, help='Phenotype type: real or sim')
    parser.add_argument('--h2', type=float, default=0.8, help='Heritability (used in naming)')
    parser.add_argument('--quantiles', '-q',type=int,default=7, help='Number of quantile bins to split PRS (default: 7)')
    parser.add_argument('--optTot', default=1e-6, type=float, help='Error tolerance for DCA')
    parser.add_argument('--itr', default=10, type=int, help='Number of iterations for DCA')
    
    return parser.parse_args()

# Helper Function
def est_var_y(y, n, eps):
    """Estimate variance of phenotype y under ε-DP Laplace noise."""
    var = np.var(y)
    sensitivity = (y.max() - y.min()) ** 2 / n
    return var + laplace_noise(sensitivity, eps, 1)

def main():
    args = parse_args()

    # Parse ε list
    eps_all = [float(e) for e in args.eps_list.split(',')] if args.eps_list else [1.0, 2.0, 3.0, 4.0, 5.0]

    print("\n===== Configuration =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    # --- Paths and Inputs ---
    pheno_file = Path(args.pheno_file)
    score_file = Path(args.score_file)
    lp_file = Path(args.lp_file) if args.lp_file else None
    pheno_name = args.pheno
    seed = args.seed
    bins = args.bins
    eps1 = args.eps
    sam =  args.sam

    h2 = args.h2
    tag = args.tag

    genoFile = args.geno_file
    pgen_file = f"{genoFile}.pgen"
    pvar_file = f"{genoFile}.pvar"
    psam_file = f"{genoFile}.psam"

    print(f"Running Multi-QP mechanism for phenotype: {pheno_name}", flush=True)


    # Load phenotype data
    pheno_df = load_phenotype(pheno_file,sample_subset=None)
    pheno_df.index = pheno_df.index.astype(str)
    
    Y_full =pheno_df[pheno_name].to_numpy(dtype=np.float32)
    Y_max=np.nanmax(Y_full)
    Y_min=np.nanmin(Y_full)

    # If LP file provided, exclude existing privatized samples
    if lp_file:
        lp_df = load_phenotype(lp_file,sample_subset=None)
        lp_df.index = lp_df.index.astype(str)
        exclude_ids = set(lp_df['IID'])
        
        is_in_set = pheno_df['IID'].isin(exclude_ids)
        row_numbers =  np.where(~is_in_set)[0]
        
        pheno_df = pheno_df[~pheno_df['IID'].isin(exclude_ids)]
        print(f"Excluded {len(exclude_ids)} samples from existing LP file", flush=True)

    # Load genotype data
    X_df = load_genotypes(pgen_file, pvar_file, psam_file, np.int8)
    X_df = X_df.loc[pheno_df.index]
    X_snp = X_df.to_numpy(dtype=np.float32)
    del X_df
    gc.collect()

    # Standardize in chunks
    chunk_size = 1000
    n_rows, n_cols = X_snp.shape
    for col_start in range(0, n_cols, chunk_size):
        col_end = min(col_start + chunk_size, n_cols)
        chunk = X_snp[:, col_start:col_end]
        mean = np.mean(chunk, axis=0)
        std = np.std(chunk, axis=0)
        std[std == 0] = 1.0
        X_snp[:, col_start:col_end] = (chunk - mean) / std

    gc.collect()
    print(f"Standardized genotype shape: {X_snp.shape}", flush=True)

    # --- Extract phenotype vector ---
    Y = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    n_full = Y_full.shape[0]
    n = Y.shape[0]
    print(f"Phenotype size: {n}", flush=True)
    
    
    
    # Load PRS scores and align indices
    Xmean_df = pd.read_csv(score_file, sep='\t', index_col='#FID')
    Xmean_df['group'] = pd.qcut(Xmean_df['SCORE1_SUM'], q=args.quantiles, labels=False) + 1

    # Align with phenotype individuals
    Xmean_df = Xmean_df[Xmean_df['IID'].isin(pheno_df['IID'])]
    Xmean_df = Xmean_df.loc[pheno_df['IID']]
    
    # Group-based DP mean and variance of phenotypes
    if tag == "real":
        # Compute group means/variances
        Xmean_df['Y'] = Y
        
        sensitivity_mean = (Y_max - Y_min) / n
        overall_mean_dp = np.mean(Y_full)+ laplace_noise(sensitivity_mean, eps1 * 0.1, size=1)
        overall_mean_dp=overall_mean_dp.astype('float32')[0]
        var1 = est_var_y(Y_full, n, eps1*0.9).astype('float32')[0]
        
        sd = np.sqrt(var1)
        lower,upper = stats.norm.interval(0.90, loc=overall_mean_dp, scale=sd)
        lower,upper = max(np.min(Y_full), lower), min(np.max(Y_full), upper)
        
        print(lower,upper)
        Xmean_df['X']= (Xmean_df['SCORE1_SUM']*np.sqrt(var1)) + overall_mean_dp
        # Xmean_df['group'] = pd.qcut(Xmean_df['X'].rank(method='first'), q=args.quantiles, labels=False) + 1

        # Compute 1st and 99th percentiles of Y for outlier bins
        Xmean_df['group'] = np.nan

        # Assign Q0 for below 1st percentile of Y
        Xmean_df.loc[Xmean_df['Y'] <= lower, 'group'] = 'Q0'
        # Assign Q8 for above 99th percentile of Y
        Xmean_df.loc[Xmean_df['Y'] >= upper, 'group'] = 'Q8'

        # Assign Q1–Q7 for values within 1–99th percentile using X
        mask = (Xmean_df['Y'] > lower) & (Xmean_df['Y'] < upper)
        Xmean_df.loc[mask, 'group'] = pd.qcut(
            Xmean_df.loc[mask, 'X'].rank(method='first'), q=args.quantiles,
            labels=[f'Q{i}' for i in range(1, args.quantiles+1)]
        )

        # Make categorical and ordered
        bin_order = ['Q0'] + [f'Q{i}' for i in range(1, args.quantiles+1)] + ['Q8']
        Xmean_df['group'] = Xmean_df['group'].astype('category')
        Xmean_df['group'] = Xmean_df['group'].cat.reorder_categories(bin_order, ordered=True)

    
        # Count how many elements are in each group
        counts = Xmean_df['group'].value_counts().sort_index()
        # Convert to list of categories
        groups = list(Xmean_df['group'].cat.categories)
        # Find which groups have only one element
        singletons = [g for g in groups if counts[g] == 1]

        # Create a mapping to merge singleton group → next group
        mapping = {}
        for i, g in enumerate(groups[:-1]):  # skip last since it has no "next"
            if g in singletons:
                mapping[g] = groups[i + 1]
        # If the last category has 1 element, merge it with the previous one
        if counts[groups[-1]] == 1:
            mapping[groups[-1]] = groups[-2]

        # Apply the mapping
        Xmean_df['group'] = Xmean_df['group'].replace(mapping)

        # Recreate as categorical (optional)
        Xmean_df['group'] = pd.Categorical(Xmean_df['group'], ordered=True)

        y_grouped = Xmean_df.groupby('group')['Y']
        y_group_mean = y_grouped.mean()
        y_group_var = y_grouped.var()
        group_counts = y_grouped.count()

        # Sensitivity calculations
        sensitivity_mean = (Y_max - Y_min) / group_counts
        sensitivity_var = ((Y_max - Y_min) ** 2) / group_counts

        # Apply Laplace noise
        dp_group_mean = y_group_mean + laplace_noise(sensitivity_mean, eps1 * 0.1, size=len(y_group_mean))
        dp_group_var = y_group_var + np.abs(laplace_noise(sensitivity_var, eps1 * 0.9, size=len(y_group_var)))

        # Global variance aggregation
        # overall_mean_dp = (group_counts * dp_group_mean).sum() / n
        # overall_var_dp = (group_counts * dp_group_var).sum() / n
        # between_group_var = ((group_counts * (dp_group_mean - overall_mean_dp) ** 2).sum()) / n
        # overall_var_dp += between_group_var

        # Assign privatized stats
        Xmean_df['dp_group_mean'] = Xmean_df['group'].map(dp_group_mean)
        Xmean_df['dp_group_var'] = Xmean_df['group'].map(dp_group_var)
        Xmean_df['dp_group_var'] = Xmean_df['dp_group_var'].astype(float)
        Xmean_df['dp_group_mean'] = Xmean_df['dp_group_mean'].astype(float)
        
        # Xmean_df['mean_score'] = (
        #     Xmean_df['SCORE1_SUM'] * np.sqrt(Xmean_df['dp_group_var']) + Xmean_df['dp_group_mean']
        # )
        # mean_dp = Xmean_df['dp_group_mean'].to_numpy() 
        # var1 = overall_var_dp
        # var_dp = var1* np.ones(len(mean_dp),dtype=np.float32)
        
        mean_dp = Xmean_df['mean_score'].to_numpy()
        var_dp = Xmean_df['dp_group_var'].to_numpy()
        # var1 = overall_var_dp

    else:
        # Simulated data case
        mean_dp = Xmean_df['SCORE1_SUM'].values
        overall_mean_dp = np.mean(Y_full)+ laplace_noise(sensitivity_mean, args.eps * 0.1, size=1)
        overall_mean_dp=overall_mean_dp.astype('float32')[0]
        
        overall_var_dp = est_var_y(Y_full, n, args.eps*0.9).astype('float32')[0]
        var_dp = np.full(len(mean_dp), overall_var_dp, dtype=np.float32)
        
    print(f"DP Mean/Var prepared. DP-mean: {overall_mean_dp}, DP-var: {overall_var_dp}. Example means: {mean_dp[:5]}, Var example: {var_dp[:5]}", flush=True)

    # Preprocessing for QP
    Q1, Q2, B, Y_uniq, Y_hat, chunks = sample_pre_process(X_snp, Y, bins, mean_dp, var_dp,overall_mean_dp,overall_var_dp, args.quantiles, seed, pheno_name)

    optTot = args.optTot
    max_iter = args.itr
        
    # Run Multi-LP across all epsilon values
    for eps_itr in eps_all:
        
        eps_temp = eps_itr - eps1
        print(f"\n Running Multi-QP for ε = {eps_itr}", flush=True)
        
        sol = opt_dca(Q1, B, Q2, Y_uniq.shape[0], Y_hat.shape[0], eps_temp, max_iter, optTot, 7)
        priv_Y = save_QP_Yhat(sol, Y_full, Y_uniq, Y_hat, seed, chunks)

        if h2 is not None:
            out_file =  f"MultiQP_sample_{args.sam}_eps_{eps_itr}_{pheno_name}_h2_{h2}.txt"
        else:
            out_file = f"MultiQP_sample_{args.sam}_eps_{eps_itr}_{pheno_name}.txt"
        # out_file = f"MultiLP_Priv_{args.sam}_eps_{eps_itr}_{pheno_name}.txt"


        # If file already exists, update; else create new
        # --- Save privatized phenotypes ---
        if os.path.isfile(out_file):
            pc_df = pd.read_csv(out_file, sep='\t', index_col=0)
            pc_df.index = pc_df.index.astype(str)

            # Identify rows not in id_set
            is_in_set = pc_df['IID'].isin(exclude_ids)
            row_numbers = np.where(~is_in_set)[0]

            # Extract the FIDs to update
            fids_to_update = pheno_df.iloc[row_numbers].index.astype(str)
            # print(len(fids_to_update))

            # Assign new values safely
            pc_df.loc[fids_to_update, pheno_name] = multi_Y_priv
            # print(pc_df.head(),flush=True)
            
        else:
            pc_df = pd.DataFrame(index=pheno_df.index.copy())
            pc_df.insert(0,'FID','')
            pc_df.insert(1,'IID','')
            pc_df['FID'] = pheno_df.index.copy()
            pc_df['IID'] = pheno_df.index.copy()   
            pc_df = pc_df.set_index("FID")
            pc_df[pheno_name] = multi_Y_priv
            # pc_df[pheno_name].update(lp_df[pheno_name])
            pc_df = pd.concat([pc_df, lp_df], axis=0)
        
        pc_df.to_csv(out_file, sep="\t", na_rep='NA',index=True)
        
    print(f"\n GOPHER-MultiQP mechanism completed for {pheno_name}.\n")


if __name__ == '__main__':
    main()


