########################################################################
#          Implementation of the GOPHER-MultiLP mechanism for          #
#    generating private phenotypes with varying privacy budgets (ε).   #
#                                                                      #
# It supports both real and simulated phenotypes, optionally excluding #
# previously privatized samples(`--lp_file`) and reusing PRS scores    #
# for phenotypic mean and variance.                                    #
########################################################################


"""Example:
--------
python multiLPMech.py \
  --geno_file ukb22828_c1_b0_v3 \
  --pheno_file ./data/phenotypes.txt \
  --score_file ./data/scores.sscore \
  --pheno Sim_Y_100 \
  --eps_list 0.5,1.0,2.0 \
  --sam 10000 \
  --h2 0.8 \
  --tag real \
  --seed 1234
"""
import argparse
import os
import math
from pathlib import Path

from pgen_reader import *
from functions import *
from utils import *
from rr_lp_utils import *
from calcDPMeanVar import est_DPMean_Var

# ---------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run GOPHER-MultiLP mechanism on phenotype data")
    
    # parser.add_argument('--geno_file', required=True, help='Base tag for genotype files (without .pgen/.pvar/.psam)')
    parser.add_argument('--dest', default=None, help='Path to destination folder (default: current directory)')
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
    parser.add_argument('--h2', type=float, default=None, help='Heritability (used in naming)')
    parser.add_argument('--quantiles', '-q',type=int,default=7, help='Number of quantile bins to split PRS (default: 7)')

    # Add optional mean and variance input
    parser.add_argument('--mean', type=float, default=None, help='Optional DP mean of phenotype')
    parser.add_argument('--var', type=float, default=None, help='Optional DP variance of phenotype')

    return parser.parse_args()

# Helper Function
def est_var_y(y, n, eps):
    """Estimate variance of phenotype y under ε-DP Laplace noise."""
    var = np.var(y)
    sensitivity = (y.max() - y.min()) ** 2 / n
    return var + laplace_noise(sensitivity, eps, 1)

def main():
    args = parse_args()

    dest = Path(args.dest) if args.dest else Path.cwd()
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "MultiLP"
    out_path.mkdir(parents=True, exist_ok=True)
    
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

    if args.mean is None or args.var is None:
        mean, var = est_DPMean_Var(args.pheno_file, args.pheno)
    else:
        mean, var = args.mean, args.var


    print(f"Running Multi-LP mechanism for phenotype: {pheno_name}", flush=True)


    # Load phenotype data
    full_pheno_df = load_phenotype(pheno_file,sample_subset=None)
    print(len(full_pheno_df))
    full_pheno_df.index = full_pheno_df.index.astype(str)
    

    Y_full =full_pheno_df[pheno_name].to_numpy(dtype=np.float32)
    Y_max=np.nanmax(Y_full)
    Y_min=np.nanmin(Y_full)

    # If LP file provided, exclude existing privatized samples
    if lp_file:
        lp_df = load_phenotype(lp_file,sample_subset=None)
        lp_df.index = lp_df.index.astype(str)
        exclude_ids = set(lp_df['IID'])
        
        is_in_set = full_pheno_df['IID'].isin(exclude_ids)
        row_numbers =  np.where(~is_in_set)[0]
        
        pheno_df = full_pheno_df[~full_pheno_df['IID'].isin(exclude_ids)]
        print(f"Excluded {len(exclude_ids)} samples from existing LP file", flush=True)

    # --- Extract phenotype vector ---
    Y = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    n = len(Y)
    print(f"Phenotype size: {n}", flush=True)
    
    id_set = set(pheno_df['IID'])
    
    # Load PRS scores and align indices
    Xmean_df = pd.read_csv(score_file, sep='\t', index_col='#FID')
    
    # Align with phenotype individuals
    # Xmean_df = Xmean_df[Xmean_df['IID'].isin(pheno_df['IID'])]
    Xmean_df = Xmean_df.loc[full_pheno_df['IID']]
    
    Xmean_df['Y'] = Y_full

    sensitivity_mean = (Y_max - Y_min) / n
    overall_mean_dp=mean
    overall_var_dp=var
    
    sd = np.sqrt(overall_var_dp)
    lower,upper = stats.norm.interval(0.90, loc=overall_mean_dp, scale=sd)
    lower,upper = max(Y_min, lower), min(Y_max, upper)
    
    print(lower,upper)
    Xmean_df['Y'] = Xmean_df['Y'].clip(lower, upper)
    Xmean_df['X']= (Xmean_df['SCORE1_SUM']*sd) + overall_mean_dp
    Xmean_df['group'] = pd.qcut(Xmean_df['SCORE1_SUM'].rank(method='first'), q=args.quantiles, labels=False) + 1

    y_grouped = Xmean_df.groupby('group')['Y']
    y_group_mean = y_grouped.mean()
    y_group_var = y_grouped.var()
    group_counts = y_grouped.count()
    y_min = lower#y_grouped.min()
    y_max = upper#y_grouped.max()
    # Sensitivity calculations
    sensitivity_mean = (y_max - y_min) / group_counts
    sensitivity_var = ((y_max - y_min) ** 2) / group_counts

    # Apply Laplace noise
    dp_group_mean = y_group_mean + laplace_noise(sensitivity_mean, eps1 * 0.1, size=len(y_group_mean))
    dp_group_var = y_group_var + np.abs(laplace_noise(sensitivity_var, eps1 * 0.9, size=len(y_group_var)))


    # Assign privatized stats
    Xmean_df['dp_group_mean'] = Xmean_df['group'].map(dp_group_mean)
    Xmean_df['dp_group_var'] = Xmean_df['group'].map(dp_group_var)

    Xmean_df['dp_group_var'] = Xmean_df['dp_group_var'].astype(float)
    Xmean_df['dp_group_mean'] = Xmean_df['dp_group_mean'].astype(float)
    # Xmean_df['mean_score'] = (
    #     Xmean_df['SCORE1_SUM'] * np.sqrt(Xmean_df['dp_group_var']) + Xmean_df['dp_group_mean']
    # )

    # Align with remaining phenotype individuals
    Xmean_df = Xmean_df[Xmean_df['IID'].isin(pheno_df['IID'])]
    Xmean_df = Xmean_df.loc[pheno_df['IID']]
    print(len(Xmean_df))
    
    # Group-based DP mean and variance of phenotypes
    if tag == "real":
        # Compute group means and variances 
        mean_dp = Xmean_df['dp_group_mean'].to_numpy()
        var_dp = Xmean_df['dp_group_var'].to_numpy()
        # var_dp = var1* np.ones(n,dtype=np.float32)
        # mean_dp = overall_mean_dp * np.ones(n,dtype=np.float32)
        # var1 = overall_var_dp

    else:
        # Simulated data case
        mean_dp = Xmean_df['SCORE1_SUM'].values
        var_dp = Xmean_df['dp_group_var'].to_numpy()
        # overall_mean_dp = np.mean(Y_full)+ laplace_noise(sensitivity_mean, eps1 * 0.1, size=1)
        # overall_mean_dp=overall_mean_dp.astype('float32')[0]
        # var1 = est_var_y(Y_full, n, eps1*0.9).astype('float32')[0]
        
        # var_dp = np.full(len(mean_dp), var, dtype=np.float32)
        # mean_dp = mean_dp* np.sqrt(var) + mean

    print(f"DP Mean/Var prepared. DP-mean: {overall_mean_dp}, DP-var: {overall_var_dp}. Example means: {mean_dp[:5]}, Var example: {var_dp[:5]}", flush=True)

    # Run Multi-LP across all epsilon values
    for eps_itr in eps_all:
        print(f"\n Running Multi-LP for ε = {eps_itr}", flush=True)
        
        # Apply pool RR mechanism
        num_chunks = min(1000, len(Y))
        multi_Y_priv = pool_rr_on_bins(Y, bins, eps_itr, 2*eps1, mean_dp, var_dp,overall_mean_dp,overall_var_dp,num_chunks,seed)
        
        print(np.unique(multi_Y_priv))
        if h2 is not None:
            out_file = out_path /f"MultiLP_sample_{args.sam}_eps_{eps_itr}_{pheno_name}_h2_{h2}1.txt"
        else:
            out_file = out_path /f"MultiLP_sample_{args.sam}_eps_{eps_itr}_{pheno_name}1.txt"
        
        

        # If file already exists, update; else create new
        # --- Save privatized phenotypes ---
        if os.path.isfile(out_file):

            temp_df = pd.DataFrame(index=pheno_df.index.copy())
            temp_df.insert(0,'FID','')
            temp_df.insert(1,'IID','')
            temp_df['FID'] = pheno_df.index.copy()
            temp_df['IID'] = pheno_df.index.copy()   
            temp_df = temp_df.set_index("FID")
            temp_df[pheno_name] = multi_Y_priv
            # pc_df[pheno_name].update(lp_df[pheno_name])
            temp_df = pd.concat([temp_df, lp_df], axis=0)

            pc_df = pd.read_csv(out_file, sep='\t', index_col=0)
            pc_df.index = pc_df.index.astype(str)
            pc_df[pheno_name].update(temp_df[pheno_name])
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

       
    print(f"\n GOPHER-MultiLP mechanism completed for {pheno_name}.\n")

        
        

if __name__ == '__main__':
    main()


