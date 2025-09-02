import numpy as np
import pandas as pd
import pgenlib as pg
import os
import re

"""Read pvar file as pd.DataFrame"""
def load_pvar(pvar_file):
    pvar_df = pd.read_csv(pvar_file, sep='\t', comment='#',
                       names=['chrom', 'pos', 'id', 'ref', 'alt'],
                       dtype={'chrom':str, 'pos':np.int32, 'id':str, 'ref':str, 'alt':str})
    return pvar_df


"""Read psam file as pd.DataFrame"""
def load_psam(psam_file):
    psam_df = pd.read_csv(psam_file, sep='\t', index_col=0)
    # psam_df.columns = ['IID','SEX','31-0.0-Sex','Center','BirthNorth','BirthEast','Ethnicity','BMI','Age','Caucasian','1-PC','2-PC','3-PC','4-PC','5-PC','6-PC','7-PC','8-PC','9-PC','10-PC']
    psam_df.index = psam_df.index.astype(str)
    return psam_df

"""Impute missing genotypes to mean"""
def impute_mean(genotypes):
    ids = genotypes == -9
    if genotypes.ndim == 1 and any(ids):
        genotypes[ids] = genotypes[~ids].mean()
    else:  
        rows = np.nonzero(ids)[0]
        # cols = np.nonzero(ids)[1]
        if len(rows) > 0:
            a = genotypes.sum(1)
            b = ids.sum(1)
            mu = (a + 9*b) / (genotypes.shape[1] - b)
            # mu = genotypes[~cols].mean(axis=0)
            genotypes[ids] = mu[rows]

"""Read genotypes with variant range from pgen file as pd.DataFrame. Impute missing values to mean (default)."""
def read_range_pgen(pgen_file, pvar_file, psam_file, start_idx,end_idx, dtype=np.int8):

    pvar_df = load_pvar(pvar_file)
    variant_ids = pvar_df['id'].tolist()
    
    if end_idx < 0:
        end_idx= len(variant_ids) -1
    print (end_idx)
    psam_df = load_psam(psam_file)
    sample_ids = psam_df.index.tolist()

    reader = pg.PgenReader(pgen_file.encode())
    num_samples = reader.get_raw_sample_ct()
    num_variants = end_idx- start_idx + 1
    genotypes = np.zeros([num_variants, num_samples], dtype=np.int8)
    with reader as r:
        r.read_range(start_idx,end_idx+1, genotypes)
    
    genotypes = genotypes.astype(dtype)
    impute_mean(genotypes)
    return pd.DataFrame(genotypes.T, index=sample_ids, columns=[variant_ids[start_idx:end_idx+1]]),psam_df


"""Read genotypes for given variant ids as pd.DataFrame. Impute missing values to mean (default)."""
def read_list_pgen(pgen_file, pvar_file, variant_ids, sample_id_list, dtype=np.int8):

    pvar_df = load_pvar(pvar_file)
    variant_list = pvar_df['id'].tolist()
    variant_idx_dict = {i:k for k,i in enumerate(variant_list)}
    variant_idxs = [variant_idx_dict[i] for i in variant_ids]

    reader = pg.PgenReader(pgen_file.encode())
    num_samples = reader.get_raw_sample_ct()
    num_variants = len(variant_idxs)
    
    print(num_variants,num_samples,flush=True)
    genotypes = np.zeros([num_variants, num_samples], dtype=dtype)
    with reader as r:
        r.read_list(np.array(variant_idxs, dtype=np.uint32), genotypes)

    genotypes = genotypes.astype(dtype)
    impute_mean(genotypes)
    X_df = pd.DataFrame(genotypes.T, index=sample_id_list, columns=variant_ids)

    return X_df 



"""Read genotypes for given subsample of variant ids and sample ids as pd.DataFrame. Impute missing values to mean (default)."""  
# X_df = read_subsample_pgen(geno_file.as_posix(), variant_idxs,sample_subset,dtype=np.int8)
def read_subsample_pgen(geno_file_prefix, variant_ids,sample_set=None,dtype=np.int8):

    pgen_file = f"{geno_file_prefix}.pgen"
    pvar_file = f"{geno_file_prefix}.pvar"
    psam_file = f"{geno_file_prefix}.psam"
    psam_df = load_psam(psam_file)

    print(psam_file)
    
    # psam_df.index = psam_df.index.astype(str)
    sample_id_list = psam_df.index.tolist()
    
    genotypes = read_list_pgen(pgen_file, pvar_file, variant_ids, sample_id_list, dtype)
    # print(genotypes.index,sample_set)
    
    genotypes.index = genotypes.index.astype(str)
    if sample_set is not None:
        # genotypes = genotypes[genotypes.index.isin(sample_set)]
        print(f"Genotypes len: {len(genotypes)}",flush = True)
        genotypes = genotypes[genotypes.index.isin(sample_set)]
        
    # X = X_df.to_numpy(dtype=dtype)
    print(f"Genotypes len: {len(genotypes)}",flush = True)
    return genotypes


"""Read genotypes as pd.DataFrame. Impute missing values to mean (default)."""
def load_genotypes(pgen_file, pvar_file, psam_file, dtype=np.int8):

    genotypes,psam_df = read_range_pgen(pgen_file, pvar_file, psam_file, 0,-1, dtype=np.int8)
    return genotypes

    

"""Read phenotype from file as pd.DataFrame. """
def load_phenotype(pheno_file,sample_subset=None):

    pheno_df = pd.read_csv(pheno_file, delimiter=r"\s+", index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    if sample_subset is not None:
        pheno_df = pheno_df[pheno_df.index.isin(sample_subset)]

    return pheno_df


