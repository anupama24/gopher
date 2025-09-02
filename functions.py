from utils import *
from pgen_reader import *
import re

def rr_baseline(yarr,tot,eps,seed,fname=None):
    n = yarr.shape[0]
    rng = np.random.RandomState(seed)
    new_yarr = np.zeros(n)
    for i in range(n):
        new_yarr[i] =  rand_response(yarr[i],tot-yarr[i],eps,rng)
    if fname is not None:
        np.save(fname,new_yarr)
        
    return new_yarr



"""Save phenotype in format required by PLINK for GWAS"""
def save_pheno_gwas(Y_priv, phenoFile, pheno_name,outFile):

    pheno_df = load_phenotype(phenoFile,sample_subset=None)
    if os.path.isfile(outFile):
        pc_df= pd.read_csv(outFile, sep='\t', index_col=0)
        pc_df.index = pc_df.index.astype(str)
        # pc_df = pc_df[pc_df.columns]
        # pc_df['IID'] = pc_df.index.copy()   
        print(pc_df.head(),flush=True)
    else:
        pc_df = pd.DataFrame(index=pheno_df.index.copy())
        pc_df.insert(0,'FID','')
        pc_df.insert(1,'IID','')
        pc_df['FID'] = pheno_df.index.copy()
        pc_df['IID'] = pheno_df.index.copy()   
        pc_df = pc_df.set_index("FID")

    # pheno_df.reset_index(drop=True, inplace=True)
    pc_df[pheno_name] =  Y_priv
    print(pc_df.columns,flush=True)
    print(len(pc_df),flush=True)
    
    pc_df.to_csv(outFile, sep="\t", na_rep='NA',index=True)    
    
    
"""Read genotype from 0 to given postions and phenotype from file as np.array. Returns genotype, phenotype and sites id array"""
def getXY_range(geno_file_prefix, pheno_file, npositions, pheno_name, dtype=np.int8):

    pgen_file = f"{geno_file_prefix}.pgen"
    pvar_file = f"{geno_file_prefix}.pvar"
    psam_file = f"{geno_file_prefix}.psam"
    genotypes_df,psam_df = read_range_pgen(pgen_file, pvar_file, psam_file, 0,npositions-1, dtype)
    
    X = genotypes_df.to_numpy(dtype=dtype)

    pheno_df = read_pheno(pheno_file,sample_subset=None)
    pheno_df = pheno_df.reindex(genotypes_df.index)
    Y = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    
    # Y_ids = np.where((np.isclose(Y,-9))| (np.isclose(Y,-1)))[0] #~np.isnan(Y)
    # mask = np.ones(Y.shape, bool)
    # mask[Y_ids] = False
    # X = X[mask]
    # Y = Y[mask]
    sample_subset = genotypes_df.index.tolist()
    sample_subset = sample_subset[~np.isnan(Y)]
    X=X[~np.isnan(Y)]
    Y=Y[~np.isnan(Y)]
    print(f"X shape: {X.shape}, Y shape {Y.shape}")
    
    return X,Y,sample_subset

	

"""Get genotype and phenotype data for subsample of varaint ids and sample ids as numpy array"""
def getXY(sample_file,var_file_name,geno_file,phenoOutFile,pheno_name):
    
    # sample_subset = read_list(sample_file)
    # psam_df = load_psam(sample_file)
    psam_df = pd.read_csv(sample_file, sep=' ', index_col=0)
    psam_df.index = psam_df.index.astype(str)
    sample_subset = psam_df.index.tolist()
    print(len(sample_subset))
    
    variant_idxs = read_list(var_file_name)
    # X_df = read_subsample_pgen(geno_file.as_posix(), variant_idxs,sample_subset,dtype=np.int8)
    X_df = read_subsample_pgen(geno_file, variant_idxs,sample_subset,dtype=np.int8)
    X = X_df.to_numpy(dtype=np.int8)
    
    pheno_df = load_phenotype(phenoOutFile,sample_subset)
    pheno_df = pheno_df.reindex(X_df.index)
    Y = pheno_df[pheno_name].to_numpy(dtype=np.float32)
    sample_subset = X_df.index.to_numpy()

    sample_subset = sample_subset[~np.isnan(Y)]
    
    X=X[~np.isnan(Y)]
    Y=Y[~np.isnan(Y)]
    
    print(f"X shape: {X.shape}, Y shape {Y.shape}")
    return X,Y,sample_subset


"""Get genotype, phenotype and covariate data for subsample of varaint ids and sample ids as numpy array"""
def getXWY(sample_file,var_file_name,geno_file,phenoOutFile,pheno_name,covOutFile):

    X,Y,sample_ids = getXY(sample_file,var_file_name,geno_file,phenoOutFile,pheno_name)
    
    W_df = pd.read_csv(covOutFile, sep='\t', index_col=0)
    W_df.index = W_df.index.astype(str)
    if sample_ids is not None:
        W_df = W_df[W_df.index.isin(sample_ids)]
    
    W_df = W_df.reindex(sample_ids)
    # W_df.reset_index(drop=True, inplace=True)
    W_df = W_df.drop(['IID'], axis=1)
    W = W_df.to_numpy(dtype=np.float32)
    print(f"X shape: {X.shape}, W shape: {W.shape}, Y shape {Y.shape}")
    return X,W,Y

    


"""Store covariate file with private columns of age and sex"""
def savePrivCovData(covFile,age,sex,covOutFile,flag=None):

    W_df = pd.read_csv(covFile, sep='\t', index_col=0)

    pheno_df = W_df.copy(deep=True)
    print(pheno_df.head(),flush=True)
    # ['IID','SEX','31-0.0-Sex','Center','BirthNorth','BirthEast','Ethnicity','BMI','Age','Caucasian','1-PC','2-PC','3-PC','4-PC','5-PC','6-PC','7-PC','8-PC','9-PC','10-PC']
    pheno_df.set_index('IID', inplace=True)
    pheno_df['AGE'] = age
    pheno_df['SEX'] = sex
    pheno_df['AGE_SQ'] = np.square(pheno_df['AGE'])
    pheno_df['AGE_SEX'] = pheno_df['AGE'] * pheno_df['SEX']
    pheno_df['AGE_SQ_SEX'] = np.square(pheno_df['AGE']) * pheno_df['SEX']
    
    pheno_df['FID'] = pheno_df.index.copy()
    pheno_df['IID'] = pheno_df.index.copy()

    print(pheno_df.head(),flush=True)
    if flag is None:
        pheno_df = pheno_df[ ['FID', 'IID', 'AGE', 'SEX' , 'AGE_SQ', 'AGE_SEX', 'AGE_SQ_SEX', 'PC1' ,'PC2' ,'PC3','PC4', 'PC5', 'PC6','PC7','PC8','PC9','PC10']] 
    else:
        pheno_df = pheno_df[['FID', 'IID', 'AGE', 'SEX' , 'AGE_SQ', 'AGE_SEX', 'AGE_SQ_SEX', 'PC1' ,'PC2' ,'PC3','PC4', 'PC5']] 

    pheno_df.to_csv(covOutFile, sep="\t", na_rep='NA',index=False)


