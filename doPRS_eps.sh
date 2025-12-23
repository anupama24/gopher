#!/bin/bash
# =============================================================================
# PRS-CS Workflow Script
# Simulate GWAS with differential privacy preprocessing and compute PRS
# =============================================================================

# -------------------------
# Clone PRS-CS repository
# -------------------------
if [[ ! -d "PRScs" ]]; then
    git clone https://github.com/getian107/PRScs.git PRScs
fi
PRS_SCRIPT="PRScs/PRScs.py"

# -------------------------
# Required arguments
# -------------------------
pheno=$1       # phenotype name
t=$2           # total sample size
seed=$3        # random seed
geno_file=$4   # path to genotype PGEN prefix
sam=$5         # sample index or ID
LD=$6          # LD reference: "1KG", "ukb", or custom path
outdir=$7
shift 7        # shift past first 6 positional args
eps_list=("$@") # remaining arguments: epsilon values


# -------------------------
# Debug info
# -------------------------
echo "Phenotype: $pheno"
echo "Total samples: $t"
echo "Seed: $seed"
echo "Genotype file: $geno_file"
echo "LD reference: $LD"
echo "Epsilon list: ${eps_list[*]}"
echo "-----------------------------"

# -------------------------
# Download or set LD reference
# -------------------------
if [[ "$LD" == "1KG" ]]; then
    # Download 1KG LD blocks (EUR example)
    curl -O https://personal.broadinstitute.org/hhuang/public/PRS-CSx/Reference/1KG/ldblk_1kg_eur.tar.gz
    tar xvfz ldblk_1kg_eur.tar.gz --no-same-owner
    rm ldblk_1kg_eur.tar.gz
    refLDDir="ldblk_1kg_eur"

elif [[ "$LD" == "ukb" ]]; then
    curl -O https://personal.broadinstitute.org/hhuang/public/PRS-CSx/Reference/UKBB/ldblk_ukbb_eur.tar.gz
    tar xvfz ldblk_ukbb_eur.tar.gz --no-same-owner
    rm ldblk_ukbb_eur.tar.gz
    refLDDir="ldblk_ukbb_eur"
else
    refLDDir="$LD"
fi


# -------------------------
# Prepare genotype data
# -------------------------
plink2 \
    --pfile ${geno_file} \
    --threads 16 \
    --chr 1-22 \
    --make-bed \
    --out ${geno_file}_prs_sampled;
   

# -------------------------
# Limit threading for PRS-CS
# -------------------------
export N_THREADS=1
export MKL_NUM_THREADS=$N_THREADS
export NUMEXPR_NUM_THREADS=$N_THREADS
export OMP_NUM_THREADS=$N_THREADS

# -------------------------
# Main loop over epsilon values
# -------------------------
for eps in "${eps_list[@]}"; do
    echo "Running PRS-CS with eps = $eps"

    # GWAS with PLINK2
    plink2 \
        --pfile ${geno_file} \
        --threads 16 \
        --pheno ${outdir}/LP/LP_sample_${sam}_eps_${eps}_${pheno}.txt \
        --pheno-name ${pheno} \
        --variance-standardize ${pheno} \
        --glm allow-no-covars \
        --ci 0.95 \
        --vif 99999999 \
        --max-corr 0.99999999 \
        --freq \
        --out ${outdir}/LP/PrivGWAS_LP_sample_${sam}_eps_${eps}_${pheno}
    
    
    # Convert GWAS output to PRS-CS SST format
    input_stat="${outdir}/LP/PrivGWAS_LP_sample_${sam}_eps_${eps}_${pheno}.${pheno}.glm.linear"
    sst_file="${outdir}/LP/PRS_LP_sam_${sam}_eps_${eps}_sst_file_${pheno}.txt"
   
    # awk 'BEGIN {FS="\t"; OFS="\t"} NR==1 {print "SNP", "A1", "A2", "BETA", "P"; next} {print $3, $7, $8, $12, $17}' "${input_stat}" > "${sst_file}";
    awk 'BEGIN {FS="\t"; OFS="\t"}
    NR==1 {
        print "SNP", "A1", "A2", "BETA", "P";
        next
    }
    $12!="NA" && $17!="NA" && $12!="" && $17!="" {
        print $3, $7, $8, $12, $17
    }' "${input_stat}" > "${sst_file}"

    for chr in $(seq 1 22); 
    do
        # Run PRS-CS per chromosome in parallel
        python3 ${PRS_SCRIPT} --ref_dir=${refLDDir} \
                       --bim_prefix=${geno_file}_prs_sampled \
                       --sst_file=${sst_file} \
                       --n_gwas=${t} \
                       --out_dir="${outdir}/PRS/PRS_LP_sam_${sam}_eps_${eps}_${pheno}" \
                       --chrom=${chr} \
                       --n_iter=1000 \
                       --n_burnin=500 \
                       --beta_std=True &
    done      

    wait $(jobs -p)

    # Concatenate PRS-CS output across chromosomes
    start_index=1
    end_index=22
    output_file="${outdir}/PRS/PRS_LP_sam_${sam}_eps_${eps}_${pheno}_pst_eff_a1_b0.5_phiauto.txt"
    > "${output_file}"

    for i in $(seq 1 22);
    do
        filename="${outdir}/PRS/PRS_LP_sam_${sam}_eps_${eps}_${pheno}_pst_eff_a1_b0.5_phiauto_chr${i}.txt"
        if [ -f "$filename" ]; then
            echo "Concatenating $filename"
            cat "$filename" >> "$output_file"
            rm "$filename"
        else
            echo "File $filename not found."
        fi
    done
    echo "File names have been written to $output_file."

    # Compute PRS scores
    plink2 --pfile ${geno_file} \
           --threads 16 \
           --score ${output_file} 2 4 6 cols=+scoresums variance-standardize \
           --out ${outdir}/PRS/LP_PRS_sam_${sam}_eps_${eps}_${pheno};
           
    # Cleanup intermediate files       
    rm -f \
        ${outdir}/LP/LP_sample_${sam}_eps_${eps}_${pheno}.txt \
        ${outdir}/LP/PrivGWAS_LP_sample_${sam}_eps_${eps}_${pheno}.* \
        ${outdir}/LP/PRS_LP_sam_${sam}_eps_${eps}_sst_file_${pheno}.txt \
        ${output_file}

    rm -rf "${outdir}/PRS/PRS_LP_sam_${sam}_eps_${eps}_${pheno}"

done

# Final cleanup
rm -rf PRScs
rm -rf "${refUKBDir}"
rm -f ${geno_file}_prs_sampled.*

# rm -f plink2
# rm -f vcf_subset
    
    

