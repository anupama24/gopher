#!/bin/bash

# # Install plink
# wget -q https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_latest.zip > /dev/null 2>&1
# unzip -q plink2_linux_x86_64_latest.zip > /dev/null 2>&1
# chmod +x plink2
# git clone https://github.com/getian107/PRScs.git PRScs
curl -O https://personal.broadinstitute.org/hhuang/public/PRS-CSx/Reference/UKBB/ldblk_ukbb_eur.tar.gz
tar xvfz ldblk_ukbb_eur.tar.gz --no-same-owner

# rm plink2_linux_x86_64_latest.zip
rm ldblk_ukbb_eur.tar.gz


# Set static paths
data_field="ukb_imp_qc"
PRS="PRScs/PRScs.py"
refUKBDir="ldblk_ukbb_eur"

# Required arguments
pheno_file=$1
pheno=$2
t=$3
seed=$4
geno_file=$5
bins=$6
sam=$7
shift 7

# Remaining arguments are interpreted as eps_list
eps_list=("$@")

# Debug info
echo "Phenotype file: $pheno_file"
echo "Phenotype: $pheno"
echo "t: $t"
echo "Seed: $seed"
echo "Genotype file: $geno_file"
echo "Bins: $bins"
echo "Eps list: ${eps_list[*]}"
echo "-----------------------------"


# sam=$((t / 5))

lp_file="${geno_file}_${pheno}_lp_${sam}.txt"

echo "Processing for sample size= ${t}"
echo -e "#FID\tIID" > temp_sample_${sam}.txt
tail -n +2 ${geno_file}.psam | shuf --random-source=<(yes ${seed}) -n "${sam}" | awk '{print $1, $2}' >> temp_sample_${sam}.txt

head -n 1 ${pheno_file} > ${lp_file}
awk 'NR > 1 {print $1 "\t" $2}' temp_sample_${sam}.txt | grep -Ff - ${pheno_file} >> ${lp_file}

python3 phenoSampleLP.py --pheno_file ${lp_file} --pheno ${pheno} --bins ${bins} --sam ${sam} --seed ${seed} --eps_list "${eps_list[@]}";  

./plink2 \
    --pfile ${geno_file} \
    --threads 16 \
    --make-bed \
    --out ${data_field}_prs_sampled;
   

# mkdir -p PRScs  # Create target folder if it doesn't exist
# dx download /PRScs/* -r -o PRScs/

# mkdir -p ldblk_ukbb_eur  # Create target folder if it doesn't exist
# dx download /code/data/ldblk_ukbb_eur/* -r -o ldblk_ukbb_eur/

# Set the number of threads to use (e.g., 1)
export N_THREADS=1

# Limit threading for MKL, NumExpr, and OpenMP-based operations
export MKL_NUM_THREADS=$N_THREADS
export NUMEXPR_NUM_THREADS=$N_THREADS
export OMP_NUM_THREADS=$N_THREADS


for eps in "${eps_list[@]}"; do
    echo "Running PRS-CS with eps = $eps"

    ./plink2 \
        --pfile ${geno_file} \
        --threads 16 \
        --pheno LP_sample_${sam}_eps_${eps}_${pheno}.txt \
        --pheno-name ${pheno} \
        --variance-standardize ${pheno} \
        --glm allow-no-covars \
        --ci 0.95 \
        --vif 99999999 \
        --max-corr 0.99999999 \
        --freq \
        --out PrivGWAS_LP_sample_${sam}_eps_${eps}_${pheno}
    
    

    input_stat="PrivGWAS_LP_sample_${sam}_eps_${eps}_${pheno}.${pheno}.glm.linear"
    sst_file="PRS_LP_sam_${sam}_eps_${eps}_sst_file_${pheno}.txt"
   
    awk 'BEGIN {FS="\t"; OFS="\t"} NR==1 {print "SNP", "A1", "A2", "BETA", "P"; next} {print $3, $7, $8, $12, $17}' "${input_stat}" > "${sst_file}";
    for chr in $(seq 1 22); 
    do

        python3 ${PRS} --ref_dir=${refUKBDir} \
                       --bim_prefix=${data_field}_prs_sampled \
                       --sst_file=${sst_file} \
                       --n_gwas=${t} \
                       --out_dir="PRS_LP_sam_${sam}_eps_${eps}_${pheno}" \
                       --chrom=${chr} \
                       --n_iter=1000 \
                       --n_burnin=500 \
                       --beta_std=True &
    done      


    wait $(jobs -p)

    start_index=1
    end_index=22
    output_file="PRS_LP_sam_${sam}_eps_${eps}_${pheno}_pst_eff_a1_b0.5_phiauto.txt"
    > "${output_file}"

    for i in $(seq 1 22);
    do
        filename="PRS_LP_sam_${sam}_eps_${eps}_${pheno}_pst_eff_a1_b0.5_phiauto_chr${i}.txt"
        if [ -f "$filename" ]; then
            echo "Concatenating $filename"
            cat "$filename" >> "$output_file"
            rm "$filename"
        else
            echo "File $filename not found."
        fi
    done
    echo "File names have been written to $output_file."

    ./plink2 --pfile ${geno_file} \
           --threads 16 \
           --score ${output_file} 2 4 6 cols=+scoresums variance-standardize \
           --out LP_PRS_sam_${sam}_eps_${eps}_${pheno};
           
    rm -f \
        LP_sample_${sam}_eps_${eps}_${pheno}.txt \
        PrivGWAS_LP_sample_${sam}_eps_${eps}_${pheno}.* \
        PRS_LP_sam_${sam}_eps_${eps}_sst_file_${pheno}.txt \
        ${output_file}

    rm -rf "PRS_LP_sam_${sam}_eps_${eps}_${pheno}"

done

# Final cleanup
rm -rf PRScs
rm -rf "${refUKBDir}"
rm -f ${data_field}_prs_sampled.*
rm "temp_sample_${sam}.txt"
rm -f "${lp_file}"
rm -f plink2
rm -f vcf_subset
    
    

