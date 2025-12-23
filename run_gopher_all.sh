#!/bin/bash
# ===================================================================================================
# Wrapper script to compute DP mean/var, sample phenotypes, call PRS-CS, and call all GOPHER methods
# ===================================================================================================

# -------------------------
# Required inputs
# -------------------------
pheno_file=$1      # phenotype file
pheno=$2           # phenotype name
t=$3               # total sample size
seed=$4            # random seed
geno_file=$5       # genotype prefix
bins=$6            # number of bins
sam=$7             # number of samples
LD=$8              # LD reference ("1KG", "ukb", or path)
tag=$9             # phenotype type: real or sim
outdir=${10}
shift 10
eps_list=("$@")    # remaining arguments are epsilons

# -------------------------
# Debug info
# -------------------------
echo "Phenotype file: $pheno_file"
echo "Phenotype: $pheno"
echo "Sample size: $t"
echo "Seed: $seed"
echo "Genotype file: $geno_file"
echo "Bins: $bins"
echo "Sample subset size: $sam"
echo "LD reference: $LD"
echo "Phenotype type: $tag"
echo "Output dir: $outdir"
echo "Epsilon list: ${eps_list[*]}"
echo "-----------------------------"

if [[ ! -d "$outdir" ]]; then
    mkdir "$outdir" || { echo "ERROR: Cannot create $outdir"; exit 1; }
fi

output=$(python3 calcDPMeanVar.py  --pheno_file ${pheno_file} --pheno ${pheno})

# Split output into variables
mean=$(echo $output | cut -d' ' -f1)
var=$(echo $output | cut -d' ' -f2)

echo "DP mean: $mean"
echo "DP variance: $var"

# -------------------------
# Sample phenotypes
# -------------------------
lp_file="${geno_file}_${pheno}_lp_${sam}.txt"

echo "Processing for sample size= ${t}"
echo -e "#FID\tIID" > temp_sample_${sam}.txt
tail -n +2 ${geno_file}.psam | shuf --random-source=<(yes ${seed}) -n "${sam}" | awk '{print $1, $2}' >> temp_sample_${sam}.txt

head -n 1 ${pheno_file} > ${lp_file}
awk 'NR > 1 {print $1 "\t" $2}' temp_sample_${sam}.txt | grep -Ff - ${pheno_file} >> ${lp_file}

# --------------------------------------------------
# Run GOPHER-LP on sampled phenotype subset
# --------------------------------------------------
python3 runLPMech.py --dest ${outdir} --pheno_file ${lp_file} --pheno ${pheno} --mean ${mean} --var ${var} --bins ${bins} --sam ${sam} --seed ${seed} --eps_list "${eps_list[*]}";  

# -------------------------
# Call PRS-CS workflow script
# -------------------------
bash doPRS_eps.sh \
    "${pheno}" \
    "${t}" \
    "${seed}" \
    "${geno_file}" \
    "${sam}" \
    "${LD}" \
    "${outdir}" \
    "${eps_list[@]}"


for eps in "${eps_list[@]}"; do
    # Construct PRS score file path (matches output in doPRS_eps.sh)
    score_file="${outdir}/PRS/LP_PRS_sam_${sam}_eps_${eps_list[0]}_${pheno}.sscore"
    
    # -------------------------
    # Run GOPHER-MultiLP mechanism
    # -------------------------
    python3 multiLPMech.py \
        --dest "${outdir}" \
        --pheno_file "${pheno_file}" \
        --pheno "${pheno}" \
        --eps_list "${eps}" \
        --seed "${seed}" \
        --bins "${bins}" \
        --lp_file "${lp_file}" \
        --score_file "${score_file}" \
        --sam "${t}" \
        --mean "${mean}" \
        --var "${var}" \
        --tag $tag \
        --h2 0.8 \

    # -------------------------
    # Run GOPHER-MultiQP mechanism
    # -------------------------
    python3 multiQPMech.py \
        --dest "${outdir}" \
        --geno_file "${geno_file}" \
        --pheno_file "${pheno_file}" \
        --pheno "${pheno}" \
        --eps_list "${eps}" \
        --seed "${seed}" \
        --bins "${bins}" \
        --lp_file "${lp_file}" \
        --score_file "${score_file}" \
        --sam "${t}" \
        --mean "${mean}" \
        --var "${var}" \
        --tag $tag \
        --h2 0.8 \
        
done    
echo "Completed DP phenotype generation, PRS computation, and MultiLP,MultiQP processing."

# -------------------------
# Cleanup
# -------------------------
rm "temp_sample_${sam}.txt"
rm -f "${lp_file}"

# --------------------------------------------------
# Run GOPHER-LP on full phenotype data
# --------------------------------------------------
python3 runLPMech.py --dest ${outdir} --pheno_file ${pheno_file} --pheno ${pheno} --mean ${mean} --var ${var} --bins ${bins} --sam ${t} --seed ${seed} --eps_list "${eps_list[*]}";  

echo "All done!"