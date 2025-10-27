#!/bin/bash
#===============================================================================
# Title: run_qc_chr1_22.sh
# Description: Run PLINK2 QC filtering on imputed genotype data  (UK Biobank dataset)
#              (chromosomes 1–22), merge, and subset to unrelated/related samples.
#
# Usage:
#   bash run_qc_chr1.sh <input_dir> <output_dir> <file_prefix> <keep_file> <tag>
#
# Example:
#   bash run_qc_chr1_22.sh "/Bulk/Imputation/UKB_imputation" "/data" "ukb22828_c${i}_b0_v3.bgen" "keep_related.txt" "related"
#
#===============================================================================


# ---- Parse arguments ----
input_dir=$1       # Directory with input .bgen/.sample files
output_dir=$2      # Directory to save QC outputs
hapmap_file=$3     # Hapmap variants file name
file_prefix=$4     # File prefix
keep_file=$5       # File with list of samples to be extracted
tag=$6             # Save final .pgen files with specified tag



# ---- Validate input ----
if [[ -z "$input_dir" || -z "$output_dir" || -z "$file_prefix" || -z "$file_prefix" || -z "$keep_file" || -z "$tag" ]]; then
    echo "Usage: $0 <input_dir> <output_dir> <hapmap_file> <file_pattern_with_{chr}> <keep_file> <tag>"
    echo "Example pattern: ukb22828_c{chr}_b0_v3"
    exit 1
fi

# ---- Configuration ----
threads=${THREADS:-16}
keep_variants="keep_variants.txt"
chr_start=1
chr_end=22

echo "=== Starting  QC pipeline ==="
echo "Input directory:  $input_dir"
echo "Output directory: $output_dir"
echo "HAPMAP_FILE: $hapmap_file"
echo "Prefix:           $file_prefix"
echo "Keep file:        $keep_file"
echo "Tag:              $tag"
echo "Threads:          $threads"
echo "--------------------------------------"

# ---- Generate list of HapMap3 variants if missing ----
if [[ ! -f "$keep_variants" ]]; then
    if [[ ! -f "$hapmap_file" ]]; then
        echo "Error: HapMap file '$hapmap_file' not found."
        exit 1
    fi
    echo "Extracting variant IDs from HapMap3 reference..."
    awk '{print $2}' "$hapmap_file" > "$keep_variants"
fi

# ---- Per-chromosome QC filtering ----
echo "Running QC on imputed genotypes using pattern: '$file_prefix'"

for chr in $(seq "$chr_start" "$chr_end"); do
    (
        echo "Processing chromosome ${chr}..."

        # Dynamically substitute {chr} in the user-provided pattern
        prefix=$(echo "$file_prefix" | sed "s/{chr}/${chr}/g")

        input_bgen="${input_dir}/${prefix}.bgen"
        input_sample="${input_dir}/${prefix}.sample"

        if [[ ! -f "$input_bgen" || ! -f "$input_sample" ]]; then
            echo "Warning: Missing files for chromosome ${chr} (expected: ${input_bgen})"
            continue
        fi

        echo "Using files:"
        echo "   BGEN:   $input_bgen"
        echo "   SAMPLE: $input_sample"

        # Convert BGEN → PGEN
        plink2 \
            --bgen "$input_bgen" ref-first \
            --sample "$input_sample" \
            --make-pgen \
            --out "ukb_imp_chr${chr}"

        # Apply QC filters
        plink2 \
            --pfile "ukb_imp_chr${chr}" \
            --min-alleles 2 --max-alleles 2 \
            --geno 0.01 \
            --maf 0.01 --max-maf 0.99 \
            --hwe 1e-10 \
            --snps-only just-acgt \
            --exclude-if-info 'R2<=0.8' \
            --extract "$keep_variants" \
            --make-bed \
            --out "ukb_qc_chr${chr}"

        rm -f "ukb_imp_chr${chr}"* 
    ) &
done

# Wait for all background jobs to finish
wait
echo "All chromosomes processed."

# ---- Merge per-chromosome QC data ----
echo "Merging QC’d chromosomes..."
merge_list="merge_list.txt"
rm -f "$merge_list"

for chr in $(seq 2 "$chr_end"); do
    echo "ukb_qc_chr${CHR}.bed ukb_qc_chr${CHR}.bim ukb_qc_chr${CHR}.fam" >> "$merge_list"
done

plink2 \
    --bfile ukb_qc_chr1 \
    --pmerge-list "$merge_list" \
    --threads "$threads" \
    --mind 0.01 \
    --sample-inner-join \
    --merge-max-allele-ct 2 \
    --delete-pmerge-result \
    --make-pgen \
    --out "ukb_qc_allChrs"


# ---- Subset and thin data ----
echo "Thinning and subsetting data..."
plink2 \
    --pfile "ukb_qc_allChrs" \
    --keep "$keep_file" \
    --thin-indiv-count 100000 \
    --thin-count 500000 \
    --threads "$threads" \
    --make-pgen \
    --out "ukb_qc_thinned_${tag}_samples"

echo "=== PLINK2 QC pipeline completed successfully ==="







