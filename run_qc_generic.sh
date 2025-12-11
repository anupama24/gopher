#!/bin/bash
#===============================================================================
# Title: run_qc_generic.sh
# Description: Generic PLINK2 QC pipeline for any genotype dataset.
#              Supports:
#                - Single dataset QC
#                - Chromosomes 1–22 QC + merging
#                - Optional variant list filtering
#                - Optional sample filtering
#
# Usage:
#   bash run_qc_generic.sh \
#       --mode {single|chr} \
#       --input_dir <dir> \
#       --pattern <file_prefix_or_chr_pattern> \
#       [--variant_file <variants_to_keep.txt>] \
#       [--keep <sample_list>] \
#       --outdir <output_dir> \
#       --tag <name>
#
# Pattern with {chr} is required only in --mode chr.
#
#===============================================================================

threads=${THREADS:-16}

# -----------------------------
# Parse arguments
# -----------------------------
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift ;;
        --input_dir) input_dir="$2"; shift ;;
        --pattern) pattern="$2"; shift ;;
        --variant_file) variant_file="$2"; shift ;;
        --keep) keep_file="$2"; shift ;;
        --outdir) outdir="$2"; shift ;;
        --tag) tag="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# -----------------------------
# Validate required arguments
# -----------------------------
if [[ -z "$mode" || -z "$input_dir" || -z "$outdir" || -z "$tag" ]]; then
    echo "Usage: bash run_qc_generic.sh --mode {single|chr} --input_dir DIR --outdir DIR --tag NAME"
    exit 1
fi

# -----------------------------
# Create output directory only if not existing
# -----------------------------
if [[ ! -d "$outdir" ]]; then
    mkdir "$outdir" || { echo "ERROR: Cannot create $outdir"; exit 1; }
fi
# cd "$outdir" || { echo "ERROR: Cannot access $outdir"; exit 1; }

# =====================================================================
# FUNCTION: Load dataset → convert to BED → run QC → output BED
# =====================================================================

run_qc() {

    detect_dataset_type() {
        local prefix="$1"
        
        # Look for PGEN
        if [[ -f "${prefix}.pgen" ]]; then
            echo "pgen"
            return
        fi
    
        # Look for BED
        if [[ -f "${prefix}.bed" ]]; then
            echo "bed"
            return
        fi
    
        # Look for BGEN
        if [[ -f "${prefix}.bgen" ]]; then
            echo "bgen"
            return
        fi
    
        # Nothing matched
        echo "none"
    }
 
    local input_prefix=$1
    local out_prefix=$2
    # echo ${out_prefix}
    # -----------------------------
    # Detect dataset type based on actual files
    # -----------------------------
    dtype=$(detect_dataset_type "$input_prefix")
    echo ${dtype}
    
    if [[ "$dtype" == "pgen" ]]; then
        echo "Detected PGEN dataset: $input_prefix"
        plink2 --pfile "$input_prefix" --make-bed --out "${out_prefix}_tmp"
    
    elif [[ "$dtype" == "bed" ]]; then
        echo "Detected BED dataset: $input_prefix"
        plink2 --bfile "$input_prefix" --make-bed --out "${out_prefix}_tmp"
    
    elif [[ "$dtype" == "bgen" ]]; then
        echo "Detected BGEN dataset: $input_prefix"
        plink2 --bgen "${input_prefix}.bgen" ref-first \
               --sample "${input_prefix}.sample" \
               --make-bed --out "${out_prefix}_tmp"
    
    else
        echo "ERROR: No .pgen/.bed/.bgen files found for prefix: $input_prefix"
        exit 1
    fi


    # -----------------------------
    # Confirm PLINK succeeded
    # -----------------------------
    if [[ ! -f "${out_prefix}_tmp.bed" ]]; then
        echo "ERROR: PLINK failed to convert input for prefix: $input_prefix"
        exit 1
    fi

    src="${out_prefix}_tmp"

    # -----------------------------
    # Apply QC
    # -----------------------------
    plink2 \
        --bfile "$src" \
        --min-alleles 2 --max-alleles 2 \
        --geno 0.01 \
        --maf 0.01 \
        --hwe 1e-10 \
        --snps-only just-acgt \
        $( [[ -n "$variant_file" ]] && echo "--extract $variant_file" ) \
        --threads "$threads" \
        --make-bed \
        --out "$out_prefix"

    rm -f "${out_prefix}_tmp".*
}

# =====================================================================
# MODE 1: SINGLE DATASET QC → Final PGEN
# =====================================================================
if [[ "$mode" == "single" ]]; then

    input_prefix="${input_dir}/${pattern}"
    out_prefix="${outdir}/qc_${tag}"

    # input_prefix="${input_dir}/${pattern}"
    echo "Running QC on single dataset: $input_prefix"

    run_qc "$input_prefix" "$out_prefix"

    # -----------------------------
    # Convert to PGEN (final output)
    # -----------------------------
    if [[ -n "$keep_file" ]]; then
        plink2 \
            --bfile "$out_prefix" \
            --keep "$keep_file" \
            --mind 0.01 \
            --thin-count 500000 \
            --make-pgen \
            --out "${outdir}/${tag}_qc_thinned"
    else
        plink2 \
            --bfile "$out_prefix" \
            --mind 0.01 \
            --thin-count 500000 \
            --make-pgen \
            --out "${outdir}/${tag}_qc_thinned"
    fi


    echo "Final PGEN output: ${tag}_qc_thinned.pgen"
    exit 0
fi

# =====================================================================
# MODE 2: CHR-WISE QC → BED MERGE → Final PGEN
# =====================================================================
if [[ "$mode" == "chr" ]]; then

    echo "Running chromosome-wise QC..."

    for chr in {1..22}; do
        (
            echo "[CHR $chr] Starting..."
            chr_prefix=$(echo "$pattern" | sed "s/{chr}/$chr/g")
            run_qc "${input_dir}/${chr_prefix}" "${outdir}/qc_chr${chr}"
        ) &
    done
    wait

    echo "Building merge list..."
    rm -f merge_list.txt
    for chr in $(seq 2 22); do
        [[ -f qc_chr${chr}.bed ]] \
            && echo "${outdir}/qc_chr${chr}.bed ${outdir}/qc_chr${chr}.bim ${outdir}/qc_chr${chr}.fam" >> merge_list.txt
    done

    echo "Merging chromosomes (BED merge)..."
    plink \
        --bfile "${outdir}/qc_chr1" \
        --pmerge-list merge_list.txt \
        --threads "$threads" \
        --mind 0.01 \
        --sample-inner-join \
        --merge-max-allele-ct 2 \
        --delete-pmerge-result \
        --make-pgen \
        --out "${outdir}/qc_allChrs"


    if [[ -n "$keep_file" ]]; then
        plink2 \
            --pfile "${outdir}/qc_allChrs" \
            --keep "$keep_file" \
            --thin-count 500000 \
            --make-pgen \
            --out "${outdir}/${tag}_qc_thinned"
    else
        plink2 \
            --pfile "${outdir}/qc_allChrs" \
            --thin-count 500000 \
            --make-pgen \
            --out "${outdir}/${tag}_qc_thinned"
    fi

    rm -f qc_allChrs.*
    rm -f merge_list.txt
    
    echo "Final PGEN output: ${tag}_qc_thinned.pgen"
    exit 0
fi

echo "ERROR: --mode must be 'single' or 'chr'"
exit 1
