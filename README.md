# GOPHER: Optimization-based Phenotype Randomization Mechanisms for Genome-Wide Association Studies with Differential Privacy

Genome-wide association studies (GWAS) are essential in biomedical research for identifying genetic factors linked to health and disease. However, publicly releasing GWAS association statistics poses privacy risks, such as potential disclosure of individual participation or sensitive phenotypic information (e.g., disease status).

Differential privacy (DP) provides a rigorous framework for protecting individual privacy and enabling broader sharing of GWAS results. Existing DP methods either introduce excessive noise or limit the scope of analysis. GOPHER introduces novel DP mechanisms for the private release of full GWAS statistics, improving accuracy and utility.

This repository implements GOPHER's optimization-based phenotype randomization techniques, as described in:

**GOPHER: Optimization-based Phenotype Randomization Mechanisms for Genome-Wide Association Studies with Differential Privacy**  
*Authors:   
*Journal/Conference, Year*

---

## Installation

### Dependencies

GOPHER requires the following software and libraries:

- Python (>=3.8)  
- [NumPy](https://numpy.org/)  
- [SciPy](https://scipy.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [cvxopt](https://cvxopt.org/) (for convex optimization)  
- [pgenlib](https://pgenlib.org/) (for handling PLINK2 PGEN genotype files)  
- [PLINK2](https://www.cog-genomics.org/plink/2.0/) (for genotype data manipulation)

You can install most Python dependencies via pip:

```bash
pip install numpy scipy scikit-learn h5py cvxopt


#Installing pgenlib
#pgenlib is a C++ library with Python bindings for efficient processing of PLINK2 genotype files.
pip install pgenlib

#Installing PLINK2
#Download PLINK2 binaries or source from the official site. Ensure the plink2 executable is in your system PATH.
wget -q https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_latest.zip > /dev/null 2>&1
unzip -q plink2_linux_x86_64_latest.zip > /dev/null 2>&1
chmod +x plink2
```

### Clone the repository
```bash
git clone https://github.com/username/gopher
cd gopher
```

### Resource Requirements
All experiments require access to a high-performance multi-processor system with at least 48 CPU cores and 256 GB of RAM. For GOPHER-MultiQP, a cloud instance equivalent to an Amazon EC2 g4dn.16xlarge (64 vCPUs and 256 GB RAM) is recommended to ensure optimal performance.


## Usage

## Input Data

### Example Data  
The `data/` directory contains a **minimal, self-contained subset of 1000 Genomes Phase 3 (1000G)** data.  
This lightweight dataset is provided to support **quick testing, validation, and illustration** of the workflow without requiring the full 1000G dataset.

---

### Download and decompress 1000 Genomes phase 3 data
1000 Genomes phase III (1000GenomesIII) is available in
[PLINK 2 binary format](https://www.cog-genomics.org/plink/2.0/input#pgen) at https://www.cog-genomics.org/plink/2.0/resources#1kg_phase3.
In addition, a sample file with information about the individuals' ancestry
is available. 
The following code chunk downloads and decompresses the data.
The genome build of
these files is the same as the original release of the 1000GenomesIII, namely
CGRCh37. 

```bash
cd data

pgen=https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1
pvar=https://www.dropbox.com/s/vx09262b4k1kszy/all_hg38.pvar.zst?dl=1
sample=https://www.dropbox.com/s/2e87z6nc4qexjjm/all_hg38.psam?dl=1

wget $pgen
mv 'all_hg38.pgen.zst?dl=1' all_hg38.pgen.zst
plink2 --zst-decompress all_hg38.pgen.zst > all_hg38.pgen

wget $pvar
mv 'all_hg38.pvar.zst?dl=1' all_hg38.pvar.zst

wget $sample
mv 'all_hg38.psam?dl=1' all_hg38.psam
```

### Main UKBiobank Input Data Files

The pipeline expects **imputed UK Biobank genotype data in BGEN format**, organized as:

- `data/chr[1–22].bgen`  
- `data/chr[1–22].sample`  
These are the standard per-chromosome imputed genotype files for chromosomes **1–22**.  
The pipeline automatically converts each BGEN file into PLINK2 PGEN format during QC.

- `pheno.txt`: Phenotype file; each line contains the phenotype value for each sample listed in the corresponding `.psam` file.
- `sample_keep.txt`: A list of sample IDs (from the `.psam` file) to include in the analysis. This is used with the `--keep` flag in PLINK2. (See [PLINK2 file specification](https://www.cog-genomics.org/plink/2.0/input#keep) for more details.)

## Pipeline Overview

The GOPHER pipeline automates the full process from raw genotype data to private GWAS results. It includes quality control, phenotype simulation, privacy mechanism application, and downstream GWAS analysis. The steps are as follows:

1. **Genotype Quality Control (QC)**  
Genotype data is filtered using the `run_qc_chr1_22.sh` or `run_qc_generic.sh` script. This script applies the following standard QC filters:
  - Remove variants with high missingness (`--geno` threshold)
  - Filter variants with low minor allele frequency (`--maf`)
  - Exclude variants failing Hardy-Weinberg equilibrium (`--hwe`)
  - Retain only bi-allelic SNPs

2. **Phenotype Simulation**  
   Phenotypes are simulated from the genotype data based on a specified heritability parameter (`h2`). This step allows for controlled experimentation using synthetic data with known genetic architecture.

3. **Baseline GWAS (Non-Private)**  
   A standard GWAS is run on the original (non-private) phenotype using PLINK2. This serves as a baseline for evaluating the impact of privacy-preserving mechanisms.

4. **Polygenic Risk Score (PRS) and Prior Estimation**  
   PRS are computed using summary statistics from the (non-private or previous) GWAS. These scores are then used to estimate marginal priors for phenotypes, which improve inference under certain privacy mechanisms.

5. **Privacy Mechanism Application**  
   One of the supported privacy mechanisms is applied to the phenotype to ensure differential privacy:
   - **RR** (Randomized Response)
   - **Lap** (Laplace Mechanism)
   - **GOPHER-LP** (RR-on-bins)
   - **GOPHER-QP** 
   - **GOPHER-MultiLP** 
   - **GOPHER-MultiQP** 

   These mechanisms generate privatized versions of the phenotype that can be shared or analyzed while preserving individual privacy.

6. **Private GWAS Execution**  
   For each specified privacy budget (`ε`), a private GWAS is run using PLINK2 on the privatized phenotypes. This produces differential privacy-compliant association statistics for downstream interpretation.

---

Each step is configurable, allowing researchers to adapt the pipeline for their own datasets and privacy constraints.

### 1. Data Preprocessing
Before running simulations or downstream analyses, the genotype data are preprocessed to ensure quality and compatibility:

- **Format conversion:** Convert genotype files into PLINK2 PGEN format (if needed) for efficient computation.
- **Quality control (QC):** Apply standard QC filters.
- **Sample selection:** Subset to a set of unrelated or target samples using a provided `keep` file.
- **Output:** Cleaned, filtered `.pgen/.pvar/.psam` files ready for phenotype simulation or GWAS analysis.

The repository supports both UK Biobank imputed data and generic genotype datasets. The `run_qc_chr1_22.sh` script is specifically designed for **UK Biobank** imputed genotype data. Following are the features of this script:
- Variant selection: Optionally subset to HapMap3 or other reference variants.
- Per-chromosome QC on chromosomes 1–22
- Filtering on genotype quality, minor allele frequency, Hardy-Weinberg equilibrium, and SNPs
- Merging all chromosomes
- Optional subsetting of individuals (e.g., unrelated or related)

**Usage:**
```bash
bash run_qc_chr1_22.sh <input_dir> <output_dir> <hapmap_file> <file_prefix> <keep_file> <tag>
```
The `run_qc_generic.sh` script provides a **flexible QC pipeline** that works with any genotype dataset, including **PGEN, BED, or BGEN** formats. It gives the flexibility to do  the following:
- QC for a single dataset or per chromosome (1–22) and then merging them
- Optional variant filtering via a variant list
- Optional sample subsetting via a sample list

**Usage:**

```bash
bash run_qc_generic.sh \
    --mode {single|chr} \
    --input_dir <dir> \
    --pattern <file_prefix_or_chr_pattern> \
    [--variant_file <variants_to_keep.txt>] \
    [--keep <sample_list>] \
    --outdir <output_dir> \
    --tag <name>
```

---

### 2. Phenotype Simulation
IF working with simulated phenotypes, they are simulated from the preprocessed genotype data based on a user-specified **heritability (`h2`)**. This allows controlled experimentation with synthetic traits of known genetic architecture.
- **Inputs:** 
  - Genotype files (`.pgen`, `.pvar`, `.psam`)
  - Heritability parameter (`h2`)
- **Process:**
  - Assign random effect sizes to a subset of causal variants
  - Generate phenotype values by combining genetic effects with normally distributed environmental noise
- **Outputs:** 
  - Phenotype file (e.g., `simulated_pheno.txt`) compatible with PLINK2 or downstream analysis pipelines

**Usage:**
```
python simulate_phenotypes.py [options]
```
**Options:**
`--data-path`: Directory containing `.pgen`, `.pvar`, and `.psam` files.
`--geno-prefix`: Prefix of genotype files.
`--pheno-file` : Output phenotype file path.
`--h2` : Heritability for phenotype simulation.
`--seed`: Random seed for reproducibility.
`--chunk-size`: Number of variants processed per chunk.

---

This workflow demonstrates how to run QC on BGEN genotype files and simulate phenotypes using the provided scripts.

**Example:**
```bash
# 1. Run QC on UKB BGEN files and generate a PLINK2 dataset
bash run_qc_chr1_22.sh \
    "/path/to/bgen_files" \
    "/data" \
    "hapmap3_r2_ref.txt" \
    "ukb22828_c{chr}_b0_v3" \
    "keep_unrelated.txt" \
    "unrelated"

# 2. Simulate phenotypes with 0.8 heritability
python simulate_phenotypes.py \
    --data-path /data \
    --geno-prefix ukb_qc_thinned_unrelated_samples \
    --h2 0.8
```
    
# Running GOPHER
## Baseline Methods

The `baselineMethods.py` script implements **Laplace** and **Randomized Response (RR)** mechanisms to produce privacy-preserving versions of phenotypes. This allows experimentation with differential privacy in GWAS or synthetic datasets.

---

### Usage Example

```
python baselineMethods.py \
    --pheno_file simulated_phenotypes.txt \
    --pheno Sim_Y_100 \
    --mech both \
    --eps_list 1.0,2.0 \
    --dest ./results \
    --seed 1234 \
    --bins 100 \
    --h2 0.8
```

## GOPHER-LP: 
The `runLPMech.py` script implements the **GOPHER-LP mechanism** (Randomized Response on discretized bins) to generate privacy-preserving phenotypes across multiple privacy budgets (ε). It can be applied to any simulated or real phenotype dataset in tab-delimited format.

---

### Usage Example

```
python runLPMech.py \
    --pheno_file ./data/simulated_phenotypes.txt \
    --pheno Sim_Y_100 \
    --eps_list 1.0,2.0 \
    --dest ./results/LP \
    --bins 100 \
    --seed 1234 \
    --sam 10000 \
    --eps 0.1
```

## GOPHER-MultiLP Pipeline

The `run_gopher_multiLP.sh` wrapper script orchestrates the **full GOPHER-MultLP workflow**, including:

1. Differentially private (DP) mean and variance estimation of full phenotype vector  
2. Subsampling individuals and phenotypes for PRScs
3. GOPHER-LP phenotype privatization on subsampled phenotype vector
4. PRS-CS polygenic score construction  based on GWAS results generated using Step 3 output.
5. GOPHER-MultiLP mechanism for final DP phenotype generation  

---

### Script Overview

```bash
bash run_gopher_multiLP.sh \
    <pheno_file> \
    <pheno> \
    <total_sample_size> \
    <seed> \
    <genotype_prefix> \
    <bins> \
    <subsample_size> \
    <LD_reference> \
    <pheno_type> \
    <outdir> \
    <eps1> <eps2> ...
```
### Usage Example

```
bash run_gopher_multiLP.sh \
    ./data/phenotypes.txt \
    Sim_Y_100 \
    10000 \
    1234 \
    ./data/genotype \
    80 \
    1234 \
    1KG \
    sim \
    ./results \
    1.0 3.0 5.0
```
## GOPHER-All-in-One Pipeline

The `run_gopher_all.sh` script provides an **end-to-end wrapper** that runs **GOPHER-LP, GOPHER-MultiLP, and GOPHER-MultiQP** in a single workflow. It computes differentially private (DP) mean and variance, subsamples phenotypes, calls PRS-CS, and generates final DP phenotypes across a list of privacy budgets (ε). It uses the same DP mean and variance across all the GOPHER methods.

