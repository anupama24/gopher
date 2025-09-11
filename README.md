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

###Input Data

### Example Data
We provide an example synthetic dataset in the `example_data/` directory. 

### Main Input Data Files

- `geno/chr[1-22].{pgen, psam, pvar}`: PGEN-format genotype data files for chromosomes 1 through 22.
- `pheno.txt`: Phenotype file; each line contains the phenotype value for each sample listed in the corresponding `.psam` file.
- `sample_keep.txt`: A list of sample IDs (from the `.psam` file) to include in the analysis. This is used with the `--keep` flag in PLINK2. (See [PLINK2 file specification](https://www.cog-genomics.org/plink/2.0/input#keep) for more details.)

### Data Preprocessing



### Running GOPHER
To run the phenotype randomization mechanisms with differential privacy guarantees:


### Output
Differentially private phenotype values are saved in the specified output directory, ready for downstream analysis with formal privacy guarantees.

