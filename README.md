
<div align="center">
  <h1>QTLScan</h1>
  <p>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-BSD3-blue.svg" alt="License">
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version">
    </a>
    <a href="https://github.com/swu1019lab/qtlscan">
      <img src="https://img.shields.io/badge/status-active-success" alt="Status">
    </a>
  </p>
</div>

## 📖 Overview

**QTLScan** is a comprehensive bioinformatics toolkit designed to streamline post-GWAS analysis, enable automate the transition from summary statistics to defined **Quantitative Trait Loci (QTL)** intervals using Linkage Disequilibrium (LD) and Union-Find algorithms. Beyond simple mapping, it integrates modules for **Epistatic Interaction Analysis** and **Genetic Network Construction**, enabling researchers to dissect the complex genetic architecture of agronomic traits.

### Key Features

* **⚡ Automated QTL Mapping:** Efficiently merges GWAS signals into defined QTL blocks based on LD patterns.
* **🧬 Epistatic Interaction:** Detects pairwise non-additive effects (epistasis) between identifying loci
* **🕸️ Genetic Networks:** Constructs and visualizes complex trait-associated networks using graph theory layouts (Spring, Kamada-Kawai, ForceAtlas2).
* **📊 High-Quality Visualization:** Generates publication-ready figures including Manhattan plots, LD heatmaps with gene tracks, QTL distribution maps, and QTN haplotype heatmaps.
* **📑 Interactive Reporting:** Produces HTML summaries for rapid exploration of results.


## 🛠️ Installation

### Prerequisites
QTLScan requires **Python 3.8+** and the following external tools in your `$PATH`:
* **[PLINK 1.9](https://www.cog-genomics.org/plink/)** (Required for LD calculation)
* **[GEMMA](https://github.com/genetics-statistics/gemma)** or **[EMMAX](http://genetics.cs.ucla.edu/emmax/)** (Required for GWAS execution)

### Python Dependencies

The tool relies on standard scientific libraries:

  * `pandas`, `numpy`, `scipy`
  * `matplotlib`, `networkx`, `pysam`, `statsmodels`
  * `jinja2` (for HTML reports)
  * `fa2`, `python-igraph` (optional, for advanced network layouts)

### Install via Source
```bash
git clone https://github.com/swu1019lab/qtlscan.git
python -m build
pip install dist/qtlscan-1.0.0.tar.gz
````

## 🚀 Usage Guide

QTLScan is organized into several subcommands. Below are examples based on typical workflows.

### 1\. Phenotype Processing (`phe`)

Utilities to split multi-trait tables, merge files, or inspect distributions.

```bash
# Split a multi-trait CSV into individual phenotype files for GWAS
qtlscan phe split \
    --input data/phenotypes.csv \
    --out-format plink \
    --out-dir data/phe/split \
    --prefix trait

# Check phenotype distribution and normality
qtlscan phe stat \
    --input data/phe/split/trait.Yield.phe \
    --plot-kind hist \
    --fit-curve \
    --out-dir plots
```

### 2\. Run GWAS (`gwas`)

Wrapper for GEMMA or EMMAX to perform association analysis.

```bash
qtlscan gwas \
    --vcf data/genotypes.vcf.gz \
    --phe data/phe/split/trait.Yield.phe \
    --method gemma \
    --out_dir results/gwas \
    --out_name Yield_GWAS
```

### 3\. QTL Identification (`scan`)

The core module. Scans GWAS summary statistics to define QTL blocks based on LD decay.

```bash
qtlscan scan \
    --vcf data/genotypes.vcf.gz \
    --summary results/gwas/Yield_GWAS.assoc.txt \
    --names Yield \
    --gff data/genome_annotation.gff3 \
    --pvalue 1e-6 \
    --min_snps 3 \
    --ld_window_kb 100 \
    --out_dir results/qtl \
    --out_name Yield_QTL
```

### 4\. Genetic Architecture Dissection

#### Network Analysis (`network`)

Construct linkage networks among QTLs to visualize genetic architecture.

```bash
qtlscan network \
    --qtl results/qtl/Yield_QTL.blocks.csv \
    --vcf data/genotypes.vcf.gz \
    --layout spring \
    --min_weight 0.4 \
    --node_size_factor 10 \
    --out_dir results/network \
    --out_name Yield_Network
```

#### Epistasis Analysis (`epistasis`)

Calculate pairwise epistatic interactions between identified QTL lead SNPs.

```bash
qtlscan epistasis \
    --qtl results/qtl/Yield_QTL.blocks.csv \
    --phe data/phe/split/trait.Yield.phe \
    --vcf data/genotypes.vcf.gz \
    --genome data/genome.len \
    --pvalue 0.05 \
    --out_dir results/epistasis \
    --out_name Yield_Epistasis
```

### 5\. Visualization (`plot`)

Generate publication-quality figures.

**Manhattan & QQ Plot:**

```bash
qtlscan plot manhattan \
    --summary results/gwas/Yield_GWAS.assoc.txt \
    --sig_threshold 1e-6 \
    --qq \
    --out_dir plots \
    --out_name Yield_Manhattan
```

**LD Heatmap (Local Block Zoom):**

```bash
qtlscan plot ldheatmap \
    --ld results/qtl/ld/Yield_QTL.ld \
    --blocks results/qtl/Yield_QTL.blocks.csv \
    --chr chrA08 \
    --gff data/genome_annotation.gff3 \
    --out_dir plots \
    --out_name Yield_A08_Heatmap
```

**QTN Genotype Map:**
Visualize allelic distribution across the population for identified loci.

```bash
qtlscan plot qtnmap \
    --qtl results/qtl/Yield_QTL.blocks.csv \
    --vcf data/genotypes.vcf.gz \
    --samples_file data/sample_groups.txt \
    --out_dir plots \
    --out_name Yield_QTN_Map
```

## 📂 Input & Output

### Input Files

  * **VCF:** Standard Variant Call Format (gzipped).
  * **GWAS Summary:** Tab-delimited file containing `chr`, `rs`, `ps`, `p_wald`.
  * **GFF3:** Genome annotation for candidate gene mapping.

### Key Output Files

| File Extension | Description |
| :--- | :--- |
| `.blocks.csv` | Table of identified QTL intervals, lead SNPs, and candidate genes. |
| `.edges.csv` | Network edges representing LD linkage or epistatic interactions. |
| `.html` | Interactive report summary (generated by `qtlscan report`). |
| `.png/.pdf` | High-resolution figures for publication. |


## 🗺️ Roadmap & Ecosystem

  * **Integration:** We recommend using QTLScan in conjunction with **[hastat](https://github.com/swu1019lab/hastat)** for fine-scale haplotype mining of candidate genes.
  * **Future Updates:**
      * Machine learning-based candidate gene prioritization.
      * Pan-genome data format compatibility.



## 📝 Citation

If you use **QTLScan** in your research, please cite:

> **Li, X., et al. (2025).** QTLScan: An automated toolkit for QTL identification, genetic architecture dissection, and visualization. *[Journal Name]*. DOI: [DOI Link]


## 📧 Contact

For questions, bugs, or feature requests, please open an issue on the GitHub repository or contact:

**Xiaodong Li**:  Email: lxd1997xy@163.com GitHub: [swu1019lab](https://github.com/swu1019lab)
