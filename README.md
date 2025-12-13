
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

* **⚡ Automated QTL Scan:** Efficiently merges GWAS signals into defined QTL blocks based on LD patterns.
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

QTLScan is organized into several subcommands. Below are detailed parameter descriptions and examples for each workflow.

### 1\. Phenotype Processing (`phe`)

Utilities to split multi-trait tables, merge files, or inspect distributions.

#### `phe split`
Split a multi-trait CSV into individual phenotype files for GWAS.

**Parameters:**
*   `--input`: Input TXT/TSV/CSV file (with header) **[Required]**
*   `--sep`: Input separator: auto/csv/tsv/tab/','/'\t' (default: auto)
*   `--sample-col`: Sample ID column name (default: first column)
*   `--traits`: Regex to select trait columns (default: all non-sample columns)
*   `--out-format`: Output format for per-trait files (choices: txt, tsv, csv, plink) (default: tsv)
*   `--out-dir`: Output directory (default: .)
*   `--prefix`: Output file prefix (default: input basename)
*   `--out-sample-col`: Rename sample column in outputs (non-PLINK)
*   `--header`: Whether output contains header (1) or not (0) for non-PLINK (default: 1)
*   `--encoding`: File encoding (default: utf-8)

**Example:**
```bash
qtlscan phe split \
    --input data/phenotypes.csv \
    --out-format plink \
    --out-dir data/phe/split \
    --prefix trait
```

#### `phe stat`
Compute phenotype statistics and plot distribution.

**Parameters:**
*   `--input`: Input phenotype file (TXT/TSV/CSV/PLINK) **[Required]**
*   `--format-in`: Force input format (choices: auto, table, plink) (default: auto)
*   `--sep`: Input separator for table formats (default: auto)
*   `--sample-col`: Sample ID column name for table input (default: first column)
*   `--columns`: Regex to select traits (default: all non-sample columns)
*   `--out-dir`: Output directory (default: .)
*   `--out-name`: Output name prefix (default: phe_stat)
*   `--stats-file`: Custom path for stats table (default: <out_dir>/<out_name>.stats.tsv)
*   `--plot-kind`: Plot kind for distribution figure (choices: box, violin, hist, kde) (default: hist)
*   `--bins`: Histogram bins (default: 30)
*   `--density`: Histogram as density
*   `--fit-curve`: Add fitted curve for histogram
*   `--fit-bins`: Bins for fitted curve (default: 100)
*   `--hist-style`: Histogram layout for multi-columns (choices: stack, dodge, side) (default: dodge)
*   `--kde-style`: KDE layout for multi-columns (choices: stack, side) (default: stack)
*   `--kde-points`: KDE points (default: 200)
*   `--extend-tail`: Extend lower bound (0-1) for KDE (default: 0.0)
*   `--extend-head`: Extend upper bound (0-1) for KDE (default: 0.0)
*   `--orientation`: Plot orientation (choices: vertical, horizontal) (default: vertical)
*   `--colors`: Colors for plots
*   `--alpha`: Transparency for plots (default: 0.7)
*   `--xlabel`: X label
*   `--ylabel`: Y label
*   `--width`: Figure width (default: 8)
*   `--height`: Figure height (default: 6)
*   `--format`: Figure format: png/pdf/svg (default: png)

**Example:**
```bash
qtlscan phe stat \
    --input data/phe/split/trait.Yield.phe \
    --plot-kind hist \
    --fit-curve \
    --out-dir plots
```

### 2\. Run GWAS (`gwas`)

Wrapper for GEMMA or EMMAX to perform association analysis.

**Parameters:**
*   `--vcf`: Path to VCF genotype file **[Required]**
*   `--phe`: Phenotype file (tab-separated, no header, first two columns are sample identifiers) **[Required]**
*   `--method`: GWAS engine to use (choices: gemma, emmax) (default: gemma)
*   `--out_dir`: Output directory (default: .)
*   `--out_name`: Output file name prefix (default: gwas)

**Example:**
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

**Parameters:**
*   `--summary`: Path(s) to GWAS summary statistics txt file(s). multiple files with space separated are allowed. Required columns: chr, rs, p_wald and ps **[Required if --phe not provided]**
*   `--phe`: Path to phenotype file (no header, tab-separated, used for GWAS analysis with no summary statistics) **[Required if --summary not provided]**
*   `--vcf`: Path to VCF genotype file **[Required]**
*   `--names`: Trait names in output order (comma-separated string or path to a newline-delimited file); count must match summaries or phenotypes **[Required]**
*   `--out_name`: Output file name prefix (default: gwas)
*   `--out_dir`: Output directory (default: .)
*   `--min_snps`: Minimum number of SNPs per QTL block (default: 3)
*   `--ld_window`: Maximum number of SNPs in LD window (default: 9999999)
*   `--ld_window_kb`: Maximum number of kb in LD window (default: 100)
*   `--ld_window_r2`: LD threshold (R^2) for LD window (default: 0.2)
*   `--threads`: Number of threads for PLINK (default: cpu_count)
*   `--pvalue`: One or two P-value thresholds. If two are provided, the larger value filters GWAS input and the smaller value evaluates core SNP counts within QTL blocks. (default: [1.0])
*   `--gff`: Path to GFF file for gene annotation
*   `--main_type`: Main feature type to extract from GFF file (default: mRNA)
*   `--attr_id`: Attribute ID to extract gene names from GFF file (default: ID)

**Example:**
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

**Parameters:**
*   `--qtl`: Path(s) to QTL blocks csv file(s). Multiple files with space separated are allowed. **[Required if --edge not provided]**
*   `--edge`: Path to existing edge dataframe file (skips LD calculation). **[Required if --qtl not provided]**
*   `--vcf`: Path to VCF genotype file **[Required if --qtl is used]**
*   `--names`: Rename different QTL blocks file(s) for visualization
*   `--trait_colors`: Path to a trait–color mapping file (trait and color separated by comma/tab/space per line)
*   `--layout`: Layout for QTL network (choices: spring, kamada_kawai, fruchterman_reingold, forceatlas2) (default: spring)
*   `--k`: Spring layout parameter k (default: 0.3)
*   `--seed`: Random seed for deterministic node layouts (default: 42)
*   `--min_weight`: Minimum weight for edges (default: 0)
*   `--node_size_factor`: Factor to multiply -log10(pvalue) for node size (default: 5)
*   `--node_alpha`: Node transparency (default: 0.8)
*   `--node_linewidths`: Node border width (default: 1.0)
*   `--edge_width_factor`: Factor to multiply LD value for edge width (default: 3)
*   `--edge_alpha`: Edge transparency (default: 0.5)
*   `--edge_color`: Edge color (default: gray)
*   `--edge_style`: Edge line style (choices: solid, dashed, dotted, dashdot) (default: solid)
*   `--edge_arrows`: Show edge arrows (default: True)
*   `--no_edge_arrows`: Hide edge arrows
*   `--edge_arrowsize`: Edge arrow size (default: 10)
*   `--edge_connectionstyle`: Edge connection style (default: arc3,rad=0.2)
*   `--with_labels`: Whether to show node labels
*   `--node_labels_size`: Node label font size (default: 6)
*   `--node_labels_color`: Node label color (default: white)
*   `--label_font_weight`: Label font weight (default: normal)
*   `--label_font_family`: Label font family (default: sans-serif)
*   `--width`: Figure width (default: 10)
*   `--height`: Figure height (default: 8)
*   `--format`: Output format, e.g., pdf or png (default: png)
*   `--out_dir`: Output directory (default: .)
*   `--out_name`: Output file name prefix (default: output)

**Example:**
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

**Parameters:**
*   `--qtl`: Path to QTL blocks file **[Required]**
*   `--phe`: Path to phenotype file **[Required]**
*   `--vcf`: Path to VCF file **[Required]**
*   `--genome`: Path to genome file (chr, length) for visualization
*   `--out_dir`: Output directory (default: .)
*   `--out_name`: Output name (default: epistasis)
*   `--pvalue`: P-value threshold (default: 0.05)

**Example:**
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

#### Manhattan & QQ Plot (`manhattan`)

**Parameters:**
*   `--summary`: Path to summary statistics txt file **[Required]**
*   `--chr_unit`: Unit for x-axis (default: mb)
*   `--chr_colors`: Colors for chromosomes
*   `--sig_threshold`: Significance p-value threshold for line plot
*   `--point_size`: Point size for Manhattan plot (default: 5)
*   `--qq`: Whether to plot QQ plot (default: False)
*   `--width`: Figure width (default: 10)
*   `--height`: Figure height (default: 3)
*   `--format`: Output format, e.g., pdf or png (default: png)
*   `--out_dir`: Output directory (default: .)
*   `--out_name`: Output file name prefix (default: output)

**Example:**
```bash
qtlscan plot manhattan \
    --summary results/gwas/Yield_GWAS.assoc.txt \
    --sig_threshold 1e-6 \
    --qq \
    --out_dir plots \
    --out_name Yield_Manhattan
```

#### LD Heatmap (`ldheatmap`)

**Parameters:**
*   `--ld`: Path to LD information txt file **[Required]**
*   `--blocks`: Path to QTL blocks csv file **[Required]**
*   `--chr`: Target chromosome for plotting **[Required]**
*   `--gff`: Path to GFF file for gene annotation **[Required]**
*   `--main_type`: Main feature type to extract from GFF file (default: mRNA)
*   `--attr_id`: Attribute ID to extract gene names from GFF file (default: ID)
*   `--unit`: Unit for x-axis (default: mb)
*   `--ft`: Feature type to plot (default: ['exon', 'five_prime_UTR', 'three_prime_UTR'])
*   `--fc`: Face color for features (default: ['#38638D', 'gray', 'gray'])
*   `--ec`: Edge color for features (default: [None, None, None])
*   `--fh`: Feature height to plot (default: [0.2, 0.2, 0.2])
*   `--fz`: Feature zorder to plot (default: [1, 1, 1])
*   `--th`: Track height to plot (default: 0.5)
*   `--plot_value`: Whether to plot values in LD heatmap (default: False)
*   `--cmap`: Colormap for LD heatmap (default: Reds)
*   `--width`: Figure width (default: 10)
*   `--height`: Figure height (default: 8)
*   `--format`: Output format, e.g., pdf or png (default: png)
*   `--out_dir`: Output directory (default: .)
*   `--out_name`: Output file name prefix (default: output)

**Example:**
```bash
qtlscan plot ldheatmap \
    --ld results/qtl/ld/Yield_QTL.ld \
    --blocks results/qtl/Yield_QTL.blocks.csv \
    --chr chrA08 \
    --gff data/genome_annotation.gff3 \
    --out_dir plots \
    --out_name Yield_A08_Heatmap
```

#### QTN Genotype Map (`qtnmap`)

Visualize allelic distribution across the population for identified loci.

**Parameters:**
*   `--qtl`: Path(s) to QTL blocks csv file(s) **[Required]**
*   `--vcf`: Path to VCF genotype file **[Required]**
*   `--samples_file`: Path to samples file (2 columns: sample, group)
*   `--by`: Group variants by trait or chr (choices: trait, chr) (default: trait)
*   `--sample_metric`: Sample metric (choices: mis, het, count, count0, count1, count2) (default: count)
*   `--variant_metric`: Variant metric (choices: mis, het, maf, count, count0, count1, count2) (default: maf)
*   `--width`: Figure width (default: 12)
*   `--height`: Figure height (default: 10)
*   `--format`: Output format (default: png)
*   `--out_dir`: Output directory (default: .)
*   `--out_name`: Output file name prefix (default: output)

**Example:**
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

> **Xiaodong Li. (2025).** QTLScan: https://github.com/swu1019lab/qtlscan


## 📧 Contact

For questions, bugs, or feature requests, please open an issue on the GitHub repository or contact:

**Xiaodong Li**:  Email: lxd1997xy@163.com GitHub: [swu1019lab](https://github.com/swu1019lab)
