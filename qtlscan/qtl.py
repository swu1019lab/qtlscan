import pandas as pd
import numpy as np
import subprocess
import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from multiprocessing import cpu_count
from typing import Optional, List, Dict, Any
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
import matplotlib.patches as mpatches

from qtlscan.log import logger


class UnionFind:
    """Union-Find data structure for optimizing QTL block finding."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


class QTL:
    def __init__(self):
        """
        Initialize the QTL analysis class.
        """
        self._emmax_cmd: Optional[str] = None
        self._emmax_kin_cmd: Optional[str] = None

    def read_gwas(self, gwas_file: str, chunksize: int = 100000, pvalue_threshold: float = 1.0):
        """
        Read GWAS data and check if the required columns exist.

        :param gwas_file: Path to GWAS results file
        :param chunksize: Chunk size for reading files
        :param pvalue_threshold: P-value threshold for filtering significant SNPs
        """
        logger.info("Loading GWAS data...")

        # Check for required columns
        required_columns = {"chr", "rs", "p_wald", "ps"}  # Required columns for GWAS data
        chunks = []
        for chunk in pd.read_csv(gwas_file, sep="\t", chunksize=chunksize, usecols=required_columns):
            missing_columns = required_columns - set(chunk.columns)
            if missing_columns:
                raise ValueError(
                    f"The GWAS file is missing the following required columns: {missing_columns}. "
                    f"Please ensure the file contains columns: {required_columns}."
                )
            # Filter significant SNPs
            chunk = chunk[chunk["p_wald"] <= pvalue_threshold]
            chunks.append(chunk)

        # Combine chunks into a single DataFrame
        df = pd.concat(chunks)
        logger.info(f"Loaded {len(df)} GWAS SNPs.")
        return df

    def read_snp_file(self, snp_file: str, snps_col: list = None):
        """
        Read SNP file containing a list of SNPs.

        :param snp_file: Path to a file containing SNPs positions and haplotypes information with the following format (4+N columns):
            chrom pos ref alt hap1 hap2 ... hapN
        :param snps_col: List of haplotype column names to select, e.g., hap1 hap2 hap3. default: all haplotypes columns
        """
        logger.info(f"Loading SNP file: {snp_file}")
        snp_df = pd.read_csv(snp_file)
        # check for required columns
        required_columns = {"chrom", "pos", "ref", "alt"}
        missing_columns = required_columns - set(snp_df.columns)
        if missing_columns:
            raise ValueError(
                f"The SNP file is missing the following required columns: {missing_columns}. "
                f"Please ensure the file contains columns: {required_columns}."
            )
        # count haplotype columns (4+N columns)
        num_haps = len(snp_df.columns) - 4
        logger.info(f"Loaded {len(snp_df)} SNPs with {num_haps} haplotypes.")
        if snps_col is not None and len(snps_col) > 0:
            snp_df = snp_df.loc[:, ["chrom", "pos", "ref", "alt"] + snps_col]
            logger.info(f"Selected {len(snps_col)} haplotypes: {snps_col}.")
        return snp_df

    def read_anno_file(self, anno_file: str):
        """
        Read annotation file containing SNP positions and annotation information.

        :param anno_file: Path to annotation file
        """
        logger.info(f"Loading annotation file: {anno_file}")
        anno_df = pd.read_csv(anno_file)
        anno_df.columns = ["chrom", "pos", "anno"]
        anno_df = anno_df.drop_duplicates(subset=["chrom", "pos"])
        anno_df = anno_df.dropna()
        logger.info(f"Loaded {len(anno_df)} annotation with valid values.")
        return anno_df

    def read_gff(self, gff_file, main_type="mRNA", attr_id="ID", extract_gene=True):
        """Load GFF file and extract gene annotation information."""
        logger.info(f"Loading GFF file: {gff_file}")
        gff_df = pd.read_csv(gff_file, sep="\t", comment="#", header=None,
                            names=["chr", "source", "type", "start", "end", "score", "strand", "phase", "attributes"])
        logger.info(f"Loaded {len(gff_df)} features from GFF file.")
        if extract_gene:
            # Filter for specified main feature type
            gff_df = gff_df[gff_df["type"] == main_type]
            # Extract gene names
            gff_df["gene"] = gff_df["attributes"].str.extract(fr'{attr_id}=([^;]+)')
            logger.info(f"Loaded {len(gff_df)} features of type '{main_type}' from GFF file.")
            return gff_df[["chr", "start", "end", "gene"]]
        return gff_df

    def read_block_file(self, block_file: str):
        """
        Read QTL block data from a file.

        :param block_file: Path to QTL block file
        """
        logger.info(f"Loading block data from {block_file}...")
        # Load block file from .csv or .txt file
        if block_file.endswith(".csv"):
            return pd.read_csv(block_file)
        elif block_file.endswith(".txt"):
            return pd.read_csv(block_file, sep="\t")
        else:
            raise ValueError(f"Unsupported file format: {block_file}")

    def read_genome_file(self, genome_file: str):
        """
        Read genome file containing chromosome name and length information.

        :param genome_file: Path to genome file with 2 columns: chromosome name and length
        """
        logger.info(f"Loading genome file: {genome_file}")
        return pd.read_csv(genome_file, sep="\t")

    def read_ld_file(self, ld_file: str):
        """
        Read LD data from a file.

        :param ld_file: Path to LD file
        """
        logger.info(f"Loading LD data from {ld_file}...")
        return pd.read_csv(ld_file, sep='\s+', 
                           usecols=["CHR_A", "BP_A", "SNP_A", "CHR_B", "BP_B", "SNP_B", "R2"],
                           dtype={"CHR_A": "category", "SNP_A": "str", "BP_A": "int",
                                  "CHR_B": "category", "SNP_B": "str", "BP_B": "int", 
                                  "R2": "float"})

    def find_genes_in_region(self, gff_df, chrom, bp1, bp2):
        """Find genes in a specified region."""
        genes = gff_df[(gff_df["chr"] == chrom) & (gff_df["start"] <= bp2) & (gff_df["end"] >= bp1)]
        return ";".join(genes["gene"].dropna().unique())

    def check_plink(self):
        """Check if PLINK is installed and in the system PATH."""
        try:
            subprocess.run(["plink", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
            raise RuntimeError("PLINK is not installed or not in the system PATH. Please install PLINK first.")

    def check_gemma(self):
        """Check if GEMMA is installed and in the system PATH."""
        try:
            subprocess.run(["gemma", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
            raise RuntimeError("GEMMA is not installed or not in the system PATH. Please install GEMMA first.")

    def check_emmax(self):
        """Check if EMMAX binaries are installed and remember the available commands."""

        emmax_candidates = ["emmax-intel64", "emmax"]
        kin_candidates = ["emmax-kin-intel64", "emmax-kin"]

        self._emmax_cmd = None
        self._emmax_kin_cmd = None

        for candidate in emmax_candidates:
            try:
                subprocess.run([candidate], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError:
                # Command exists but returned non-zero; still accept the binary.
                self._emmax_cmd = candidate
                break
            else:
                self._emmax_cmd = candidate
                break

        if self._emmax_cmd is None:
            raise RuntimeError(
                "EMMAX executable not found. Please install emmax-intel64 or emmax and ensure it is in PATH."
            )

        for candidate in kin_candidates:
            try:
                subprocess.run([candidate], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError:
                self._emmax_kin_cmd = candidate
                break
            else:
                self._emmax_kin_cmd = candidate
                break

        if self._emmax_kin_cmd is None:
            raise RuntimeError(
                "emmax-kin executable not found. Please install emmax-kin-intel64 or emmax-kin and ensure it is in PATH."
            )

    def calculate_ld(
        self,
        snps: list,
        vcf_file: str,
        ld_window_r2: float = 0.2,
        ld_window_kb: int = 100,
        ld_window: int = 10,
        threads: int = 1,
        out_dir: str = ".",
        out_name: str = "qtlscan",
    ):
        """
        Calculate LD.

        :param snps: List of SNPs to calculate LD
        :param vcf_file: Path to VCF file
        :param ld_window_r2: R² threshold for LD
        :param ld_window_kb: Maximum distance between SNPs (kb)
        :param ld_window: Window size for LD calculation
        :param threads: Number of threads for parallel processing
        :param out_dir: Output directory
        :param out_name: Output file name prefix
        """
        if shutil.which("plink") is None:
            raise RuntimeError("PLINK executable not found in PATH. Please install PLINK (v1.9) and ensure it is available in your system path.")

        if len(snps) < 2:
            raise ValueError("At least two SNPs are required for LD calculation.")

        os.makedirs(os.path.join(out_dir, "ld"), exist_ok=True)
        snp_file = os.path.join(out_dir, f"ld/{out_name}.snps.txt")
        pd.Series(snps).to_csv(snp_file, index=False, header=False)

        # Calculate LD
        ld_path = os.path.join(out_dir, f"ld/{out_name}")
        plink_command = [
            "plink",
            "--vcf", vcf_file,
            "--extract", snp_file,
            "--r2",
            "--allow-extra-chr",
            "--allow-no-sex",
            "--ld-window-kb", str(ld_window_kb),
            "--ld-window-r2", str(ld_window_r2),
            "--ld-window", str(ld_window),
            "--memory", "2048",
            "--threads", str(threads),
            "--out", ld_path
        ]
        logger.info(f"Running PLINK command: {' '.join(plink_command)}")
        try:
            subprocess.run(plink_command, 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"PLINK command failed: {e.stderr.decode()}")
            raise

        if not os.path.exists(ld_path + ".ld"):
            raise FileNotFoundError(f"PLINK output file missing: {ld_path}.ld")
        return pd.read_csv(ld_path + ".ld", sep='\s+')

    def calculate_maf(self, vcf_file: str, threads: int = 1, out_dir: str = "."):
        """
        Calculate minor allele frequency (MAF)
        """
        os.makedirs(os.path.join(out_dir, "frq"), exist_ok=True)

        # Calculate MAF
        frq_path = os.path.join(out_dir, f"frq/plink.{os.getpid()}.{uuid.uuid4().hex}")
        plink_command = [
            "plink",
            "--vcf", vcf_file,
            "--freq",
            "--allow-extra-chr",
            "--allow-no-sex",
            "--memory", "2048",
            "--threads", str(threads),
            "--out", frq_path
        ]
        logger.info(f"Running PLINK command: {' '.join(plink_command)}")
        try:
            subprocess.run(plink_command, 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"PLINK command failed: {e.stderr.decode()}")
            raise

        if not os.path.exists(frq_path + ".frq"):
            raise FileNotFoundError(f"PLINK output file missing: {frq_path}.frq")
        # CHR SNP A1 A2 MAF NCHROBS
        return pd.read_csv(frq_path + ".frq", sep='\s+')

    def run_gemma(self, vcf_file: str, phe_file: str, out_dir: str, out_name: str) -> list:
        """
        Run GEMMA to perform GWAS analysis for multiple phenotypes.

        :param vcf_file: Path to VCF genotype file
        :param phe_file: Path to phenotype file (first two columns are sample IDs, followed by phenotypes)
        :param out_dir: Output directory
        :param out_name: Output file name prefix
        :return: Path to the generated GWAS summary file
        """
        # Read phenotype file to determine the number of phenotypes
        phe_df = pd.read_csv(phe_file, sep="\t", header=None)
        num_phenotypes = phe_df.shape[1] - 2  # Subtract the first two columns (sample IDs)
        if num_phenotypes < 1:
            raise ValueError("The phenotype file must contain at least one phenotype column.")

        logger.info(f"Found {num_phenotypes} phenotypes in the phenotype file.")

        # Check if PLINK and GEMMA are installed
        self.check_plink()
        self.check_gemma()

        plink_prefix = os.path.join(out_dir, out_name)

        # Loop through each phenotype and run GEMMA
        summary_files = []
        for i in range(1, num_phenotypes + 1):
            logger.info(f"Processing phenotype {i}...")

            # Run PLINK to extract the current phenotype
            plink_pheno_command = [
                "plink",
                "--vcf", vcf_file,
                "--pheno", phe_file,
                "--mpheno", str(i),
                "--missing-phenotype", "-9",
                "--out", f"{plink_prefix}_{i}",
                "--vcf-half-call", "m",
                "--allow-extra-chr",
                "--allow-no-sex",
                "--make-bed"
            ]
            logger.info(f"Running PLINK command: {' '.join(plink_pheno_command)}")
            try:
                subprocess.run(plink_pheno_command, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"PLINK command failed: {e.stderr.decode()}")
                raise

            # Run GEMMA to generate the kinship matrix
            gemma_kinship_command = [
                "gemma",
                "-bfile", f"{plink_prefix}_{i}",
                "-gk", "2",  # Generate centered kinship matrix
                "-o", f"{out_name}_{i}_kinship",
                "-outdir", out_dir
            ]
            logger.info(f"Running GEMMA kinship command: {' '.join(gemma_kinship_command)}")
            try:
                subprocess.run(gemma_kinship_command, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"GEMMA command failed: {e.stderr.decode()}")
                raise

            # Run GEMMA to perform LMM analysis
            gemma_lmm_command = [
                "gemma",
                "-bfile", f"{plink_prefix}_{i}",
                "-k", os.path.join(out_dir, f"{out_name}_{i}_kinship.sXX.txt"),  # Kinship matrix
                "-lmm", "4",  # Linear mixed model
                "-o", f"{out_name}_{i}_lmm",
                "-outdir", out_dir
            ]
            logger.info(f"Running GEMMA LMM command: {' '.join(gemma_lmm_command)}")
            try:
                subprocess.run(gemma_lmm_command, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"GEMMA command failed: {e.stderr.decode()}")
                raise

            # Save the path to the summary file
            summary_file = os.path.join(out_dir, f"{out_name}_{i}_lmm.assoc.txt")
            if not os.path.exists(summary_file):
                raise FileNotFoundError(f"GEMMA output file not found: {summary_file}")
            summary_files.append(summary_file)

        logger.info(f"GEMMA analysis completed for {num_phenotypes} phenotypes.")
        return summary_files

    def run_emmax(self, vcf_file: str, phe_file: str, out_dir: str, out_name: str) -> list:
        """Run EMMAX (Intel build compatible) for a single phenotype.

        The workflow follows Intel's recommended steps:
            1. Transpose the input VCF directly to TPED/TFAM format required by EMMAX.
            2. Build the kinship matrix once via :code:`emmax-kin-intel64` (or legacy :code:`emmax-kin`).
            3. Align the single phenotype to TFAM ordering, run :code:`emmax`, and enrich the
               association output with chromosome and position metadata from the TPED file.

        :param vcf_file: Path to VCF genotype file.
        :param phe_file: Path to phenotype file (exactly three columns: FID, IID, phenotype).
        :param out_dir: Output directory.
        :param out_name: Output file name prefix.
    :return: List containing the path to the enriched EMMAX output (.ps1).
        """

        phe_df = pd.read_csv(
            phe_file,
            sep="\s+",
            header=None,
            na_values=["NA", "na", "Na", "-9"],
        )

        if phe_df.shape[1] != 3:
            raise ValueError("Phenotype file must contain exactly three columns: FID, IID, phenotype.")

        logger.info("Detected single phenotype column as required by EMMAX.")

        phe_df = phe_df.copy()
        phe_df.iloc[:, 0] = phe_df.iloc[:, 0].astype(str).str.strip()
        phe_df.iloc[:, 1] = phe_df.iloc[:, 1].astype(str).str.strip()
        phe_df.columns = ["FID", "IID", "PHENO"]
        phe_df["PHENO"] = pd.to_numeric(phe_df["PHENO"], errors="coerce")

        self.check_plink()
        self.check_emmax()

        os.makedirs(out_dir, exist_ok=True)
        output_prefix = os.path.join(out_dir, out_name)

        # Step 1: Recode VCF directly to TPED/TFAM for EMMAX.
        tped_path = f"{output_prefix}.tped"
        tfam_path = f"{output_prefix}.tfam"
        if not (os.path.exists(tped_path) and os.path.exists(tfam_path)):
            plink_tped_command = [
                "plink",
                "--vcf", vcf_file,
                "--vcf-half-call", "m",
                "--allow-extra-chr",
                "--allow-no-sex",
                "--recode12",
                "--output-missing-genotype", "0",
                "--transpose",
                "--out", output_prefix,
            ]
            logger.info(f"Running PLINK (VCF→TPED transpose) command: {' '.join(plink_tped_command)}")
            try:
                subprocess.run(
                    plink_tped_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"PLINK recode command failed: {e.stderr.decode()}")
                raise

        if not (os.path.exists(tped_path) and os.path.exists(tfam_path)):
            raise FileNotFoundError("Failed to generate TPED/TFAM files required by EMMAX.")

        tfam_df = pd.read_csv(tfam_path, sep="\s+", header=None, dtype=str)
        tfam_df = tfam_df.iloc[:, :6]
        tfam_df.iloc[:, 0] = tfam_df.iloc[:, 0].astype(str).str.strip()
        tfam_df.iloc[:, 1] = tfam_df.iloc[:, 1].astype(str).str.strip()

        try:
            tped_metadata = pd.read_csv(
                tped_path,
                sep="\s+",
                header=None,
                usecols=[0, 1, 3],
                names=["chr", "rs", "ps"],
                dtype={"chr": str, "rs": str, "ps": object},
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load TPED metadata for EMMAX: {exc}") from exc

        tped_metadata["ps"] = pd.to_numeric(tped_metadata["ps"], errors="coerce")
        tped_metadata = tped_metadata.dropna(subset=["chr", "rs", "ps"])
        snp_metadata = (
            tped_metadata.drop_duplicates(subset="rs")
            .set_index("rs")
            .loc[:, ["chr", "ps"]]
        )

        phenotype_series = phe_df.set_index(["FID", "IID"])["PHENO"]

        # Step 2: Build kinship matrix once.
        if self._emmax_kin_cmd is None:
            raise RuntimeError("EMMAX kinship command not initialized. Did you call check_emmax()?")

        kinship_file = f"{output_prefix}.aBN.kinf"

        if not os.path.exists(kinship_file):
            emmax_kin_command = [self._emmax_kin_cmd, "-v", "-d", "10", output_prefix]

            logger.info(f"Running {self._emmax_kin_cmd} command: {' '.join(emmax_kin_command)}")
            try:
                subprocess.run(
                    emmax_kin_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"{self._emmax_kin_cmd} command failed: {e.stderr.decode()}")
                raise

        if not os.path.exists(kinship_file):
            raise FileNotFoundError(
                "EMMAX kinship output file not found after execution. Expected .aBN.kinf or .aIBS.kinf file."
            )

        # Step 3: Generate phenotype file and run EMMAX.
        logger.info("Processing phenotype with EMMAX...")

        phenotype_out = pd.DataFrame({
            "FID": tfam_df.iloc[:, 0],
            "IID": tfam_df.iloc[:, 1],
        })

        phenotype_map = phenotype_series.to_dict()
        ordered_values = [phenotype_map.get((fid, iid)) for fid, iid in zip(phenotype_out["FID"], phenotype_out["IID"])]
        ordered_numeric = pd.to_numeric(ordered_values, errors="coerce")

        matched = int(pd.notna(ordered_numeric).sum())
        missing = len(ordered_numeric) - matched
        if missing:
            logger.warning(
                "%d TFAM samples lacked matching phenotype entries; they will be set to NA.",
                missing,
            )

        trait_values = ordered_numeric.astype(object)
        trait_values[pd.isna(trait_values)] = "NA"
        phenotype_out["PHENO"] = trait_values

        phenotype_path = os.path.join(out_dir, f"{out_name}.phe")
        phenotype_out.to_csv(phenotype_path, sep="\t", header=False, index=False)

        emmax_command = [
            self._emmax_cmd,
            "-v",
            "-d", "10",
            "-t", output_prefix,
            "-p", phenotype_path,
            "-k", kinship_file,
            "-o", f"{output_prefix}_emmax",
        ]

        logger.info(f"Running {self._emmax_cmd} command: {' '.join(emmax_command)}")
        try:
            subprocess.run(
                emmax_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"{self._emmax_cmd} command failed: {e.stderr.decode()}")
            raise

        summary_file = f"{output_prefix}_emmax.ps"
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"EMMAX output file not found: {summary_file}")

        try:
            emmax_df = pd.read_csv(summary_file, sep="\s+", header=None)
        except Exception as exc:
            raise RuntimeError(f"Unable to read EMMAX summary file {summary_file}: {exc}") from exc

        if emmax_df.empty:
            raise RuntimeError("EMMAX output file is empty; expected association results.")

        expected_order = ["rs", "beta", "se_beta", "p_wald"]
        rename_pairs = {idx: col for idx, col in enumerate(expected_order) if idx in emmax_df.columns}
        if rename_pairs:
            emmax_df = emmax_df.rename(columns=rename_pairs)
        emmax_df = emmax_df[[col for col in expected_order if col in emmax_df.columns]]

        required_columns = {"rs", "p_wald"}
        if not required_columns.issubset(emmax_df.columns):
            missing = required_columns - set(emmax_df.columns)
            raise RuntimeError(
                f"EMMAX output is missing required columns after renaming: {missing}"
            )

        emmax_df["rs"] = emmax_df["rs"].astype(str)
        emmax_df["p_wald"] = pd.to_numeric(emmax_df["p_wald"], errors="coerce")
        emmax_df = emmax_df.dropna(subset=["p_wald"]).copy()

        enriched_df = emmax_df.merge(snp_metadata, left_on="rs", right_index=True, how="left")

        missing_coord = int(enriched_df["chr"].isna().sum())
        if missing_coord:
            logger.warning(
                "%d SNPs in EMMAX output lacked TPED metadata and will be dropped from the enriched file.",
                missing_coord,
            )
        enriched_df = enriched_df.dropna(subset=["chr", "ps"])
        if enriched_df.empty:
            raise RuntimeError("No overlapping SNPs found between EMMAX output and TPED metadata.")

        enriched_df["chr"] = enriched_df["chr"].astype(str)
        enriched_df["ps"] = pd.to_numeric(enriched_df["ps"], errors="coerce")
        enriched_df = enriched_df.dropna(subset=["ps"]).copy()
        enriched_df["ps"] = enriched_df["ps"].astype(int)

        result_df = enriched_df.loc[:, ["chr", "rs", "ps", "p_wald"]]

        enriched_file = f"{output_prefix}_emmax.ps1"
        result_df.to_csv(enriched_file, sep="\t", index=False)

        logger.info("EMMAX analysis completed for the phenotype. Enriched output saved to %s", enriched_file)
        return [enriched_file]

    def process_gwas(self, 
            vcf_file: str,
            gwas_data: pd.DataFrame = None,
            ann_data: pd.DataFrame = None,
            ld_window: int = 10, 
            ld_window_r2: float = 0.2, 
            ld_window_kb: int = 100, 
            min_snps: int = 3, 
            threads: int = 1, 
            out_dir: str = ".", 
            block_pvalue: Optional[float] = None,
        ) -> pd.DataFrame:
        """Process GWAS data and find QTL blocks.

        :param block_pvalue: Optional stricter P-value threshold used to
            ensure each block contains more than three SNPs that meet the
            criterion. When None, all SNPs retained after GWAS filtering are
            treated equally.
        """
        logger.info("Processing GWAS QTL data...")

        qtl_blocks = pd.DataFrame()
        if not gwas_data.empty:
            # Calculate LD
            all_snps = gwas_data["rs"].drop_duplicates().tolist()
            logger.info(f"Calculating LD for {len(all_snps)} SNPs...")
            ld_df = self.calculate_ld(all_snps, vcf_file, ld_window_r2, ld_window_kb, ld_window, threads, out_dir)
            logger.info(f"LD calculation completed.")

            # Find QTL blocks
            qtl_blocks = self.find_qtl_blocks(
                ld_df,
                gwas_data,
                min_snps,
                ann_data,
                block_pvalue=block_pvalue,
            )
        return qtl_blocks

    def find_qtl_blocks(
        self,
        ld_df: pd.DataFrame,
        qtl_df: pd.DataFrame,
        min_snps: int = 3,
        gff_df=None,
        block_pvalue: Optional[float] = None,
    ) -> pd.DataFrame:
        """Find QTL blocks based on LD results (optimized with Union-Find).

        :param block_pvalue: Optional stricter P-value threshold.
            When provided, blocks must contain more than min_snps SNPs with
            p-values less than or equal to this threshold to be retained.
        """
        logger.info("Finding QTL blocks based on LD results...")
        uf = UnionFind()
        for _, row in ld_df.iterrows():
            uf.union(row["SNP_A"], row["SNP_B"])

        # Count SNPs in each set
        blocks = defaultdict(list)
        for snp in uf.parent:
            blocks[uf.find(snp)].append(snp)

        qtl_blocks = []

        for block in blocks.values():
            if len(block) < min_snps:
                continue
            block_df = qtl_df[qtl_df["rs"].isin(block)]
            if block_df.empty:
                continue
            if block_pvalue is not None:
                sig_count = int((block_df["p_wald"] <= block_pvalue).sum())
                if sig_count < min_snps:
                    logger.warning(
                        f"Discarding block with SNPs %s due to insufficient SNPs (< {min_snps}) below threshold %.3g.",
                        ",".join(map(str, block[:5])),
                        block_pvalue,
                    )
                    print(block_df)
                    continue
            lead_snp = block_df.loc[block_df["p_wald"].idxmin()]
            # Extract SNP position information from block_df
            bp1 = min(block_df["ps"])
            bp2 = max(block_df["ps"])

            # Find genes in the region
            genes = ""
            if gff_df is not None:
                genes = self.find_genes_in_region(gff_df, lead_snp["chr"], bp1, bp2)

            qtl_blocks.append({
                "chr": lead_snp["chr"],  # Chromosome information
                "lead": lead_snp["rs"],  # Lead SNP
                "ps": lead_snp["ps"],  # Lead SNP position
                "pvalue": lead_snp["p_wald"],  # Lead SNP p-value
                "bp1": bp1,  # QTL block start position
                "bp2": bp2,  # QTL block end position
                "kb": (bp2 - bp1) / 1000,  # QTL block size (kb)
                "genes": genes,  # Genes in QTL block
                "num": len(block_df),  # Number of SNPs in QTL block
                "ids": ";".join(block_df["rs"].tolist()),  # SNP IDs in QTL block
                "bps": ";".join(block_df["ps"].astype(str).tolist()),  # Significant SNP positions
                "pvalues": ";".join(block_df["p_wald"].astype(str).tolist())  # Significant SNP p-values
            })

        logger.info(f"Found {len(qtl_blocks)} QTL blocks with at least {min_snps} SNPs.")
        return pd.DataFrame(qtl_blocks)

    def save(self, qtl_blocks: pd.DataFrame, out_dir: str = ".", out_name: str = "qtlscan"):
        """
        Save QTL blocks to a file.

        :param qtl_blocks: QTL block DataFrame
        :param out_dir: Output directory
        :param out_name: Output file name prefix
        """
        output_path = os.path.join(out_dir, out_name + ".blocks.csv")
        qtl_blocks.to_csv(output_path, index=False, float_format="%.6g")
        logger.info(f"Saved {len(qtl_blocks)} QTL blocks to {output_path}.")

        # Check if the 'genes' column exists
        if "genes" in qtl_blocks.columns:
            logger.info("Extracting genes from QTL blocks...")

            # Extract genes and split by comma
            genes_list = []
            for genes in qtl_blocks["genes"]:
                if pd.notna(genes) and genes.strip():  # Check if the value is not NaN or empty
                    genes_list.extend(genes.split(";"))

            # Remove duplicates and sort
            genes_list = sorted(set(genes_list))

            # Save genes to a file
            genes_file = os.path.join(out_dir, f"{out_name}.genes.list")
            with open(genes_file, "w") as f:
                f.write("\n".join(genes_list))
                f.write("\n")
            logger.info(f"Saved {len(genes_list)} unique genes to {genes_file}.")
        else:
            logger.info("No 'genes' column found in QTL blocks, skipping genes extraction.")


class QTLNetwork(QTL):
    def __init__(self):
        """
        Initialize the QTLNetwork analysis class.
        """
        super().__init__()
        self.blocks = None
        self.edge_df = None

    def load_data(self, blocks_files: list, blocks_names: list = None):
        """Load necessary data for QTL network analysis."""
        if not blocks_files:
            raise ValueError("blocks_files must contain at least one path to a blocks table.")

        if blocks_names and len(blocks_names) != len(blocks_files):
            raise ValueError("Length of blocks_names must match blocks_files.")

        blocks = []
        required_cols = {"lead", "pvalue", "trait", "ids"}

        for i, qtl_file in enumerate(blocks_files):
            base_name = None
            if blocks_names:
                base_name = blocks_names[i]
            else:
                candidate = os.path.splitext(os.path.basename(qtl_file))[0]
                base_name = f"{candidate.split('.')[0]}_{i}"

            qtl_df = self.read_block_file(qtl_file)
            missing = required_cols - set(qtl_df.columns)
            if missing:
                raise ValueError(
                    f"Block file '{qtl_file}' is missing required columns: {sorted(missing)}"
                )

            qtl_df = qtl_df.copy()
            if blocks_names:
                qtl_df["trait"] = str(base_name)
            else:
                qtl_df["trait"] = qtl_df["trait"].astype(str)
            blocks.append(qtl_df)

        if not blocks:
            raise ValueError("No QTL blocks were loaded from the provided files.")

        concatenated = pd.concat(blocks, ignore_index=True)
        if concatenated.empty:
            raise ValueError("Combined QTL block table is empty.")

        concatenated["qtl"] = [f"q{i}" for i in range(len(concatenated))]
        self.blocks = concatenated
        self.edge_df = None
        logger.info(f"Loaded {len(self.blocks)} QTL blocks from {len(blocks_files)} file(s).")

    def load_edge_data(self, edge_file: str):
        """Load existing edge data for visualization."""
        if not os.path.exists(edge_file):
            raise FileNotFoundError(f"Edge file not found: {edge_file}")
        
        self.edge_df = pd.read_csv(edge_file)
        required_cols = {"source", "target", "weight", "trait1", "trait2", "pvalue1", "pvalue2"}
        if not required_cols.issubset(self.edge_df.columns):
            raise ValueError(f"Edge file is missing required columns: {required_cols - set(self.edge_df.columns)}")
        
        logger.info(f"Loaded {len(self.edge_df)} edges from {edge_file}.")

    def process_data(
        self,
        vcf_path: str,
        min_weight: float = None,
        ld_window_r2: float = 0.0,
        ld_window_kb: int = 1000,
        ld_window: int = 999999,
        threads: Optional[int] = None,
        out_dir: str = ".",
        out_name: str = "qtl_network"
    ):
        """Process data to create QTL network based on LD values.
        
        :param vcf_path: Path to VCF file for LD calculation
        :param min_weight: Minimum weight (LD value) for edges
        :param ld_window_r2: R² threshold passed to PLINK (default mirrors previous behaviour)
        :param ld_window_kb: Maximum window size in kilobases passed to PLINK
        :param ld_window: Maximum number of SNPs in LD window passed to PLINK
        :param threads: Number of threads for PLINK (default: use all available cores)
        :param out_dir: Output directory for temporary files and LD results
        :param out_name: Output file name prefix
        """
        if shutil.which("plink") is None:
            raise RuntimeError("PLINK executable not found in PATH. Please install PLINK (v1.9) and ensure it is available in your system path.")

        if self.blocks is None or self.blocks.empty:
            raise ValueError("No QTL blocks data loaded. Please run load_data() first.")
        logger.info("Processing QTL network data...")
        
        threads = threads or cpu_count()
        os.makedirs(out_dir, exist_ok=True)
        
        ld_file_path = os.path.join(out_dir, "ld", f"{out_name}.ld")
        if os.path.exists(ld_file_path):
             logger.info(f"Loading existing LD results from {ld_file_path}...")
             ld_df = pd.read_csv(ld_file_path, sep=r"\s+")
             # Ensure columns match what calculate_ld returns (SNP_A, SNP_B, R2)
             required_ld_cols = {"SNP_A", "SNP_B", "R2"}
             if not required_ld_cols.issubset(ld_df.columns):
                 raise ValueError(f"Existing LD file {ld_file_path} is missing required columns: {required_ld_cols}")
             
             # Reconstruct block_snps from ids column without VCF validation
             logger.info("Mapping SNPs from blocks...")
             block_snps = defaultdict(list)
             for _, row in self.blocks.iterrows():
                if pd.isna(row['ids']): continue
                snps = str(row['ids']).replace(';', ',').split(',')
                for s in snps:
                    s = s.strip()
                    if s:
                        block_snps[row['qtl']].append(s)
        else:
            # 1. Extract all SNPs within QTL blocks
            logger.info("Extracting variants within QTL intervals...")
            
            # Collect all SNPs from 'ids' column
            all_snps_set = set()
            for _, row in self.blocks.iterrows():
                if pd.isna(row['ids']): continue
                # Handle comma or semicolon separator
                snps = str(row['ids']).replace(';', ',').split(',')
                all_snps_set.update(s.strip() for s in snps if s.strip())
                
            if not all_snps_set:
                 raise ValueError("No SNPs found in 'ids' column of QTL blocks.")
    
            # Create SNP list file for PLINK
            snp_list_file = os.path.join(out_dir, "qtl_snps.list")
            with open(snp_list_file, 'w') as f:
                f.write('\n'.join(all_snps_set))
            
            # Run PLINK to get .bim file with variants
            bim_prefix = os.path.join(out_dir, "qtl_snps")
            cmd = [
                "plink",
                "--vcf", vcf_path,
                "--extract", snp_list_file,
                "--make-just-bim",
                "--out", bim_prefix,
                "--allow-extra-chr",
                "--allow-no-sex"
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                if os.path.exists(snp_list_file): os.remove(snp_list_file)
                raise RuntimeError(f"PLINK failed to extract variants: {e.stderr.decode()}")
                
            # Read .bim file
            try:
                bim_df = pd.read_csv(f"{bim_prefix}.bim", sep='\s+', header=None, 
                                    names=["chr", "rs", "cm", "bp", "a1", "a2"])
            except FileNotFoundError:
                 if os.path.exists(snp_list_file): os.remove(snp_list_file)
                 raise RuntimeError("PLINK failed to generate .bim file. Check if VCF contains variants in the specified ranges.")
    
            # Map SNPs to blocks
            valid_snps = set(bim_df['rs'])
            block_snps = defaultdict(list)
            
            for _, row in self.blocks.iterrows():
                if pd.isna(row['ids']): continue
                snps = str(row['ids']).replace(';', ',').split(',')
                for s in snps:
                    s = s.strip()
                    if s in valid_snps:
                        block_snps[row['qtl']].append(s)
                    
            # Clean up temp files
            if os.path.exists(snp_list_file): os.remove(snp_list_file)
            for ext in ['.bim', '.log', '.nosex']:
                if os.path.exists(bim_prefix + ext): os.remove(bim_prefix + ext)
    
            all_snps = bim_df['rs'].tolist()
            if not all_snps:
                raise ValueError("No variants found within QTL intervals.")
            logger.info(f"Found {len(all_snps)} variants in {len(self.blocks)} QTL blocks.")
    
            # 2. Calculate LD matrix for all these SNPs
            logger.info("Calculating pairwise LD for all variants...")
            
            ld_df = self.calculate_ld(
                all_snps,
                vcf_path,
                ld_window_r2=ld_window_r2,
                ld_window_kb=ld_window_kb,
                ld_window=ld_window,
                threads=threads,
                out_dir=out_dir,
                out_name=out_name
            )
        
        # 3. Process LD values
        ld_lookup = {}
        for _, row in ld_df.iterrows():
            k = tuple(sorted((str(row['SNP_A']), str(row['SNP_B']))))
            ld_lookup[k] = row['R2']
            
        def get_ld(u, v):
            if u == v: return 1.0
            return ld_lookup.get(tuple(sorted((u, v))), 0.0)

        # 4. Calculate PmaxLD for each block
        logger.info("Calculating PmaxLD for each block...")
        pmax_ld = {}
        for qtl, snps in block_snps.items():
            if not snps:
                pmax_ld[qtl] = 1.0
                continue
            if len(snps) == 1:
                pmax_ld[qtl] = 1.0
                continue
            
            max_avg = 0
            for u in snps:
                sum_ld = 0
                count = 0
                for v in snps:
                    if u == v: continue
                    sum_ld += get_ld(u, v)
                    count += 1
                avg = sum_ld / count if count > 0 else 0
                if avg > max_avg:
                    max_avg = avg
            pmax_ld[qtl] = max_avg

        # 5. Calculate Edge Weights
        logger.info("Calculating edge weights...")
        qtl_pairs = []
        
        for row1, row2 in combinations(self.blocks.itertuples(index=False), 2):
            q1, q2 = row1.qtl, row2.qtl
            snps1 = block_snps[q1]
            snps2 = block_snps[q2]
            
            if not snps1 or not snps2:
                continue
            
            # Only calculate for same chromosome (PLINK limitation)
            if str(row1.chr) != str(row2.chr):
                continue
                
            sum_ld = 0
            count = 0
            for u in snps1:
                for v in snps2:
                    sum_ld += get_ld(u, v)
                    count += 1
            
            if count == 0: continue
            avg_ld_cross = sum_ld / count
            
            pmax1 = pmax_ld.get(q1, 1.0)
            pmax2 = pmax_ld.get(q2, 1.0)
            
            term1 = avg_ld_cross / pmax1 if pmax1 > 1e-9 else 0
            term2 = avg_ld_cross / pmax2 if pmax2 > 1e-9 else 0
            
            weight = 0.5 * (term1 + term2)
            
            # Ensure weight does not exceed 1.0
            if weight > 1.0:
                weight = 1.0
            
            if min_weight is not None and weight < min_weight:
                continue
                
            qtl_pairs.append({
                "source": q1,
                "target": q2,
                "weight": weight,
                "trait1": row1.trait,
                "lead1": row1.lead,
                "pvalue1": row1.pvalue,
                "trait2": row2.trait,
                "lead2": row2.lead,
                "pvalue2": row2.pvalue,
            })
            
        self.edge_df = pd.DataFrame(qtl_pairs)
        if self.edge_df.empty:
             logger.warning("No edges found with current parameters.")
        else:
             self.edge_df = self.edge_df.reset_index(drop=True)
             logger.info(f"Constructed QTL network with {len(self.edge_df)} edges.")

    def save(self, out_dir = ".", out_name = "qtlnetwork"):
        if self.blocks is None or self.blocks.empty:
            raise ValueError("No QTL blocks available to save. Run load_data() first.")
        if self.edge_df is None or self.edge_df.empty:
            raise ValueError("No edge data available to save. Run process_data() first.")

        blocks_path = os.path.join(out_dir, f"{out_name}.blocks.csv")
        edges_path = os.path.join(out_dir, f"{out_name}.edges.csv")
        self.blocks.to_csv(blocks_path, index=False)
        self.edge_df.to_csv(edges_path, index=False)
        logger.info(f"Saved QTL network blocks to {blocks_path} and edges to {edges_path}.")


    def degree_plot(self, G, color_map=None, plot_type='rank', ax=None):
        """Plot node-level metrics for the QTL network.

        Supported plot_type values:
        - 'rank': Degree vs rank scatter.
        - 'histogram': Degree frequency (overall or per trait if color_map provided).
        - 'ntraits': Degree vs number of unique neighbor traits (ntraits).
        - 'pvalue': Degree vs -log10(pvalue).

        :param G: NetworkX graph object
        :param color_map: custom color mapping dict {trait: color}
        :param plot_type: one of 'rank', 'histogram', 'ntraits', 'pvalue'
        :param ax: Matplotlib Axes instance to draw on
        :return: DataFrame of node metrics (node, degree, trait, ntraits, pvalue, neglog10, ...)
        """
        if ax is None:
            raise ValueError("An axes instance must be provided for plotting.")

        # calculate degree for each node
        degree_dict = dict(G.degree())
        
        # create a list containing nodes, degrees, and traits
        node_data = []
        for node, degree in degree_dict.items():
            trait = G.nodes[node]['trait']
            # neighbors list and related trait names
            neighbors_list = list(G.neighbors(node))
            related_traits = [G.nodes[neighbor]['trait'] for neighbor in neighbors_list]
            # number of unique related traits
            ntraits = len(set([t for t in related_traits if t is not None and str(t) != '']))
            # node pvalue and -log10(pvalue)
            pvalue = G.nodes[node].get('pvalue', np.nan)
            try:
                pvalue_num = float(pvalue)
            except Exception:
                pvalue_num = np.nan
            neglog10 = -np.log10(np.clip(pvalue_num, 1e-300, None)) if not pd.isna(pvalue_num) else np.nan
            node_data.append({
                'node': node,
                'degree': degree,
                'trait': trait,
                'color': color_map.get(trait, 'blue') if color_map else 'blue',
                'neighbors': ";".join(neighbors_list),
                'related': ";".join(related_traits),
                'ntraits': ntraits,
                'pvalue': pvalue_num,
                'neglog10': neglog10,
            })
        node_data = pd.DataFrame(node_data)

        # sort by degree value
        node_data.sort_values(by='degree', ascending=False, inplace=True, ignore_index=True)

        if plot_type == 'rank':
            ax.scatter(node_data.index, node_data['degree'], color=node_data['color'], s=10)
            ax.plot(node_data['degree'], "gray", alpha=0.5, linestyle='-')
            ax.set_title("Degree Rank Plot")
            ax.set_ylabel("Degree")
            ax.set_xlabel("Rank")
        elif plot_type == 'histogram':
            if color_map:
                traits = list(color_map.keys())
                for trait in traits:
                    trait_degrees = node_data[node_data['trait'] == trait]['degree']
                    if not trait_degrees.empty:
                        hist, bins = np.histogram(trait_degrees, bins=range(1, int(max(node_data['degree']))+2))
                        ax.bar(bins[:-1], hist, alpha=0.7, color=color_map[trait], label=trait, width=0.8)
            else:
                ax.bar(*np.unique(node_data['degree'], return_counts=True))
            ax.set_title("Degree histogram")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Number of Nodes")
        elif plot_type == 'ntraits':
            ax.scatter(node_data['degree'], node_data['ntraits'], color=node_data['color'], s=10)
            ax.set_title("Degree vs Traits")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Traits (unique neighbors)")
        elif plot_type == 'pvalue':
            valid_mask = pd.notna(node_data['neglog10'])
            ax.scatter(node_data.loc[valid_mask, 'degree'], node_data.loc[valid_mask, 'neglog10'],
                       color=node_data.loc[valid_mask, 'color'], s=10)
            ax.set_title("Degree vs pvalue")
            ax.set_xlabel("Degree")
            ax.set_ylabel(r"$\mathrm{-log}_{10}(\mathit{p})$")
        else:
            raise ValueError("Unsupported plot_type: {}".format(plot_type))

        return node_data

    def adjust_overlaps(self, G, pos, node_size_factor=10, nodes_style=None, min_dist_scale=0.04, iterations=200):
        """
        Iteratively adjust node positions to prevent overlap.
        """
        nodes_style = nodes_style or {}
        trait_base_size = nodes_style.get('node_size', 500) * 2
        
        # Calculate radii for all nodes
        radii = {}
        for n, d in G.nodes(data=True):
            if d.get('type') == 'trait':
                r = np.sqrt(trait_base_size)
            else:
                pval = d.get('pvalue', 1.0)
                val = -np.log10(pval) if pval > 0 else 0
                size = node_size_factor * val
                size = max(size, 10) 
                r = np.sqrt(size)
            radii[n] = r
            
        # Normalize radii to layout coordinate system (heuristic)
        max_r = max(radii.values()) if radii else 1.0
        # Map max radius to min_dist_scale of the plot (approx width 2.0)
        scale = (min_dist_scale * 2.0) / max_r
        
        layout_radii = {n: r * scale for n, r in radii.items()}
        
        nodes = list(pos.keys())
        coords = np.array([pos[n] for n in nodes])
        r_arr = np.array([layout_radii[n] for n in nodes])
        
        for _ in range(iterations):
            delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_sq = np.sum(delta**2, axis=2)
            dist = np.sqrt(dist_sq)
            dist[dist < 1e-6] = 1e-6
            
            req_dist = r_arr[:, np.newaxis] + r_arr[np.newaxis, :]
            overlap = req_dist - dist
            
            mask = overlap > 0
            np.fill_diagonal(mask, False)
            
            if not np.any(mask):
                break
                
            norm_delta = delta / dist[:, :, np.newaxis]
            push = norm_delta * overlap[:, :, np.newaxis] * 0.2
            push = push * mask[:, :, np.newaxis]
            coords += np.sum(push, axis=1)
            
        return {n: coords[i] for i, n in enumerate(nodes)}

    def spring_plot(self, G, k=0.8, seed=42, node_size_factor=10, edge_width_factor=3,
                        nodes_style=None, edges_style=None, with_labels=False, labels_style=None,
                        ax=None):
        """
        Spring plot using NetworkX with overlap removal.
        
        Optimized for:
        1. Non-overlap: High repulsion (k=0.8) + Post-layout overlap removal
        2. LD Clustering: Non-linear weight enhancement (weight^2 * 5)

        :param G: NetworkX graph object
        :param k: spring layout parameter k (controls spacing)
        :param seed: the random state for deterministic node layouts
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param edge_width_factor: factor to multiply LD value for edge width
        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param with_labels: whether to show node labels
        :param labels_style: dict with label styling parameters
        :param ax: Axes object
        """
        nodes_style = nodes_style or {}
        edges_style = edges_style or {}
        labels_style = labels_style or {}

        # Separate nodes by type
        qtl_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'qtl']
        trait_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'trait']

        # Prepare graph for layout calculation
        layout_G = G.copy()
        
        # Apply non-linear weight enhancement for tighter clustering of high LD nodes
        for u, v, d in layout_G.edges(data=True):
            if d.get('type') == 'trait_link':
                # Reduce attraction for trait links to push QTLs away from Trait nodes
                layout_G[u][v]['weight'] = 0.1
            else:
                # For QTL-QTL edges, apply non-linear transformation
                w = d.get('weight', 0.0)
                layout_G[u][v]['weight'] = (w ** 2) * 5

        # Calculate layout
        # k=0.8 provides strong repulsion to avoid overlap
        pos = nx.spring_layout(layout_G, k=k or 0.8, seed=seed, iterations=100)
        
        # Post-processing to remove overlaps
        pos = self.adjust_overlaps(G, pos, node_size_factor=node_size_factor, nodes_style=nodes_style)

        cluster_data = []

        # Draw Clusters (Ellipses)
        if qtl_nodes:
            # Create a subgraph of only QTLs to find connected components based on LD
            G_qtl_sub = G.subgraph(qtl_nodes)
            components = list(nx.connected_components(G_qtl_sub))
            
            for idx, comp in enumerate(components):
                if len(comp) < 3: continue # Only highlight clusters with 3+ nodes
                
                points = np.array([pos[n] for n in comp])
                center = points.mean(axis=0)
                
                try:
                    # Calculate covariance matrix
                    cov = np.cov(points, rowvar=False)
                    vals, vecs = np.linalg.eigh(cov)
                    
                    # Sort eigenvalues and eigenvectors
                    order = vals.argsort()[::-1]
                    vals = vals[order]
                    vecs = vecs[:, order]
                    
                    # Calculate angle
                    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                    
                    # Calculate width and height (2.5 std dev covers ~98%)
                    width, height = 2 * 2.5 * np.sqrt(vals)
                    
                    # Draw ellipse
                    ell = mpatches.Ellipse(xy=center, width=width, height=height, angle=theta, 
                                           color='red', alpha=0.05)
                    ax.add_patch(ell)
                    
                    # Draw border
                    ell_border = mpatches.Ellipse(xy=center, width=width, height=height, angle=theta, 
                                                  edgecolor='red', facecolor='none', linewidth=1, linestyle='--', alpha=0.5)
                    ax.add_patch(ell_border)

                    # Collect cluster data
                    cluster_data.append({
                        "cluster_id": f"cluster_{idx+1}",
                        "nodes": ",".join(comp),
                        "center_x": center[0],
                        "center_y": center[1],
                        "width": width,
                        "height": height,
                        "angle": theta,
                        "num_nodes": len(comp)
                    })

                except Exception:
                    # Fallback for singular cases
                    pass

        # Draw QTL nodes
        if qtl_nodes:
            node_colors = [G.nodes[node]['color'] for node in qtl_nodes]
            node_sizes = [node_size_factor * (-np.log10(G.nodes[node]['pvalue'])) for node in qtl_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=qtl_nodes, node_color=node_colors, node_size=node_sizes, **nodes_style, ax=ax)
            
            if with_labels:
                node_labels = {node: node for node in qtl_nodes}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **labels_style, ax=ax)

        # Draw Trait nodes
        if trait_nodes:
            trait_colors = [G.nodes[node]['color'] for node in trait_nodes]
            # Use a larger fixed size for trait nodes or customizable
            trait_size = nodes_style.get('node_size', 500) * 2
            nx.draw_networkx_nodes(G, pos, nodelist=trait_nodes, node_color=trait_colors, node_size=trait_size, node_shape='H', ax=ax)
            
            # Draw trait labels (white, centered)
            trait_labels = {node: node for node in trait_nodes}
            trait_label_style = labels_style.copy()
            trait_label_style['font_color'] = 'white'
            nx.draw_networkx_labels(G, pos, labels=trait_labels, **trait_label_style, ax=ax)

        # Draw edges
        qtl_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'trait_link']
        trait_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'trait_link']

        if qtl_edges:
            edge_widths = [G[u][v]['weight'] * edge_width_factor for u, v in qtl_edges]
            nx.draw_networkx_edges(G, pos, edgelist=qtl_edges, width=edge_widths, **edges_style, ax=ax)
            
        if trait_edges:
            nx.draw_networkx_edges(G, pos, edgelist=trait_edges, width=1.0, **edges_style, ax=ax)

        return cluster_data

    def kamada_kawai_plot(self, G, node_size_factor=10, edge_width_factor=3,
                        nodes_style=None, edges_style=None, with_labels=False, labels_style=None,
                        ax=None, k=None, seed=None):
        """
        Kamada-Kawai layout plot.
        
        :param G: NetworkX graph object
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param edge_width_factor: factor to multiply LD value for edge width
        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param with_labels: whether to show node labels
        :param labels_style: dict with label styling parameters
        :param ax: Axes object
        :param k: spring layout parameter k (used for relaxation)
        :param seed: random seed for relaxation
        """
        nodes_style = nodes_style or {}
        edges_style = edges_style or {}
        labels_style = labels_style or {}

        # Separate nodes by type
        qtl_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'qtl']
        trait_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'trait']

        # Apply non-linear weight enhancement for tighter clustering of high LD nodes
        layout_G = G.copy()
        for u, v, d in layout_G.edges(data=True):
            if 'weight' in d and d.get('type') != 'trait_link':
                # Square the weight and multiply by a factor to emphasize strong connections
                layout_G[u][v]['weight'] = (d['weight'] ** 2) * 5

        # Use Kamada-Kawai layout for the entire graph
        try:
            # scale controls the overall size of the layout
            pos = nx.kamada_kawai_layout(layout_G, weight='weight', scale=2.0)
        except Exception as e:
            logger.warning(f"Kamada-Kawai layout failed: {e}. Falling back to spring layout.")
            pos = nx.spring_layout(G, k=k or 0.8, seed=seed)

        # Draw QTL nodes
        if qtl_nodes:
            node_colors = [G.nodes[node]['color'] for node in qtl_nodes]
            node_sizes = [node_size_factor * (-np.log10(G.nodes[node]['pvalue'])) for node in qtl_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=qtl_nodes, node_color=node_colors, node_size=node_sizes, **nodes_style, ax=ax)
            
            if with_labels:
                node_labels = {node: node for node in qtl_nodes}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **labels_style, ax=ax)

        # Draw Trait nodes
        if trait_nodes:
            trait_colors = [G.nodes[node]['color'] for node in trait_nodes]
            trait_size = nodes_style.get('node_size', 500) * 2
            nx.draw_networkx_nodes(G, pos, nodelist=trait_nodes, node_color=trait_colors, node_size=trait_size, node_shape='H', ax=ax)
            
            trait_labels = {node: node for node in trait_nodes}
            trait_label_style = labels_style.copy()
            trait_label_style['font_color'] = 'white'
            nx.draw_networkx_labels(G, pos, labels=trait_labels, **trait_label_style, ax=ax)

        # Draw edges
        qtl_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'trait_link']
        trait_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'trait_link']

        if qtl_edges:
            edge_widths = [G[u][v]['weight'] * edge_width_factor for u, v in qtl_edges]
            nx.draw_networkx_edges(G, pos, edgelist=qtl_edges, width=edge_widths, **edges_style, ax=ax)
            
        if trait_edges:
            nx.draw_networkx_edges(G, pos, edgelist=trait_edges, width=1.0, edge_color='gray', style='dashed', alpha=0.5, ax=ax)

    def fruchterman_reingold_plot(self, G, node_size_factor=10, edge_width_factor=3,
                        nodes_style=None, edges_style=None, with_labels=False, labels_style=None,
                        ax=None, seed=42, niter=10000):
        """
        Fruchterman-Reingold layout plot using igraph.
        
        :param G: NetworkX graph object
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param edge_width_factor: factor to multiply LD value for edge width
        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param with_labels: whether to show node labels
        :param labels_style: dict with label styling parameters
        :param ax: Axes object
        :param seed: random seed
        :param niter: number of iterations for the layout
        """
        try:
            import igraph as ig
        except ImportError:
            raise ImportError("The 'igraph' library is required for Fruchterman-Reingold layout. Please install it with 'pip install igraph'.")

        nodes_style = nodes_style or {}
        edges_style = edges_style or {}
        labels_style = labels_style or {}

        # Convert NetworkX graph to igraph
        # Mapping from NX node to index
        node_keys = list(G.nodes())
        node_map = {node: i for i, node in enumerate(node_keys)}
        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
        
        # Apply non-linear weight enhancement
        weights = []
        for u, v in G.edges():
            w = G[u][v].get('weight', 1.0)
            if G[u][v].get('type') != 'trait_link':
                w = (w ** 2) * 5
            weights.append(w)

        g = ig.Graph(len(node_keys), edges)
        g.es['weight'] = weights

        # Calculate layout
        # Generate initial layout for reproducibility
        np.random.seed(seed)
        initial_layout = np.random.rand(len(node_keys), 2).tolist()
        # Increase repulsion by adjusting area parameter implicitly via niter or other params if available
        # igraph's FR layout doesn't have a direct 'k' equivalent exposed easily, but we can scale the result
        layout = g.layout_fruchterman_reingold(weights='weight', niter=niter, seed=initial_layout)
        
        # Scale layout to increase spacing (similar to k in spring_layout)
        layout.scale(2.0)
        
        # Convert layout to pos dict
        pos = {node_keys[i]: layout[i] for i in range(len(node_keys))}

        # Separate nodes by type
        qtl_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'qtl']
        trait_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'trait']

        # Draw QTL nodes
        if qtl_nodes:
            node_colors = [G.nodes[node]['color'] for node in qtl_nodes]
            node_sizes = [node_size_factor * (-np.log10(G.nodes[node]['pvalue'])) for node in qtl_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=qtl_nodes, node_color=node_colors, node_size=node_sizes, **nodes_style, ax=ax)
            
            if with_labels:
                node_labels = {node: node for node in qtl_nodes}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **labels_style, ax=ax)

        # Draw Trait nodes
        if trait_nodes:
            trait_colors = [G.nodes[node]['color'] for node in trait_nodes]
            trait_size = nodes_style.get('node_size', 500) * 2
            nx.draw_networkx_nodes(G, pos, nodelist=trait_nodes, node_color=trait_colors, node_size=trait_size, node_shape='H', ax=ax)
            
            trait_labels = {node: node for node in trait_nodes}
            trait_label_style = labels_style.copy()
            trait_label_style['font_color'] = 'white'
            nx.draw_networkx_labels(G, pos, labels=trait_labels, **trait_label_style, ax=ax)

        # Draw edges
        qtl_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'trait_link']
        trait_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'trait_link']

        if qtl_edges:
            edge_widths = [G[u][v]['weight'] * edge_width_factor for u, v in qtl_edges]
            nx.draw_networkx_edges(G, pos, edgelist=qtl_edges, width=edge_widths, **edges_style, ax=ax)
            
        if trait_edges:
            nx.draw_networkx_edges(G, pos, edgelist=trait_edges, width=1.0, edge_color='gray', style='dashed', alpha=0.5, ax=ax)

    def forceatlas2_plot(self, G, node_size_factor=10, edge_width_factor=3,
                        nodes_style=None, edges_style=None, with_labels=False, labels_style=None,
                        ax=None, seed=42, niter=2000):
        """
        ForceAtlas2 layout plot using fa2.
        
        :param G: NetworkX graph object
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param edge_width_factor: factor to multiply LD value for edge width
        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param with_labels: whether to show node labels
        :param labels_style: dict with label styling parameters
        :param ax: Axes object
        :param seed: random seed (not directly used by fa2 but kept for consistency)
        :param niter: number of iterations for the layout
        """
        try:
            from fa2 import ForceAtlas2
        except ImportError:
            raise ImportError("The 'fa2' library is required for ForceAtlas2 layout. Please install it with 'pip install fa2'.")

        # Compatibility fix for NetworkX 3.0+ where to_scipy_sparse_matrix is removed
        if not hasattr(nx, 'to_scipy_sparse_matrix'):
            def to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight='weight', format='csr'):
                return nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight, format=format)
            nx.to_scipy_sparse_matrix = to_scipy_sparse_matrix

        nodes_style = nodes_style or {}
        edges_style = edges_style or {}
        labels_style = labels_style or {}

        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED in fa2)
            edgeWeightInfluence=2.0, # Increased influence of weights (LD^2)

            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=50.0 # Greatly increased repulsion to prevent overlap
            ,strongGravityMode=False
            ,gravity=1.0

            # Log
            ,verbose=False
        )

        # Apply non-linear weight enhancement
        layout_G = G.copy()
        for u, v, d in layout_G.edges(data=True):
            if 'weight' in d and d.get('type') != 'trait_link':
                layout_G[u][v]['weight'] = (d['weight'] ** 2) * 5

        # Calculate layout
        pos = forceatlas2.forceatlas2_networkx_layout(layout_G, pos=None, iterations=niter)

        # Separate nodes by type
        qtl_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'qtl']
        trait_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'trait']

        # Draw QTL nodes
        if qtl_nodes:
            node_colors = [G.nodes[node]['color'] for node in qtl_nodes]
            node_sizes = [node_size_factor * (-np.log10(G.nodes[node]['pvalue'])) for node in qtl_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=qtl_nodes, node_color=node_colors, node_size=node_sizes, **nodes_style, ax=ax)
            
            if with_labels:
                node_labels = {node: node for node in qtl_nodes}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **labels_style, ax=ax)

        # Draw Trait nodes
        if trait_nodes:
            trait_colors = [G.nodes[node]['color'] for node in trait_nodes]
            trait_size = nodes_style.get('node_size', 500) * 2
            nx.draw_networkx_nodes(G, pos, nodelist=trait_nodes, node_color=trait_colors, node_size=trait_size, node_shape='H', ax=ax)
            
            trait_labels = {node: node for node in trait_nodes}
            trait_label_style = labels_style.copy()
            trait_label_style['font_color'] = 'white'
            nx.draw_networkx_labels(G, pos, labels=trait_labels, **trait_label_style, ax=ax)

        # Draw edges
        qtl_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'trait_link']
        trait_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'trait_link']

        if qtl_edges:
            edge_widths = [G[u][v]['weight'] * edge_width_factor for u, v in qtl_edges]
            nx.draw_networkx_edges(G, pos, edgelist=qtl_edges, width=edge_widths, **edges_style, ax=ax)
            
        if trait_edges:
            nx.draw_networkx_edges(G, pos, edgelist=trait_edges, width=1.0, edge_color='gray', style='dashed', alpha=0.5, ax=ax)

    def plot(self, nodes_style=None, edges_style=None, labels_style=None,
                color_map=None, layout="spring", k=0.3, seed=42,
                edge_width_factor=3, node_size_factor=10, with_labels=False, 
                ax_rank=None, ax_hist=None, ax_ntraits=None, ax_pvalue=None, ax_network=None,
                ):
        """
        Plot QTL network from different traits using LD value

        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param labels_style: dict with label styling parameters
        :param color_map: custom color mapping dict {trait: color}
        :param layout: graph layout ('spring', 'kamada_kawai', 'fruchterman_reingold', 'forceatlas2')
        :param k: spring layout parameter k
        :param seed: the random state for deterministic node layouts
        :param edge_width_factor: factor to multiply LD value for edge width
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param min_weight: minimum weight for edges
        :param with_labels: whether to show node labels
        :param ax_rank: Axes for degree rank plot (required)
        :param ax_hist: Axes for degree histogram plot (required)
        :param ax_ntraits: Axes for degree vs ntraits plot (required)
        :param ax_pvalue: Axes for degree vs -log10(pvalue) plot (required)
        :param ax_network: Axes for network layout plot (required)
        :return: Tuple of (edge_df, G, node_metrics, cluster_data)
        """
        # default style for network plot
        default_nodes_style = {
            'alpha': 0.8,
            'linewidths': 1,
        }

        default_edges_style = {
            'alpha': 0.5,
            'edge_color': 'gray',
            'style': 'solid',
            'arrows': True,
            'arrowsize': 10,
            'arrowstyle': None,
            'connectionstyle': 'arc3,rad=0.2'
        }

        default_labels_style = {
            'font_size': 8,
            'font_color': 'black',
            'font_weight': 'normal',
            'font_family': 'sans-serif',
        }

        # modify default style
        if nodes_style is None:
            nodes_style = {}
        nodes_style = {**default_nodes_style, **nodes_style}
        
        if edges_style is None:
            edges_style = {}
        edges_style = {**default_edges_style, **edges_style}
        
        if labels_style is None:
            labels_style = {}
        labels_style = {**default_labels_style, **labels_style}

        if self.blocks is None or self.blocks.empty:
            # If blocks are missing but edge_df is present, we can proceed if we can infer node info
            if self.edge_df is None or self.edge_df.empty:
                raise ValueError("No QTL blocks loaded. Please run load_data() first.")
            else:
                logger.warning("No QTL blocks loaded. Inferring node attributes from edge data.")
        
        if self.edge_df is None or self.edge_df.empty:
            raise ValueError("No LD edges detected. Please run process_data() before plot().")

        # Require all axes to be provided explicitly (no auto-creation inside plot)
        if any(a is None for a in (ax_rank, ax_hist, ax_ntraits, ax_pvalue, ax_network)):
            raise ValueError("Axes must be provided for all panels: ax_rank, ax_hist, ax_ntraits, ax_pvalue, ax_network")

        # create graph object - directly from edge_df
        # Include source and target as edge attributes to correctly map node attributes later
        G = nx.from_pandas_edgelist(self.edge_df, source='source', target='target', 
                                    edge_attr=['weight', 'trait1', 'trait2', 'lead1', 'lead2', 'pvalue1', 'pvalue2', 'source', 'target'])

        # create node to trait mapping and add pvalue attribute
        node_to_trait = {}
        node_to_pvalue = {}
        
        if self.blocks is not None and not self.blocks.empty:
            for _, row in self.blocks.iterrows():
                if row['qtl'] in G.nodes:
                    G.nodes[row['qtl']]['trait'] = row['trait']
                    G.nodes[row['qtl']]['pvalue'] = row['pvalue']
                    G.nodes[row['qtl']]['type'] = 'qtl'
                    node_to_trait[row['qtl']] = row['trait']
                    node_to_pvalue[row['qtl']] = row['pvalue']
        
        # Infer missing node attributes from edges
        for u, v, d in G.edges(data=True):
            src = d.get('source')
            # Determine which node corresponds to source/trait1 and target/trait2
            if u == src:
                u_trait, u_pval = d.get('trait1'), d.get('pvalue1')
                v_trait, v_pval = d.get('trait2'), d.get('pvalue2')
            else:
                u_trait, u_pval = d.get('trait2'), d.get('pvalue2')
                v_trait, v_pval = d.get('trait1'), d.get('pvalue1')

            if 'trait' not in G.nodes[u]:
                G.nodes[u]['trait'] = u_trait
                G.nodes[u]['pvalue'] = u_pval
                G.nodes[u]['type'] = 'qtl'
            if 'trait' not in G.nodes[v]:
                G.nodes[v]['trait'] = v_trait
                G.nodes[v]['pvalue'] = v_pval
                G.nodes[v]['type'] = 'qtl'

        # get all unique traits
        if self.blocks is not None and self.blocks.empty:
            all_traits = list(self.blocks['trait'].unique())
        else:
            all_traits = sorted(list(set(nx.get_node_attributes(G, 'trait').values())))
        
        # Filter traits to only those present in the graph (after weight filtering)
        traits = [t for t in all_traits if any(G.nodes[n].get('trait') == t for n in G.nodes if G.nodes[n].get('type') == 'qtl')]

        # if no custom color mapping provided, create one
        if color_map is None:
            n_traits = len(all_traits)
            
            # select appropriate color mapping
            if n_traits <= 10:
                cmap_name = 'tab10'
            elif n_traits <= 20:
                cmap_name = 'tab20'
            else:
                # for more traits, use continuous color mapping
                cmap_name = 'viridis'
            
            cmap = mpl.colormaps.get_cmap(cmap_name)
            colors = cmap(range(n_traits))

            color_map = {trait: colors[i] for i, trait in enumerate(all_traits)}

        # add color attribute to nodes
        for node in G.nodes:
            if G.nodes[node].get('type') == 'qtl':
                G.nodes[node]['color'] = color_map[G.nodes[node]['trait']]

        # Add Trait nodes and edges
        for trait in traits:
            if trait not in G.nodes:
                G.add_node(trait)
            G.nodes[trait]['trait'] = trait
            G.nodes[trait]['type'] = 'trait'
            G.nodes[trait]['color'] = color_map[trait]
            G.nodes[trait]['pvalue'] = 1.0 # Dummy pvalue
            
            # Connect all QTLs of this trait to the trait node
            # If blocks are available, use them to find all QTLs for a trait
            if self.blocks is not None and not self.blocks.empty:
                qtl_nodes = self.blocks[self.blocks['trait'] == trait]['qtl']
                for qtl in qtl_nodes:
                    if qtl in G.nodes:
                        G.add_edge(qtl, trait, weight=0.2, type='trait_link')
            else:
                # Infer from graph nodes
                for node in G.nodes:
                    if G.nodes[node].get('type') == 'qtl' and G.nodes[node].get('trait') == trait:
                        G.add_edge(node, trait, weight=0.2, type='trait_link')

        # degree-related plots (all four in the first row)
        # Use subgraph of QTL nodes for degree statistics to exclude trait nodes
        G_qtl = G.subgraph([n for n, d in G.nodes(data=True) if d.get('type') == 'qtl'])

        if ax_rank is not None:
            node_metrics = self.degree_plot(G_qtl, color_map=color_map, plot_type="rank", ax=ax_rank)
        else:
            node_metrics = None
        if ax_hist is not None:
            self.degree_plot(G_qtl, color_map=color_map, plot_type="histogram", ax=ax_hist)
        if ax_ntraits is not None:
            self.degree_plot(G_qtl, color_map=color_map, plot_type="ntraits", ax=ax_ntraits)
        if ax_pvalue is not None:
            self.degree_plot(G_qtl, color_map=color_map, plot_type="pvalue", ax=ax_pvalue)

        # network layout (second row, full width)
        cluster_data = []
        if ax_network is not None:
            logger.info(f"Generating network layout using '{layout}' method.")
            if layout == "spring":
                cluster_data = self.spring_plot(G, k=k, seed=seed, node_size_factor=node_size_factor, edge_width_factor=edge_width_factor,
                                 nodes_style=nodes_style, edges_style=edges_style, with_labels=with_labels, labels_style=labels_style, ax=ax_network)
            elif layout == "kamada_kawai":
                self.kamada_kawai_plot(G, node_size_factor=node_size_factor, edge_width_factor=edge_width_factor,
                                 nodes_style=nodes_style, edges_style=edges_style, with_labels=with_labels, labels_style=labels_style, ax=ax_network,
                                 k=k, seed=seed)
            elif layout == "fruchterman_reingold":
                self.fruchterman_reingold_plot(G, seed=seed, node_size_factor=node_size_factor, edge_width_factor=edge_width_factor,
                                 nodes_style=nodes_style, edges_style=edges_style, with_labels=with_labels, labels_style=labels_style, ax=ax_network)
            elif layout == "forceatlas2":
                self.forceatlas2_plot(G, seed=seed, node_size_factor=node_size_factor, edge_width_factor=edge_width_factor,
                                 nodes_style=nodes_style, edges_style=edges_style, with_labels=with_labels, labels_style=labels_style, ax=ax_network)
            else:
                raise ValueError(f"Unknown layout: {layout}")

        # add LD weight legend
        # generate some representative LD values
        ld_values = np.linspace(self.edge_df['weight'].min(), self.edge_df['weight'].max(), num=4)
        ld_legend_elements = [plt.Line2D([0], [0], color=edges_style['edge_color'], 
                                        linewidth=ld * edge_width_factor,
                                        alpha=edges_style['alpha'])
                            for ld in ld_values]

        # add p-value legend
        p_values = []
        p_legend_elements = []
        
        if self.blocks is not None and not self.blocks.empty:
             min_p = self.blocks['pvalue'].min()
             max_p = self.blocks['pvalue'].max()
             p_values = np.linspace(min_p, max_p, num=4)
        else:
             # Extract p-values from graph nodes
             qtl_pvalues = [d['pvalue'] for n, d in G.nodes(data=True) if d.get('type') == 'qtl' and 'pvalue' in d]
             if qtl_pvalues:
                 min_p = min(qtl_pvalues)
                 max_p = max(qtl_pvalues)
                 p_values = np.linspace(min_p, max_p, num=4)

        if len(p_values) > 0:
            p_legend_elements = [plt.Line2D([0], [0], marker='o', color='black', 
                                        markersize=np.sqrt(node_size_factor * (-np.log10(p)) / np.pi) * 2,
                                        linestyle='none', markerfacecolor='none')
                                for p in p_values]

        # LD and p-value legends only make sense with a network axis
        if ax_network is not None:
            legend = ax_network.legend(
                ld_legend_elements,
                [f'{ld:.1f}' for ld in ld_values],
                title="LD value",
                loc='upper left',
                bbox_to_anchor=(1.05, 0.8),
                frameon=False,
                borderaxespad=0.0,
            )
            ax_network.add_artist(legend)
            
            if len(p_legend_elements) > 0:
                ax_network.legend(
                    p_legend_elements,
                    [f'{p:.1e}' for p in p_values],
                    title="P-value",
                    loc='upper left',
                    bbox_to_anchor=(1.05, 0.5),
                    frameon=False,
                    borderaxespad=0.0,
                )
            ax_network.set_title('QTL network')
            ax_network.axis('off')

        return self.edge_df, G, node_metrics, cluster_data


class QTLReport(QTL):
    def __init__(self):
        """
        Initialize the QTLReport class.
        """
        super().__init__()
        self.template_dir = Path(__file__).resolve().parent / "templates"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=False,
        )

    def _load_blocks(self, blocks_files: List[str], blocks_names: Optional[List[str]] = None) -> pd.DataFrame:
        if not blocks_files:
            raise ValueError("At least one QTL blocks file must be provided.")

        if blocks_names and len(blocks_names) != len(blocks_files):
            raise ValueError("Length of blocks_names must match the number of blocks_files.")

        frames: List[pd.DataFrame] = []
        for idx, block_file in enumerate(blocks_files):
            if not os.path.exists(block_file):
                raise FileNotFoundError(f"Blocks file not found: {block_file}")

            qtl_df = self.read_block_file(block_file)
            if qtl_df.empty:
                logger.warning(f"Block file '{block_file}' is empty and will be skipped.")
                continue

            label = None
            if blocks_names:
                label = str(blocks_names[idx])
            else:
                label = os.path.splitext(os.path.basename(block_file))[0]

            qtl_df = qtl_df.copy()
            if "trait" in qtl_df.columns:
                qtl_df["trait"] = qtl_df["trait"].astype(str)
                if blocks_names:
                    qtl_df["trait"] = label
            else:
                qtl_df["trait"] = label

            qtl_df["source_file"] = os.path.basename(block_file)
            frames.append(qtl_df)

        if not frames:
            raise ValueError("No QTL blocks could be loaded from the provided files.")

        data = pd.concat(frames, ignore_index=True)

        required_cols = {"lead", "pvalue", "trait"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Combined blocks table is missing required columns: {sorted(missing_cols)}"
            )

        # type conversions and derived metrics
        data["lead"] = data["lead"].astype(str)
        data["trait"] = data["trait"].astype(str)
        data["pvalue"] = pd.to_numeric(data["pvalue"], errors="coerce")
        data = data.dropna(subset=["lead", "trait", "pvalue"])

        if "kb" not in data.columns and {"bp1", "bp2"}.issubset(data.columns):
            data["kb"] = (pd.to_numeric(data["bp2"], errors="coerce") - pd.to_numeric(data["bp1"], errors="coerce")) / 1000.0

        if "kb" in data.columns:
            data["kb"] = pd.to_numeric(data["kb"], errors="coerce")

        if "ps" in data.columns:
            data["ps"] = pd.to_numeric(data["ps"], errors="coerce")

        if "chr" in data.columns:
            data["chr"] = data["chr"].astype(str)

        return data.reset_index(drop=True)

    @staticmethod
    def _safe_float(value: Optional[float]) -> Optional[float]:
        if pd.isna(value):
            return None
        return float(value)

    def _prepare_context(self, blocks_df: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
        total_blocks = len(blocks_df)
        if total_blocks == 0:
            raise ValueError("No blocks available to summarise.")

        traits = blocks_df["trait"].unique().tolist()
        trait_counts = (
            blocks_df.groupby("trait")
            .size()
            .sort_values(ascending=False)
        )
        chrom_counts = (
            blocks_df["chr"].value_counts().sort_values(ascending=False)
            if "chr" in blocks_df.columns
            else pd.Series(dtype="int")
        )

        kb_series = blocks_df["kb"].dropna() if "kb" in blocks_df.columns else pd.Series(dtype="float")
        kb_summary = {
            "min": self._safe_float(kb_series.min()) if not kb_series.empty else None,
            "max": self._safe_float(kb_series.max()) if not kb_series.empty else None,
            "median": self._safe_float(kb_series.median()) if not kb_series.empty else None,
            "mean": self._safe_float(kb_series.mean()) if not kb_series.empty else None,
        }

        cleaned_kb = kb_series.dropna() if not kb_series.empty else pd.Series(dtype="float")
        kb_hist_data: List[Dict[str, Any]] = []
        kb_box_stats: Dict[str, Optional[float]] = {}
        if not cleaned_kb.empty:
            bin_count = min(max(len(cleaned_kb) // 2, 5), 20)
            counts, bin_edges = np.histogram(cleaned_kb, bins=bin_count)
            for idx, count in enumerate(counts):
                kb_hist_data.append({
                    "range": f"{bin_edges[idx]:.1f}-{bin_edges[idx + 1]:.1f}",
                    "count": int(count),
                })
            q0, q1, q2, q3, q4 = np.percentile(cleaned_kb, [0, 25, 50, 75, 100])
            kb_box_stats = {
                "min": self._safe_float(q0),
                "q1": self._safe_float(q1),
                "median": self._safe_float(q2),
                "q3": self._safe_float(q3),
                "max": self._safe_float(q4),
            }

        pvalue_summary = None
        if not blocks_df["pvalue"].empty:
            pvalue_summary = {
                "min": self._safe_float(blocks_df["pvalue"].min()),
                "median": self._safe_float(blocks_df["pvalue"].median()),
            }

        # Calculate total unique genes
        all_genes = set()
        if "genes" in blocks_df.columns:
            for genes_raw in blocks_df["genes"].dropna():
                if isinstance(genes_raw, str):
                    for token in genes_raw.split(";"):
                        t = token.strip()
                        if t:
                            all_genes.add(t)
        total_genes = len(all_genes)

        # Calculate per-trait stats
        trait_stats_list = []
        if not blocks_df.empty:
            grouped_traits = blocks_df.groupby("trait")
            for trait, group in grouped_traits:
                # Genes for this trait
                trait_genes = set()
                if "genes" in group.columns:
                    for genes_raw in group["genes"].dropna():
                        if isinstance(genes_raw, str):
                            for token in genes_raw.split(";"):
                                t = token.strip()
                                if t:
                                    trait_genes.add(t)
                
                trait_stats_list.append({
                    "trait": trait,
                    "qtl_count": len(group),
                    "lead_snp_count": group["lead"].nunique(),
                    "gene_count": len(trait_genes),
                    "min_pvalue": self._safe_float(group["pvalue"].min())
                })
        
        # Sort by min_pvalue ascending
        trait_stats_list.sort(key=lambda x: x["min_pvalue"] if x["min_pvalue"] is not None else float('inf'))

        top_blocks = (
            blocks_df.nsmallest(top_n, "pvalue")
            if top_n > 0 else blocks_df
        )

        top_block_rows = []
        for _, row in top_blocks.iterrows():
            genes_raw = row.get("genes")
            if isinstance(genes_raw, str):
                gene_tokens = [token.strip() for token in genes_raw.split(";") if token.strip()]
                gene_count = len(gene_tokens)
            else:
                gene_count = 0
            top_block_rows.append({
                "trait": row["trait"],
                "lead": row["lead"],
                "pvalue": self._safe_float(row["pvalue"]),
                "neglog10": self._safe_float(-np.log10(max(row["pvalue"], 1e-300))),
                "chr": row.get("chr"),
                "position": self._safe_float(row.get("ps")),
                "kb": self._safe_float(row.get("kb")),
                "genes": genes_raw,
                "gene_count": gene_count,
                "source_file": row.get("source_file"),
            })

        trait_counts_data = [
            {"name": trait, "value": int(count)} for trait, count in trait_counts.items()
        ]

        chrom_counts_data = [
            {"name": chrom, "value": int(count)} for chrom, count in chrom_counts.items()
        ]

        trait_order = list(trait_counts.index)

        scatter_source = blocks_df.copy()
        scatter_source = scatter_source.dropna(subset=["pvalue"])
        scatter_source["neglog10"] = -np.log10(np.clip(scatter_source["pvalue"], 1e-300, None))
        scatter_source = scatter_source.replace([np.inf, -np.inf], np.nan).dropna(subset=["neglog10"])
        scatter_source = scatter_source.nsmallest(min(len(scatter_source), 5000), "pvalue")

        trait_pvalue_records: List[Dict[str, Any]] = []
        for row in scatter_source.itertuples():
            kb_val = self._safe_float(getattr(row, "kb", None))
            neglog = self._safe_float(row.neglog10) or 0.0
            trait_pvalue_records.append({
                "trait": row.trait,
                "lead": row.lead,
                "neglog10": neglog,
                "kb": kb_val,
                "pvalue": self._safe_float(row.pvalue),
                "chr": getattr(row, "chr", None),
                "position": self._safe_float(getattr(row, "ps", None)),
            })

        kb_trait_box_data: List[Dict[str, Any]] = []
        if "kb" in blocks_df.columns:
            trait_box_source = blocks_df.copy()
            trait_box_source["kb"] = pd.to_numeric(trait_box_source["kb"], errors="coerce")
            trait_box_source = trait_box_source.dropna(subset=["kb"])
            if not trait_box_source.empty:
                grouped = trait_box_source.groupby("trait")["kb"]
                ordered_traits = trait_order or list(grouped.groups.keys())
                for trait in ordered_traits:
                    if trait not in grouped.groups:
                        continue
                    series = grouped.get_group(trait).dropna()
                    if series.empty:
                        continue
                    q0, q1, q2, q3, q4 = np.percentile(series, [0, 25, 50, 75, 100])
                    kb_trait_box_data.append({
                        "trait": trait,
                        "stats": [
                            self._safe_float(q0),
                            self._safe_float(q1),
                            self._safe_float(q2),
                            self._safe_float(q3),
                            self._safe_float(q4),
                        ],
                    })
                remaining_traits = [trait for trait in grouped.groups.keys() if trait not in ordered_traits]
                for trait in sorted(remaining_traits):
                    series = grouped.get_group(trait).dropna()
                    if series.empty:
                        continue
                    q0, q1, q2, q3, q4 = np.percentile(series, [0, 25, 50, 75, 100])
                    kb_trait_box_data.append({
                        "trait": trait,
                        "stats": [
                            self._safe_float(q0),
                            self._safe_float(q1),
                            self._safe_float(q2),
                            self._safe_float(q3),
                            self._safe_float(q4),
                        ],
                    })

        traits_table = trait_counts.reset_index(name="count")

        context: Dict[str, Any] = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "summary": {
                "total_blocks": int(total_blocks),
                "total_traits": int(len(traits)),
                "total_lead_snps": int(blocks_df["lead"].nunique()),
                "total_genes": total_genes,
                "kb_summary": kb_summary,
                "pvalue_summary": pvalue_summary,
            },
            "traits": traits_table.to_dict("records"),
            "top_blocks": top_block_rows,
            "chart_data": {
                "traitCounts": json.dumps(trait_counts_data, ensure_ascii=False),
                "chromCounts": json.dumps(chrom_counts_data, ensure_ascii=False),
                "traitOrder": json.dumps(trait_order, ensure_ascii=False),
                "traitPvalue": json.dumps(trait_pvalue_records, ensure_ascii=False),
                "kbHistogram": json.dumps(kb_hist_data, ensure_ascii=False),
                "kbBox": json.dumps(kb_box_stats, ensure_ascii=False),
                "kbTraitBox": json.dumps(kb_trait_box_data, ensure_ascii=False),
                "traitStats": json.dumps(trait_stats_list, ensure_ascii=False),
            },
        }

        return context

    def render(
        self,
        blocks_files: List[str],
        out_dir: str = ".",
        out_name: str = "qtl_report",
        blocks_names: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> str:
        blocks_df = self._load_blocks(blocks_files, blocks_names)
        context = self._prepare_context(blocks_df, top_n=top_n)

        if not (self.template_dir / "report.html").exists():
            raise FileNotFoundError(
                f"Report template not found at {self.template_dir / 'report.html'}."
            )

        template = self._env.get_template("report.html")
        html_content = template.render(**context)

        os.makedirs(out_dir, exist_ok=True)
        summary_path = Path(out_dir) / f"{out_name}.summary.csv"
        blocks_df.to_csv(summary_path, index=False)
        logger.info("QTL summary table saved to %s", summary_path)
        output_path = Path(out_dir) / f"{out_name}.html"
        output_path.write_text(html_content, encoding="utf-8")

        logger.info(f"QTL report saved to {output_path}")
        return str(output_path.resolve())


class QTLEffect(QTL):
    def __init__(self):
        """
        Initialize the QTLEffect class for genetic effect analysis.
        """
        super().__init__()

    def run_epistasis(
        self, 
        qtl_file: str, 
        phe_file: str, 
        vcf_file: str, 
        out_dir: str = ".", 
        out_name: str = "epistasis",
        p_threshold: float = 0.05,
        genome: str = None
    ) -> pd.DataFrame:
        """
        Calculate epistatic effects (pairwise interactions) between QTLs.

        :param qtl_file: Path to QTL block file (must contain 'trait', 'chr', 'lead', 'ps')
        :param phe_file: Path to phenotype file (header required, 1st col = sample ID)
        :param vcf_file: Path to VCF file
        :param out_dir: Output directory
        :param out_name: Output file name prefix
        :param p_threshold: P-value threshold for filtering results in the output
        :param genome: Path to genome file (chr, length) for visualization
        :return: DataFrame containing epistasis results
        """
        try:
            import pysam
            import statsmodels.formula.api as smf
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}. Please install pysam and statsmodels.")

        os.makedirs(out_dir, exist_ok=True)
        logger.info("Starting epistasis analysis...")

        # 1. Load Data
        # Load QTLs
        qtl_df = self.read_block_file(qtl_file)
        required_qtl_cols = {"trait", "chr", "lead", "ps"}
        if not required_qtl_cols.issubset(qtl_df.columns):
            raise ValueError(f"QTL file missing required columns: {required_qtl_cols - set(qtl_df.columns)}")

        # Load Phenotypes
        # Detect separator (tab or comma)
        with open(phe_file, 'r') as f:
            header_line = f.readline()
            sep = '\t' if '\t' in header_line else ','
        
        phe_df = pd.read_csv(phe_file, sep=sep)
        # Rename first column to sample_id and ensure string type
        phe_df.rename(columns={phe_df.columns[0]: "sample_id"}, inplace=True)
        phe_df["sample_id"] = phe_df["sample_id"].astype(str)
        
        # Load VCF
        if not os.path.exists(vcf_file):
            raise FileNotFoundError(f"VCF file not found: {vcf_file}")
        vcf = pysam.VariantFile(vcf_file)
        vcf_samples = list(vcf.header.samples)
        
        results = []

        # 2. Process each trait
        traits = qtl_df["trait"].unique()
        for trait in traits:
            logger.info(f"Processing trait: {trait}")
            
            if trait not in phe_df.columns:
                logger.warning(f"Trait '{trait}' not found in phenotype file. Skipping.")
                continue
            
            # Get lead SNPs for this trait
            trait_qtls = qtl_df[qtl_df["trait"] == trait]
            
            # Create a lookup for QTL intervals
            qtl_intervals = {}
            for _, row in trait_qtls.iterrows():
                # Ensure bp1 and bp2 exist, otherwise use ps
                bp1 = row.get("bp1", row["ps"])
                bp2 = row.get("bp2", row["ps"])
                qtl_intervals[row["lead"]] = f"{row['chr']}:{bp1}-{bp2}"

            lead_snps = trait_qtls[["chr", "ps", "lead"]].drop_duplicates()
            
            if len(lead_snps) < 2:
                logger.info(f"Trait '{trait}' has fewer than 2 QTLs. Skipping interaction analysis.")
                continue

            # Identify common samples
            # Filter phenotype samples that have data for this trait
            valid_phe_samples = phe_df[phe_df[trait].notna()]["sample_id"].tolist()
            common_samples = sorted(list(set(vcf_samples) & set(valid_phe_samples)))
            
            if len(common_samples) < 10:
                logger.warning(f"Not enough common samples ({len(common_samples)}) for trait '{trait}'. Skipping.")
                continue
            
            # Extract Genotypes
            genotypes = {}
            valid_snp_ids = []
            
            for _, row in lead_snps.iterrows():
                chrom = str(row["chr"])
                pos = int(row["ps"])
                snp_id = row["lead"]
                
                try:
                    # pysam fetch uses 0-based start, 1-based end for region? 
                    # Actually fetch(contig, start, stop) is 0-based, half-open [start, stop).
                    # VCF pos is 1-based. So base at pos is index pos-1.
                    # fetch(chrom, pos-1, pos) gets the specific base.
                    records = list(vcf.fetch(chrom, pos-1, pos))
                    
                    # Find exact match for position
                    target_rec = None
                    for rec in records:
                        if rec.pos == pos:
                            target_rec = rec
                            break
                    
                    if target_rec:
                        # Extract GTs
                        gts = []
                        for sample in common_samples:
                            gt = target_rec.samples[sample]["GT"]
                            # Convert to dosage: 0/0->0, 0/1->1, 1/1->2
                            if gt == (None, None) or None in gt:
                                gts.append(np.nan)
                            else:
                                gts.append(sum(gt))
                        
                        genotypes[snp_id] = gts
                        valid_snp_ids.append(snp_id)
                    else:
                        logger.warning(f"SNP {snp_id} at {chrom}:{pos} not found in VCF.")
                        
                except ValueError:
                    logger.warning(f"Could not fetch region {chrom}:{pos} from VCF (check chromosome names).")
                except Exception as e:
                    logger.warning(f"Error extracting SNP {snp_id}: {e}")

            if len(valid_snp_ids) < 2:
                logger.info(f"Not enough valid SNPs extracted for trait '{trait}'.")
                continue

            # Prepare Analysis DataFrame
            analysis_df = pd.DataFrame(genotypes, index=common_samples)
            analysis_df["Y"] = phe_df.set_index("sample_id").loc[common_samples, trait]
            
            # Pairwise Interaction Analysis
            # Iterate over all unique pairs
            for snp1, snp2 in combinations(valid_snp_ids, 2):
                # Prepare subset
                sub_df = analysis_df[["Y", snp1, snp2]].dropna()
                
                if len(sub_df) < 10:
                    continue
                
                # Check if SNPs are segregating in this subset
                if sub_df[snp1].nunique() < 2 or sub_df[snp2].nunique() < 2:
                    continue

                # OLS Model: Y ~ SNP1 * SNP2
                # Use Q() to handle special characters in column names
                formula = f"Y ~ Q('{snp1}') * Q('{snp2}')"
                
                try:
                    model = smf.ols(formula, data=sub_df).fit()
                    
                    # Interaction term name
                    interaction_term = f"Q('{snp1}'):Q('{snp2}')"
                    
                    if interaction_term in model.pvalues:
                        pval = model.pvalues[interaction_term]
                        beta = model.params[interaction_term]
                        
                        # Store result
                        results.append({
                            "trait": trait,
                            "snp1": snp1,
                            "qtl1": qtl_intervals.get(snp1, "NA"),
                            "snp2": snp2,
                            "qtl2": qtl_intervals.get(snp2, "NA"),
                            "beta": beta,
                            "p": pval,
                            "n": len(sub_df),
                            "r": model.rsquared
                        })
                except Exception as e:
                    # logger.debug(f"Model fit failed for {snp1} x {snp2}: {e}")
                    pass

        # 3. Save Results
        if results:
            res_df = pd.DataFrame(results)
            
            # Filter by p-value
            res_df = res_df[res_df["p"] <= p_threshold]
            
            if res_df.empty:
                logger.info(f"No significant interactions found (p <= {p_threshold}).")
                return pd.DataFrame()

            # Sort by p-value
            res_df = res_df.sort_values("p")
            
            out_path = os.path.join(out_dir, f"{out_name}.csv")
            res_df.to_csv(out_path, index=False)
            logger.info(f"Epistasis analysis completed. Results saved to {out_path}")
            
            # Basic Visualization (Optional but requested)
            self.plot_epistasis(res_df, out_dir, out_name, p_cutoff=p_threshold, genome=genome)
            
            return res_df
        else:
            logger.info("No interactions calculated.")
            return pd.DataFrame()

    def plot_epistasis(self, df: pd.DataFrame, out_dir: str, out_name: str, p_cutoff: float = 0.05, genome: str = None):
        """
        Plot epistasis results: Network and Linear Interaction (if genome provided).
        """

        def _plot_network(df: pd.DataFrame, out_dir: str, out_name: str, p_cutoff: float = 0.05):
            """
            Plot epistasis network for significant interactions.
            """
            # df is already filtered by p_threshold in run_epistasis, but we can filter again if needed
            sig_df = df[df["p"] < p_cutoff]
            if sig_df.empty:
                return

            # Create graph
            G = nx.Graph()
            for _, row in sig_df.iterrows():
                w = -np.log10(row["p"])
                G.add_edge(row["snp1"], row["snp2"], weight=w, trait=row["trait"])

            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            
            # Draw
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title(f"Epistasis Network (P < {p_cutoff})")
            plt.axis('off')
            
            png_path = os.path.join(out_dir, f"{out_name}.network.png")
            pdf_path = os.path.join(out_dir, f"{out_name}.network.pdf")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Epistasis network plot saved to {png_path} and {pdf_path}")

        def _plot_linear_interaction(df: pd.DataFrame, out_dir: str, out_name: str, genome: str, p_cutoff: float = 0.05):
            """
            Plot linear interaction map with Bezier curves.
            """
            import matplotlib.path as mpath
            import matplotlib.patches as mpatches
            import re
            
            sig_df = df[df["p"] <= p_cutoff].copy()
            if sig_df.empty:
                return

            # Load genome
            try:
                genome = pd.read_csv(genome, sep="\t", header=None, names=["chr", "len"])
                genome["len"] = pd.to_numeric(genome["len"], errors='coerce')
                genome = genome.dropna(subset=["len"])
            except Exception as e:
                logger.error(f"Failed to read genome file: {e}")
                return

            # Calculate cumulative offsets with gaps
            gap = genome["len"].sum() * 0.01
            offsets = []
            current_offset = 0
            for length in genome["len"]:
                offsets.append(current_offset)
                current_offset += length + gap
            genome["offset"] = offsets
            genome_dict = genome.set_index("chr")["offset"].to_dict()
            total_len = current_offset - gap
            
            # Traits
            traits = sorted(sig_df["trait"].unique())
            n_traits = len(traits)
            # Assign colors
            cmap = plt.get_cmap("tab20")
            trait_colors = {t: cmap(i % 20) for i, t in enumerate(traits)}
            trait_map = {t: i + 1 for i, t in enumerate(traits)} # Start from 1
            
            # Setup figure
            # Height depends on number of traits
            fig, ax = plt.subplots(figsize=(15, max(4, n_traits * 1.5)))
            
            # Draw genome at bottom (y=0)
            for i, row in genome.iterrows():
                start = row["offset"]
                end = start + row["len"]
                # Alternating colors
                color = "#e0e0e0" if i % 2 == 0 else "#c0c0c0"
                ax.plot([start, end], [0, 0], color=color, linewidth=10, solid_capstyle="butt")
                
                # Add chr label at the bottom track
                mid = (start + end) / 2
                label = str(row["chr"])
                label = re.sub(r"^(chr|scaffold|chromosome)", "", label, flags=re.IGNORECASE)
                ax.text(mid, -0.3, label, ha="center", va="top", fontsize=16, rotation=0)

            # Draw vertical chromosome background
            for i, row in genome.iterrows():
                start = row["offset"]
                width = row["len"]
                # Alternating colors
                color = "#f0f0f0" if i % 2 == 0 else "#e8e8e8"
                # Draw rectangle from y=0.5 to y=n_traits+0.5
                rect = mpatches.Rectangle((start, 0.5), width, n_traits, 
                                        facecolor=color, edgecolor="none", zorder=0)
                ax.add_patch(rect)

            # Draw trait labels
            for trait, y in trait_map.items():
                ax.text(-total_len * 0.01, y, trait, ha="right", va="center", fontsize=14, color='black')

            # Helper to parse position from "chr:bp1-bp2"
            def parse_pos(qtl_str):
                try:
                    c, span = qtl_str.split(":")
                    b1, b2 = span.split("-")
                    pos = (float(b1) + float(b2)) / 2
                    return c, pos
                except:
                    return None, None

            # Draw interactions
            for _, row in sig_df.iterrows():
                t = row["trait"]
                if t not in trait_map: continue
                y = trait_map[t]
                color = trait_colors[t]
                
                c1, p1 = parse_pos(row["qtl1"])
                c2, p2 = parse_pos(row["qtl2"])
                
                if c1 and c2 and c1 in genome_dict and c2 in genome_dict:
                    x1 = genome_dict[c1] + p1
                    x2 = genome_dict[c2] + p2
                    
                    # Draw dots
                    ax.plot(x1, y, 'o', color=color, markersize=10, zorder=3)
                    ax.plot(x2, y, 'o', color=color, markersize=10, zorder=3)

                    # Draw Bezier curve (Cubic)
                    # Height of arc. 
                    arc_height = 0.4
                    
                    # Cubic Bezier Curve with control points above the start/end points
                    # This creates a "vertical" departure/arrival, looking like an arch
                    verts = [
                        (x1, y),
                        (x1, y + arc_height), # Control point 1
                        (x2, y + arc_height), # Control point 2
                        (x2, y)
                    ]
                    codes = [
                        mpath.Path.MOVETO,
                        mpath.Path.CURVE4,
                        mpath.Path.CURVE4,
                        mpath.Path.CURVE4
                    ]
                    path = mpath.Path(verts, codes)
                    
                    # Line width based on -log10(p)
                    lw = min(3, -np.log10(row["p"]) / 2)
                    
                    patch = mpatches.PathPatch(path, facecolor="none", edgecolor=color, alpha=0.6, lw=lw, zorder=2)
                    ax.add_patch(patch)

            ax.set_xlim(-total_len * 0.02, total_len * 1.02)
            ax.set_ylim(-0.5, n_traits + 0.8)
            ax.axis("off")
            
            plt.tight_layout()
            png_path = os.path.join(out_dir, f"{out_name}.interaction.png")
            pdf_path = os.path.join(out_dir, f"{out_name}.interaction.pdf")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Epistasis interaction plot saved to {png_path} and {pdf_path}")

        # 1. Network Plot
        _plot_network(df, out_dir, out_name, p_cutoff)

        # 2. Linear Interaction Plot
        if genome:
            if os.path.exists(genome):
                _plot_linear_interaction(df, out_dir, out_name, genome, p_cutoff)
            else:
                logger.warning(f"Genome file not found: {genome}. Skipping interaction plot.")
