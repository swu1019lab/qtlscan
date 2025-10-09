import pandas as pd
import numpy as np
import subprocess
import os
import uuid
from collections import defaultdict
from multiprocessing import cpu_count
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

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
        pass

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
        return pd.read_csv(ld_file, delim_whitespace=True, 
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

    def calculate_ld(
        self,
        snps: list,
        vcf_file: str,
        ld_window_r2: float = 0.2,
        ld_window_kb: int = 100,
        ld_window: int = 10,
        threads: int = 1,
        out_dir: str = "."
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
        """
        if len(snps) < 2:
            raise ValueError("At least two SNPs are required for LD calculation.")

        os.makedirs(os.path.join(out_dir, "ld"), exist_ok=True)
        snp_file = os.path.join(out_dir, f"ld/snps.{os.getpid()}.{uuid.uuid4().hex}.txt")
        pd.Series(snps).to_csv(snp_file, index=False, header=False)

        # Calculate LD
        ld_path = os.path.join(out_dir, f"ld/plink.{os.getpid()}.{uuid.uuid4().hex}")
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
        ) -> pd.DataFrame:
        """Process GWAS data and find QTL blocks."""
        logger.info("Processing GWAS QTL data...")

        qtl_blocks = pd.DataFrame()
        if not gwas_data.empty:
            # Calculate LD
            all_snps = gwas_data["rs"].drop_duplicates().tolist()
            logger.info(f"Calculating LD for {len(all_snps)} SNPs...")
            ld_df = self.calculate_ld(all_snps, vcf_file, ld_window_r2, ld_window_kb, ld_window, threads, out_dir)
            logger.info(f"LD calculation completed.")

            # Find QTL blocks
            qtl_blocks = self.find_qtl_blocks(ld_df, gwas_data, min_snps, ann_data)
        return qtl_blocks

    def find_qtl_blocks(self, ld_df: pd.DataFrame, qtl_df: pd.DataFrame, min_snps: int = 3, gff_df=None) -> pd.DataFrame:
        """Find QTL blocks based on LD results (optimized with Union-Find)."""
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
        blocks = []
        for i, qtl_file in enumerate(blocks_files):
            if blocks_names:
                base_name = blocks_names[i]
            else:
                base_name = os.path.splitext(os.path.basename(qtl_file))[0]
                base_name = str(base_name.split(".")[0]) + "_" + str(i)
            qtl_df = self.read_block_file(qtl_file)
            blocks.append(qtl_df.assign(trait=base_name))
        self.blocks = pd.concat(blocks, ignore_index=True).assign(qtl=lambda x: "q" + x.index.astype(str))
        logger.info(f"Loaded {len(self.blocks)} QTL blocks from {len(blocks_files)} files.")

    def process_data(self, vcf_path: str, min_weight: float = None):
        """Process data to create QTL network based on LD values.
        
        :param vcf_path: Path to VCF file for LD calculation
        :param min_weight: Minimum weight (LD value) for edges
        """
        if self.blocks is None or self.blocks.empty:
            raise ValueError("No QTL blocks data loaded. Please run load_data() first.")
        logger.info("Processing QTL network data...")
        # calculate LD using all SNPs
        all_snps = self.blocks["lead"].unique()
        logger.info(f"Calculating LD for {len(all_snps)} SNPs...")
        ld_df = self.calculate_ld(all_snps, vcf_path, 0, 1000, 999999, threads=cpu_count())
        if ld_df.empty:
            raise ValueError("No LD values found for the given SNPs")

        # create all paired-qtl with NetworkX compatible format (source, target, weight)
        qtl_pairs = []
        for i, row1 in self.blocks.iterrows():
            for j, row2 in self.blocks.iterrows():
                # avoid self-pairing and duplicate pairing
                if i >= j:
                    continue
                
                # find LD value between two lead SNPs
                ld_value = ld_df[
                    ((ld_df['SNP_A'] == row1['lead']) & (ld_df['SNP_B'] == row2['lead'])) |
                    ((ld_df['SNP_A'] == row2['lead']) & (ld_df['SNP_B'] == row1['lead']))
                ]['R2'].values
                
                # if LD value exists, add to paired-qtl list
                if len(ld_value) > 0:
                    qtl_pairs.append({
                        'source': row1['qtl'],
                        'target': row2['qtl'],
                        'weight': float(ld_value[0]),
                        # keep extra information as edge attributes
                        'trait1': row1['trait'],
                        'lead1': row1['lead'],
                        'pvalue1': row1['pvalue'],
                        'trait2': row2['trait'],
                        'lead2': row2['lead'],
                        'pvalue2': row2['pvalue'],
                    })
        # convert to DataFrame
        self.edge_df = pd.DataFrame(qtl_pairs)
        if min_weight is not None:
            self.edge_df = self.edge_df[self.edge_df['weight'] >= min_weight]

    def save(self, out_dir = ".", out_name = "qtlnetwork"):
        self.blocks.to_csv(os.path.join(out_dir, f"{out_name}.blocks.csv"), index=False)
        self.edge_df.to_csv(os.path.join(out_dir, f"{out_name}.edges.csv"), index=False)
        logger.info(f"All QTL blocks saved to {os.path.join(out_dir, f'{out_name}.csv')}")

    def degree_plot(self, G, color_map=None, plot_type='rank', export_csv=None, ax=None):
        """
        Degree analysis and visualization

        :param G: NetworkX graph object
        :param color_map: custom color mapping dict {trait: color}
        :param plot_type: plot type ('rank', 'histogram')
        :param export_csv: path to export degree analysis results
        :param ax: Axes object
        """
        # calculate degree for each node
        degree_dict = dict(G.degree())
        
        # create a list containing nodes, degrees, and traits
        node_data = []
        for node, degree in degree_dict.items():
            trait = G.nodes[node]['trait']
            node_data.append({
                'node': node,
                'degree': degree,
                'trait': trait,
                'color': color_map.get(trait, 'blue') if color_map else 'blue',
                'neighbors': ";".join(G.neighbors(node)),
                'related': ";".join([G.nodes[neighbor]['trait'] for neighbor in G.neighbors(node)])
            })
        node_data = pd.DataFrame(node_data)

        # sort by degree value
        node_data.sort_values(by='degree', ascending=False, inplace=True, ignore_index=True)

        if export_csv:
            node_data.to_csv(export_csv, index=False)

        # degree rank plot
        if plot_type == 'rank':
            # plot each node, colored by trait
            ax.scatter(node_data.index, node_data['degree'], 
                        color=node_data['color'], s=10)

            # add lines to observe the trend
            ax.plot(node_data['degree'], "gray", alpha=0.5, linestyle='-')

            ax.set_title("Degree Rank Plot")
            ax.set_ylabel("Degree")
            ax.set_xlabel("Rank")
        # degree histogram plot
        if plot_type == 'histogram':
            # create histogram for each trait
            if color_map:
                traits = list(color_map.keys())
                for trait in traits:
                    # get all degree values for the trait
                    trait_degrees = node_data[node_data['trait'] == trait]['degree']
                    if not trait_degrees.empty:
                        hist, bins = np.histogram(trait_degrees, bins=range(1, max(node_data['degree'])+2))
                        ax.bar(bins[:-1], hist, alpha=0.7, color=color_map[trait], 
                            label=trait, width=0.8)
            else:
                ax.bar(*np.unique(node_data['degree'], return_counts=True))

            ax.set_title("Degree histogram")
            ax.set_xlabel("Degree")
            ax.set_ylabel("# of Nodes")

    def spring_plot(self, G, k=0.3, seed=42, node_size_factor=10, edge_width_factor=3,
                        nodes_style=None, edges_style=None, with_labels=False, labels_style=None,
                        ax=None):
        """
        Spring plot using NetworkX

        :param G: NetworkX graph object
        :param k: spring layout parameter k
        :param seed: the random state for deterministic node layouts
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param edge_width_factor: factor to multiply LD value for edge width
        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param with_labels: whether to show node labels
        :param labels_style: dict with label styling parameters
        :param ax: Axes object
        """
        # set node color list
        node_colors = [G.nodes[node]['color'] for node in G.nodes]

        # calculate node sizes based on -log10(pvalue)
        # Use -log10(pvalue) to determine node size (smaller p-values will have larger nodes)
        node_sizes = [node_size_factor * (-np.log10(G.nodes[node]['pvalue'])) for node in G.nodes]

        # set edge width based on LD value
        edge_widths = [G[u][v]['weight'] * edge_width_factor for u, v in G.edges()]

        pos = nx.spring_layout(G, k=k, seed=seed)  # default use spring layout
        # draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, **nodes_style, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, **edges_style, ax=ax)

        # use qtl as node labels if with_labels is True
        if with_labels:
            node_labels = {node: node for node in G.nodes}
            nx.draw_networkx_labels(G, pos, labels=node_labels, **labels_style, ax=ax)

    def atlas_plot(self, G, min_nodes=1, node_size_factor=10, edge_width_factor=3,
                    nodes_style=None, edges_style=None, with_labels=False, labels_style=None,
                    ax=None):
        """
        Atlas plot using graphviz neato

        :param G: NetworkX graph object
        :param min_nodes: minimum number of nodes in each connected subgraph
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param edge_width_factor: factor to multiply LD value for edge width
        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param with_labels: whether to show node labels
        :param labels_style: dict with label styling parameters
        :param ax: Axes object
        """
        U = nx.Graph()  # graph for union of all graphs in atlas

        # store original nodes due to disjoint union method will renumbered nodes
        ori_nodes = [] # the renumbered nodes will be same as the index of original nodes

        # process each connected subgraph
        for c in nx.connected_components(G):
            if len(c) >= min_nodes:
                H = G.subgraph(c)
                ori_nodes.extend(list(H.nodes()))
                U = nx.disjoint_union(U, H)

        # remove color attribute due to graphviz cannot handle rgba color attribute
        for node in U.nodes():
            if 'color' in U.nodes[node]:
                del U.nodes[node]['color']

        # set node color list
        node_colors = [G.nodes[ori_nodes[node]]['color'] for node in U.nodes()]
        # calculate node sizes based on -log10(pvalue)
        node_sizes = [node_size_factor * (-np.log10(G.nodes[ori_nodes[node]]['pvalue'])) for node in U.nodes()]
        # set edge width based on LD value
        edge_widths = [U[u][v]['weight'] * edge_width_factor for u, v in U.edges()]

        # layout graphs with positions using graphviz neato
        pos = nx.nx_agraph.graphviz_layout(U, prog="neato")

        # draw nodes and edges
        nx.draw_networkx_nodes(U, pos, node_color=node_colors, node_size=node_sizes, **nodes_style, ax=ax)
        nx.draw_networkx_edges(U, pos, width=edge_widths, **edges_style, ax=ax)

        # use qtl as node labels if with_labels is True
        if with_labels:
            node_labels = {node: ori_nodes[node] for node in U.nodes()}
            nx.draw_networkx_labels(U, pos, labels=node_labels, **labels_style, ax=ax)

    def plot(self, nodes_style=None, edges_style=None, labels_style=None,
                color_map=None, layout="spring", k=0.3, seed=42,
                edge_width_factor=3, node_size_factor=10, with_labels=False, 
                export_csv=None, ax1=None, ax2=None, ax3=None):
        """
        Plot QTL network from different traits using LD value

        :param nodes_style: dict with node styling parameters
        :param edges_style: dict with edge styling parameters
        :param labels_style: dict with label styling parameters
        :param color_map: custom color mapping dict {trait: color}
        :param layout: graph layout ('spring', 'atlas')
        :param k: spring layout parameter k
        :param seed: the random state for deterministic node layouts
        :param edge_width_factor: factor to multiply LD value for edge width
        :param node_size_factor: factor to multiply -log10(pvalue) for node size
        :param min_weight: minimum weight for edges
        :param with_labels: whether to show node labels
        :param export_csv: path to export degree analysis results
        :param ax1: Axes object for degree rank plot
        :param ax2: Axes object for degree histogram plot
        :param ax3: Axes object for network plot
        :return: DataFrame of QTL pairs in NetworkX format and NetworkX graph object
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

        # create graph object - directly from edge_df
        G = nx.from_pandas_edgelist(self.edge_df, source='source', target='target', 
                                    edge_attr=['weight', 'trait1', 'trait2', 'lead1', 'lead2', 'pvalue1', 'pvalue2'])

        # create node to trait mapping and add pvalue attribute
        node_to_trait = {}
        node_to_pvalue = {}
        for _, row in self.blocks.iterrows():
            if row['qtl'] in G.nodes:
                G.nodes[row['qtl']]['trait'] = row['trait']
                G.nodes[row['qtl']]['pvalue'] = row['pvalue']
                node_to_trait[row['qtl']] = row['trait']
                node_to_pvalue[row['qtl']] = row['pvalue']

        # get all unique traits
        traits = self.blocks['trait'].unique()
        
        # if no custom color mapping provided, create one
        if color_map is None:
            n_traits = len(traits)
            
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

            color_map = {trait: colors[i] for i, trait in enumerate(traits)}

        # add color attribute to nodes
        for node in G.nodes:
            G.nodes[node]['color'] = color_map[G.nodes[node]['trait']]

        # degree rank plot
        self.degree_plot(G, color_map=color_map, plot_type="rank", export_csv=export_csv, ax=ax1)

        # degree histogram plot
        self.degree_plot(G, color_map=color_map, plot_type="histogram", ax=ax2)

        # set layout
        if layout == "atlas":
            # layout graphs with positions using graphviz neato
            self.atlas_plot(G, min_nodes=1, node_size_factor=node_size_factor, edge_width_factor=edge_width_factor,
                        nodes_style=nodes_style, edges_style=edges_style, with_labels=with_labels, labels_style=labels_style, ax=ax3)
        else:
            self.spring_plot(G, k=k, seed=seed, node_size_factor=node_size_factor, edge_width_factor=edge_width_factor,
                        nodes_style=nodes_style, edges_style=edges_style, with_labels=with_labels, labels_style=labels_style, ax=ax3)

        # add legend for traits
        trait_legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color_map[trait], markersize=10)
                            for trait in traits]

        # add LD weight legend
        # generate some representative LD values
        ld_values = np.linspace(self.edge_df['weight'].min(), self.edge_df['weight'].max(), num=4)
        ld_legend_elements = [plt.Line2D([0], [0], color=edges_style['edge_color'], 
                                        linewidth=ld * edge_width_factor,
                                        alpha=edges_style['alpha'])
                            for ld in ld_values]

        # add p-value legend
        p_values = np.linspace(self.blocks['pvalue'].min(), self.blocks['pvalue'].max(), num=4)
        p_legend_elements = [plt.Line2D([0], [0], marker='o', color='black', 
                                    markersize=np.sqrt(node_size_factor * (-np.log10(p)) / np.pi) * 2,
                                    linestyle='none', markerfacecolor='none')
                            for p in p_values]

        # create combined legend
        legend1 = ax3.legend(trait_legend_elements, traits, title="Trait", loc='upper left', 
                        bbox_to_anchor=(1.05, 1), frameon=False, borderaxespad=0.)
        ax3.add_artist(legend1)  # add first legend after adding it to the figure

        # add second legend
        legend2 = ax3.legend(ld_legend_elements, [f'{ld:.1f}' for ld in ld_values], 
                        title="LD value", loc='upper left', bbox_to_anchor=(1.05, 0.6), frameon=False,
                        borderaxespad=0.)
        ax3.add_artist(legend2)

        # add last legend
        ax3.legend(p_legend_elements, [f'{p:.1e}' for p in p_values], 
                title="P-value", loc='upper left', bbox_to_anchor=(1.05, 0.3), frameon=False,
                borderaxespad=0.)

        # set title and boundaries
        ax3.set_title('QTL network')
        ax3.axis('off')
