from qtlscan.qtl import QTL, QTLNetwork, QTLReport
from qtlscan.viz import Visualizer
from qtlscan.log import logger
from qtlscan.phe import phe_split, phe_merge, phe_stat

import argparse
import os
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from typing import List, Tuple, Dict, Optional


def run_scan(args):
    """Process scan subcommand."""

    def load_trait_names(names_arg: str) -> List[str]:
        if not names_arg:
            raise ValueError("--names is required to label traits.")
        candidate = os.path.expanduser(names_arg)
        names: List[str]
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
        else:
            names = [part.strip() for part in names_arg.split(",") if part.strip()]
        if not names:
            raise ValueError("No valid trait names found in --names argument.")
        return names

    def count_phe_traits(phe_path: str) -> int:
        try:
            phe_df = pd.read_csv(phe_path, sep="\t", header=None)
        except Exception as exc:
            raise ValueError(f"Unable to read phenotype file '{phe_path}' to infer trait count.") from exc
        trait_count = phe_df.shape[1] - 2
        if trait_count < 1:
            raise ValueError("Phenotype file must contain at least one trait column after the sample identifiers.")
        return trait_count

    def make_safe_name(name: str, counters: Dict[str, int]) -> str:
        safe = re.sub(r"[^\w.-]+", "_", name.strip())
        if not safe:
            safe = "trait"
        occurrence = counters.get(safe, 0)
        counters[safe] = occurrence + 1
        if occurrence:
            safe = f"{safe}_{occurrence}"
        return safe

    trait_names = load_trait_names(args.names)

    # Determine P-value thresholds (one or two values allowed)
    if not args.pvalue or len(args.pvalue) > 2:
        raise ValueError("--pvalue accepts one or two floating-point thresholds.")
    if len(args.pvalue) == 1:
        block_pvalue = gwas_pvalue = float(args.pvalue[0])
        logger.info(
            "Using P-value threshold %.3g for both GWAS filtering and block evaluation.",
            gwas_pvalue,
        )
    else:
        low, high = sorted(float(v) for v in args.pvalue)
        block_pvalue = low
        gwas_pvalue = high
        logger.info(
            "Using layered P-value thresholds: %.3g for GWAS filtering, %.3g for block evaluation.",
            gwas_pvalue,
            block_pvalue,
        )

    logger.info("Initializing QTL analysis...")
    qtl = QTL()
    qtl.check_plink()

    ann_data: Optional[pd.DataFrame] = None

    if args.phe:
        inferred_traits = count_phe_traits(args.phe)
        if len(trait_names) != inferred_traits:
            raise ValueError(
                f"--names count ({len(trait_names)}) must match the number of phenotypes in --phe ({inferred_traits})."
            )
        logger.info("Phenotype file provided. Running GEMMA to generate GWAS summary statistics...")
        summary_files = list(qtl.run_gemma(args.vcf, args.phe, args.out_dir, args.out_name))
    else:
        summary_files = list(args.summary)
        if len(trait_names) != len(summary_files):
            raise ValueError(
                f"--names count ({len(trait_names)}) must match the number of summary files provided ({len(summary_files)})."
            )

    safe_trait_names: List[str] = []
    counters: Dict[str, int] = {}
    for trait in trait_names:
        safe_trait_names.append(make_safe_name(trait, counters))

    logger.info("Trait naming order: %s", ", ".join(trait_names))

    # Load GFF file (if provided)
    if args.gff:
        ann_data = qtl.read_gff(args.gff, main_type=args.main_type, attr_id=args.attr_id, extract_gene=True)

    # Process each summary file
    for summary_file, trait_label, safe_name in zip(summary_files, trait_names, safe_trait_names):
        logger.info(f"Processing GWAS summary file: {summary_file} (trait: {trait_label})")

        # Set output name based on the summary file name
        base_name = safe_name or os.path.splitext(os.path.basename(summary_file))[0]

        # Load GWAS summary statistics
        gwas_data = qtl.read_gwas(summary_file, pvalue_threshold=gwas_pvalue)

        # Process GWAS data
        qtl_blocks = qtl.process_gwas(
            vcf_file=args.vcf,
            gwas_data=gwas_data,
            ann_data=ann_data,
            ld_window=args.ld_window,
            ld_window_r2=args.ld_window_r2,
            ld_window_kb=args.ld_window_kb,
            min_snps=args.min_snps,
            threads=args.threads,
            out_dir=args.out_dir,
            block_pvalue=block_pvalue,
        )

        qtl_blocks = qtl_blocks.copy()
        qtl_blocks.insert(0, "trait", trait_label)

        # Save QTL blocks for each summary file
        qtl.save(qtl_blocks, out_dir=args.out_dir, out_name=str(base_name))

    logger.info("Done!")


def run_gwas(args):
    """Run GWAS workflows using GEMMA or EMMAX."""

    logger.info("Initializing GWAS analysis...")
    qtl = QTL()

    method = (args.method or "gemma").lower()
    if method == "gemma":
        runner = qtl.run_gemma
    elif method == "emmax":
        runner = qtl.run_emmax
    else:
        raise ValueError("--method must be either 'gemma' or 'emmax'.")

    summary_files = runner(args.vcf, args.phe, args.out_dir, args.out_name)

    if not summary_files:
        logger.warning("No summary files were produced by the GWAS run.")
    else:
        logger.info("Generated GWAS summary files:")
        for path in summary_files:
            logger.info("  %s", path)

    logger.info("GWAS analysis completed!")

def run_network(args):
    logger.info("Initializing QTL network analysis...")
    qtl_network = QTLNetwork()

    def load_trait_color_map(file_path: str) -> Tuple[Dict[str, str], str]:
        expanded_path = os.path.expanduser(file_path)
        if not os.path.isfile(expanded_path):
            raise FileNotFoundError(f"Trait color mapping file not found: {expanded_path}")

        mapping: Dict[str, str] = {}
        with open(expanded_path, "r", encoding="utf-8") as handle:
            for idx, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[,\s]+", line)
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid trait color mapping at line {idx}: '{raw_line.strip()}'"
                    )
                trait, color = parts[0], parts[1]
                mapping[trait] = color

        if not mapping:
            raise ValueError("Trait color mapping file is empty or contains no valid entries.")

        return mapping, expanded_path

    # Handle input mode
    if args.edge:
        logger.info(f"Loading existing edge data from {args.edge}...")
        qtl_network.load_edge_data(args.edge)
    else:
        if args.qtl is None or args.vcf is None:
            raise ValueError("--qtl and --vcf are required when --edge is not provided.")
        # Load QTL blocks if provided
        qtl_network.load_data(args.qtl, args.names)
        qtl_network.process_data(args.vcf, args.min_weight, out_dir=args.out_dir, out_name=args.out_name)

    color_map = None
    if getattr(args, "trait_colors", None):
        color_map, mapping_path = load_trait_color_map(args.trait_colors)
        
        if qtl_network.blocks is not None:
            dataset_traits = {str(trait) for trait in qtl_network.blocks["trait"].unique()}
        elif qtl_network.edge_df is not None:
            t1 = set(qtl_network.edge_df['trait1'].unique())
            t2 = set(qtl_network.edge_df['trait2'].unique())
            dataset_traits = {str(t) for t in t1.union(t2) if pd.notna(t)}
        else:
            dataset_traits = set()

        missing_traits = sorted(dataset_traits - set(color_map.keys()))
        if missing_traits:
            raise ValueError(
                "Trait color mapping file is missing entries for the following trait(s): "
                + ", ".join(missing_traits)
            )
        logger.info(
            "Loaded %d custom trait color(s) from %s",
            len(color_map),
            mapping_path,
        )
    # Create figure with 2 rows:
    # Row 1: 4 small degree-related plots (rank, histogram, ntraits, pvalue)
    # Row 2: full-width network plot
    fig = plt.figure(figsize=(args.width, args.height))
    spec = fig.add_gridspec(2, 4, height_ratios=[1, 3])
    ax_rank = fig.add_subplot(spec[0, 0])
    ax_hist = fig.add_subplot(spec[0, 1])
    ax_ntraits = fig.add_subplot(spec[0, 2])
    ax_pvalue = fig.add_subplot(spec[0, 3])
    ax_network = fig.add_subplot(spec[1, :])
    
    nodes_style = {
        "alpha": args.node_alpha,
        "linewidths": args.node_linewidths
    }
    edges_style = {
        "alpha": args.edge_alpha,
        "edge_color": args.edge_color,
        "style": args.edge_style,
        "arrows": args.edge_arrows,
        "arrowsize": args.edge_arrowsize,
        "connectionstyle": args.edge_connectionstyle
    }
    labels_style = {
        "font_size": args.node_labels_size,
        "font_color": args.node_labels_color,
        "font_weight": args.label_font_weight,
        "font_family": args.label_font_family
    }

    _, _, node_metrics, cluster_data = qtl_network.plot(layout=args.layout,
                     k=args.k, seed=args.seed,
                     node_size_factor=args.node_size_factor,
                     edge_width_factor=args.edge_width_factor,
                     with_labels=args.with_labels,
                     color_map=color_map,
                     nodes_style=nodes_style,
                     edges_style=edges_style,
                     labels_style=labels_style,
                     ax_rank=ax_rank, ax_hist=ax_hist, ax_ntraits=ax_ntraits, ax_pvalue=ax_pvalue, ax_network=ax_network)

    # Save degree metrics
    if node_metrics is not None:
        degree_path = os.path.join(args.out_dir, f"{args.out_name}.degree.csv")
        node_metrics.to_csv(degree_path, index=False)
        logger.info(f"Degree metrics saved to {degree_path}")

    # Save cluster data
    if cluster_data:
        cluster_path = os.path.join(args.out_dir, f"{args.out_name}.cluster.csv")
        pd.DataFrame(cluster_data).to_csv(cluster_path, index=False)
        logger.info(f"Cluster layout data saved to {cluster_path}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()

    if not args.edge:
        qtl_network.save(out_dir=args.out_dir, out_name=str(args.out_name))
    logger.info("Done!")

def run_report(args):
    logger.info("Initializing QTL report generation...")
    report = QTLReport()

    def parse_names(raw: Optional[str]) -> Optional[List[str]]:
        if raw is None:
            return None
        candidate = raw.strip()
        if not candidate:
            return None
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as reader:
                names = [line.strip() for line in reader if line.strip()]
        else:
            names = [item.strip() for item in candidate.split(",") if item.strip()]
        if not names:
            return None
        return names

    def collect_block_files() -> List[str]:
        if args.qtl:
            return [os.path.expanduser(path) for path in args.qtl]
        if not args.qtl_dir:
            raise ValueError("Either --qtl or --qtl_dir must be provided.")
        root = Path(os.path.expanduser(args.qtl_dir)).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        if not root.is_dir():
            raise ValueError(f"--qtl_dir must point to a directory: {root}")
        block_paths = sorted(
            str(path.resolve())
            for path in root.rglob("*.blocks.csv")
            if path.is_file()
        )
        if not block_paths:
            raise ValueError(f"No *.blocks.csv files found in directory: {root}")
        logger.info("Discovered %d block file(s) under %s", len(block_paths), root)
        return block_paths

    block_files = collect_block_files()

    block_names = parse_names(args.names)
    if block_names and len(block_names) != len(block_files):
        raise ValueError(
            "The number of names provided must match the number of QTL block files."
        )

    output_path = report.render(
        blocks_files=block_files,
        out_dir=args.out_dir,
        out_name=args.out_name,
        blocks_names=block_names,
        top_n=args.top_n,
    )
    logger.info("Report saved to %s", output_path)
    logger.info("Done!")

def plot_manhattan(args):
    """Manhattan plot"""

    logger.info("Starting plot subcommand...")
    qtl = QTL()
    visualizer = Visualizer()

    # Parameters required: summary
    if args.summary is None:
        raise ValueError("--summary is required for plotting Manhattan plot")
    # Load summary statistics if provided
    gwas_df = qtl.read_gwas(args.summary)
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
    if args.qq:
        spec = fig.add_gridspec(1, 5)
        ax1 = fig.add_subplot(spec[0, :4])
        ax2 = fig.add_subplot(spec[0, 4])
        visualizer.plot_manhattan(gwas_df, chr_unit=args.chr_unit, chr_colors=args.chr_colors, sig_threshold=args.sig_threshold, point_size=args.point_size, ax=ax1)
        visualizer.plot_qq(gwas_df, point_size=args.point_size, ax=ax2)
    else:
        ax = fig.add_subplot(111)
        visualizer.plot_manhattan(gwas_df, chr_unit=args.chr_unit, chr_colors=args.chr_colors, sig_threshold=args.sig_threshold, point_size=args.point_size, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Plotting completed!")

def plot_qq(args):
    """QQ plot"""

    logger.info("Starting plot subcommand...")
    qtl = QTL()
    visualizer = Visualizer()

    # Parameters required: summary
    if args.summary is None:
        raise ValueError("--summary is required for plotting Manhattan plot" )
    # Load summary statistics if provided
    gwas_df = qtl.read_gwas(args.summary)
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(111)
    visualizer.plot_qq(gwas_df, point_size=args.point_size, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plotting completed!")

def plot_ld(args):
    """
    Plot simple LD heatmap only
    """
    logger.info("Starting plot subcommand...")
    qtl = QTL()
    visualizer = Visualizer()
    # Parameters required: ld, chr
    if args.ld is None or args.chr is None:
        raise ValueError("--ld and --chr are required for plotting LD heatmap")
    # Load LD data if provided
    ld_df = qtl.read_ld_file(args.ld)
    logger.info("Plotting LD heatmap...")
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(111)

    # Filter data by chromosome
    chrom_ld_df = ld_df[ld_df["CHR_A"] == args.chr]
    if chrom_ld_df.empty:
        logger.error(f"No LD data found for chromosome {args.chr}.")
        return

    visualizer.plot_ld_heatmap(chrom_ld_df, plot_value=args.plot_value, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plotting completed!")

def plot_ldheatmap(args):
    logger.info("Starting plot subcommand...")
    qtl = QTL()
    visualizer = Visualizer()
    # Parameters required: ld, blocks, gff
    if args.ld is None or args.blocks is None or args.gff is None or args.chr is None:
        raise ValueError("--ld, --blocks, --gff, and --chr are required for plotting LD blocks and heatmap")
    # Load LD data if provided
    ld_df = qtl.read_ld_file(args.ld)
    # Load QTL blocks if provided
    qtl_blocks = qtl.read_block_file(args.blocks)
    # Load GFF file if provided
    gff_df = qtl.read_gff(args.gff, main_type=args.main_type, attr_id=args.attr_id, extract_gene=False)
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(111)

    # Filter LD data by chromosome
    chrom_ld_df = ld_df[ld_df["CHR_A"] == args.chr]
    if chrom_ld_df.empty:
        logger.error(f"No LD data found for chromosome {args.chr}.")
        return

    # Filter QTL blocks by chromosome
    chrom_qtl_blocks = qtl_blocks[qtl_blocks["chr"] == args.chr] if qtl_blocks is not None else None
    if chrom_qtl_blocks.empty:
        logger.error(f"No QTL blocks found for chromosome {args.chr}.")
        return

    # Keep LD SNPs within QTL blocks
    chrom_qtl_blocks_snps = []
    for _, row in chrom_qtl_blocks.iterrows():
        chrom_qtl_blocks_snps.extend(row["ids"].split(";"))
    chrom_ld_df = chrom_ld_df[chrom_ld_df["SNP_A"].isin(chrom_qtl_blocks_snps) & chrom_ld_df["SNP_B"].isin(chrom_qtl_blocks_snps)]

    # Extract SNP positions and p-values from qtl blocks
    snp_positions = []
    snp_pvalues = []
    for _, row in chrom_qtl_blocks.iterrows():
        snp_positions.extend(row["bps"].split(";"))
        snp_pvalues.extend(row["pvalues"].split(";"))
    snp_positions = np.asarray(snp_positions, dtype=np.int64)
    snp_pvalues = np.asarray(snp_pvalues, dtype=np.float64)

    # Extract genes from qtl_blocks for the current chromosome
    chrom_genes = list(set(gene for genes_str in chrom_qtl_blocks["genes"].dropna() for gene in genes_str.split(";")))
    logger.info(f"Extracted {len(chrom_genes)} genes from qtl_blocks for chromosome {args.chr}: {', '.join(chrom_genes[:10])}...")

    # Plot LD heatmap
    visualizer.plot_ld_heatmap(
        ld_df=chrom_ld_df,
        blocks=chrom_qtl_blocks,
        plot_value=args.plot_value,
        cmap=args.cmap,
        ax=ax
    )

    # Create a divider for showing SNP positions and gene structure, and p-value
    ax_divider = make_axes_locatable(ax)

    # Add SNP positions
    ax_snp = ax_divider.append_axes("top", size="10%", pad="0%")
    visualizer.plot_snp_positions(snp_positions=snp_positions, ax=ax_snp)

    # Add gene structure
    if len(chrom_genes) > 0:
        ax_gene = ax_divider.append_axes("top", size="25%", pad="0%")
        feature_params = {
                'type': args.ft,
                'height': args.fh,
                'facecolor': args.fc,
                'edgecolor': args.ec,
                'zorder': args.fz,
                'label': False
        }
        track_params = {
            'height': args.th,
            'link_height': 0.1,
            'margin': 0.1,
            'fontsize': 10,
        }
        visualizer.plot_gene_structure(chrom_genes, gff_df=gff_df, ax=ax_gene, xlim=(np.min(snp_positions), np.max(snp_positions)), 
                                unit=args.unit, feature_params=feature_params, track_params=track_params,
                                marker_size=1)
        ax_gene.set_axis_off()

    # plot p-value
    ax_scatter = ax_divider.append_axes("top", size="50%", pad="15%", sharex=ax)
    visualizer.plot_pvalue(snp_positions, snp_pvalues, chrom=args.chr, ax=ax_scatter)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plotting completed!")

def plot_qtl_map(args):
    logger.info("Starting plot subcommand...")
    qtl = QTL()
    visualizer = Visualizer()

    # Parameters required: qtl, genome
    if args.qtl is None or args.genome is None:
        raise ValueError("--qtl and --genome are required for plotting QTL map")
    # Load QTL blocks if provided
    qtl_blocks = []
    for i, qtl_file in enumerate(args.qtl):
        qtl_df = qtl.read_block_file(qtl_file)
        if "trait" not in qtl_df.columns:
            base_name = os.path.splitext(os.path.basename(qtl_file))[0]
            base_name = base_name.split(".")[0]
            qtl_df = qtl_df.assign(trait=f"{base_name}_{i}")
        qtl_blocks.append(qtl_df)
    qtl_blocks = pd.concat(qtl_blocks)
    logger.info(f"Loaded {len(qtl_blocks)} QTL blocks from {len(args.qtl)} files.")
    # Load genome file if provided
    genome = qtl.read_genome_file(args.genome)
    logger.info(f"Loaded genome file with {len(genome)} chromosomes.")

    def parse_color_spec(color_arg: Optional[str]):
        if not color_arg:
            return None
        candidate = os.path.expanduser(color_arg)
        if os.path.isfile(candidate):
            try:
                color_df = pd.read_csv(candidate, sep=None, engine="python")
            except Exception as exc:
                raise ValueError(f"Unable to parse color file '{candidate}'.") from exc
            if color_df.empty:
                raise ValueError(f"Color file '{candidate}' is empty.")
            if color_df.shape[1] == 1:
                palette = color_df.iloc[:, 0].dropna().astype(str).str.strip()
                palette = [c for c in palette if c]
                if not palette:
                    raise ValueError(f"No valid colors found in '{candidate}'.")
                return palette
            trait_col = color_df.columns[0]
            color_col = color_df.columns[1]
            mapping = {}
            for trait, color in zip(color_df[trait_col], color_df[color_col]):
                trait_str = str(trait).strip()
                color_str = str(color).strip()
                if trait_str and color_str:
                    mapping[trait_str] = color_str
            if not mapping:
                raise ValueError(f"No valid trait-color pairs found in '{candidate}'.")
            return mapping
        if "," in color_arg:
            palette = [c.strip() for c in color_arg.split(",") if c.strip()]
            if not palette:
                raise ValueError("--colors must contain at least one entry when comma-separated.")
            return palette
        return color_arg.strip()

    def load_annotation_file(path: Optional[str], delimiter: Optional[str]) -> Optional[pd.DataFrame]:
        if not path:
            return None
        candidate = os.path.expanduser(path)
        if not os.path.isfile(candidate):
            raise ValueError(f"Annotation file not found: {candidate}")
        sep = delimiter
        if sep is None:
            lowered = candidate.lower()
            if lowered.endswith((".tsv", ".txt")):
                sep = "\t"
            else:
                sep = ","
        try:
            ann_df = pd.read_csv(candidate, sep=sep)
        except Exception as exc:
            raise ValueError(f"Unable to read annotation file '{candidate}'.") from exc
        required_cols = {"chr", "bp1", "bp2", "label"}
        if not required_cols.issubset(ann_df.columns):
            missing = required_cols - set(ann_df.columns)
            raise ValueError(f"Annotation file missing required columns: {sorted(missing)}")
        return ann_df

    color_spec = parse_color_spec(args.colors)
    ann_df = load_annotation_file(args.annotation, args.annotation_sep)

    legend_kwargs: Optional[Dict[str, object]] = None
    if args.legend_loc or args.legend_ncol is not None or args.legend_anchor:
        legend_kwargs = {}
        if args.legend_loc:
            legend_kwargs["loc"] = args.legend_loc
        if args.legend_ncol is not None:
            legend_kwargs["ncol"] = args.legend_ncol
        if args.legend_anchor:
            parts = [p.strip() for p in args.legend_anchor.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError("--legend-anchor must contain two comma-separated numbers, e.g., '1.02,1'.")
            try:
                legend_kwargs["bbox_to_anchor"] = (float(parts[0]), float(parts[1]))
            except ValueError as exc:
                raise ValueError("--legend-anchor values must be numeric, e.g., '1.02,1'.") from exc

    # Create figure
    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = fig.add_subplot(111)
    chrom_style = {
        "width": args.cw, 
        "margin": args.cm,
        "spacing": args.cs,
        "rounding": args.cr,
        "facecolor": args.fc,
        "edgecolor": args.ec,
        "edge_width": args.ew
    }
    
    text_style = {
        "font_family": args.font_family,
        "title_fontsize": args.title_fontsize,
        "subtitle_fontsize": args.subtitle_fontsize,
        "label_fontsize": args.label_fontsize,
        "tick_fontsize": args.tick_fontsize,
    }

    ann_style = {
        "max_width": args.annotation_max_width,
        "line_spacing": args.annotation_line_spacing,
        "arrowstyle": args.annotation_arrowstyle,
        "boxstyle": args.annotation_boxstyle,
        "facecolor": args.annotation_facecolor,
        "edgecolor": args.annotation_edgecolor,
    }

    visualizer.plot_qtl_map(
        qtl_blocks,
        genome,
        chrom_style=chrom_style,
        ann_data=ann_df,
        ann_style=ann_style,
        colors=color_spec,
        orientation=args.orientation,
        unit=args.unit,
        ax=ax,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        title=args.title,
        subtitle=args.subtitle,
        legend_title=args.legend_title,
        legend_kwargs=legend_kwargs,
        text_style=text_style
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=args.dpi, bbox_inches="tight")
    plt.close()
    logger.info("Plotting completed!")

def plot_dist(args):
    """Data distribution plot"""
    
    logger.info("Starting plot_dist subcommand...")
    visualizer = Visualizer()
    
    # Parameters required: data
    if args.data is None:
        raise ValueError("--data is required for plotting distribution")
    
    # Validate extend parameters
    if not (0.0 <= args.extend_tail <= 1.0):
        raise ValueError(f"--extend_tail must be between 0.0 and 1.0, got {args.extend_tail}")
    if not (0.0 <= args.extend_head <= 1.0):
        raise ValueError(f"--extend_head must be between 0.0 and 1.0, got {args.extend_head}")
    
    # Load data file
    try:
        df = pd.read_csv(args.data, sep='\t' if args.data.endswith('.tsv') else ',')
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Error loading data file: {e}")
    
    # Process columns parameter
    columns = None
    if args.columns:
        if os.path.isfile(args.columns):
            # Load column list from file
            with open(args.columns, "r") as f:
                columns = [line.strip() for line in f if line.strip()]
        else:
            # Parse column list
            columns = [col.strip() for col in args.columns.split(",")]
        logger.info(f"Using specified columns: {', '.join(columns[:10])}...")
    
    # Process colors parameter
    colors = None
    if args.colors:
        colors = [color.strip() for color in args.colors.split(",")]
        logger.info(f"Using specified colors: {', '.join(colors)}")
    
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(111)
    
    # Plot distribution
    visualizer.plot_dist(
        df=df,
        kind=args.kind,
        columns=columns,
        colors=colors,
        alpha=args.alpha,
        orientation=args.orientation,
        bins=args.bins,
        density=args.density,
        fit_curve=args.fit_curve,
        fit_bins=args.fit_bins,
        hist_style=args.hist_style,
        kde_style=args.kde_style,
        kde_points=args.kde_points,
        extend_tail=args.extend_tail,
        extend_head=args.extend_head,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ax=ax
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info("Plotting completed!")


def plot_qtn_map(args):
    """QTN genotype heatmap"""
    import importlib
    try:
        pysam = importlib.import_module('pysam')
    except Exception as e:
        raise RuntimeError("pysam is required for reading VCF files. Please install pysam.") from e

    logger.info("Starting qtnmap plot subcommand...")
    qtl = QTL()
    visualizer = Visualizer()

    # Parameters required: qtl, vcf
    if args.qtl is None or args.vcf is None:
        raise ValueError("--qtl and --vcf are required for plotting QTN map")

    # Load QTL blocks
    qtl_blocks = []
    for qtl_file in args.qtl:
        qtl_df = qtl.read_block_file(qtl_file)
        qtl_blocks.append(qtl_df)
    qtl_blocks = pd.concat(qtl_blocks)
    logger.info(f"Loaded {len(qtl_blocks)} QTL blocks from {len(args.qtl)} files.")

    # Output path
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.out_name}.{args.format}")

    visualizer.plot_qtn_map(
        vcf_path=args.vcf,
        qtl_blocks=qtl_blocks,
        samples_file=args.samples_file,
        by=args.by,
        sample_metric=args.sample_metric,
        variant_metric=args.variant_metric,
        figsize=(args.width, args.height),
        out_path=out_path
    )
    logger.info("Plotting completed!")


def main():
    description = """
    qtlscan: A tool for scanning QTL and visualizing the results from GWAS summary statistics.
    """

    epilog = """
    Example usage:
    qtlscan scan --summary gwas1.txt gwas2.txt --vcf genotypes.vcf --out_name my_analysis --out_dir results --pvalue 1e-5 --gff annotations.gff
    """
    __version__ = "1.0.0"

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter  # Preserve formatting
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # phe subcommand group
    phe_parser = subparsers.add_parser("phe", help="Phenotype utilities: split, merge, stat")
    phe_subparsers = phe_parser.add_subparsers(dest="phe_command", help="phe subcommands")

    # phe split
    phe_split_p = phe_subparsers.add_parser("split", help="Split multi-trait file into per-trait files")
    phe_split_p.add_argument("--input", type=str, required=True, help="Input TXT/TSV/CSV file (with header)")
    phe_split_p.add_argument("--sep", type=str, default="auto", help="Input separator: auto/csv/tsv/tab/','/'\t'")
    phe_split_p.add_argument("--sample-col", type=str, default=None, help="Sample ID column name (default: first column)")
    phe_split_p.add_argument("--traits", type=str, default=None, help="Regex to select trait columns (default: all non-sample columns)")
    phe_split_p.add_argument("--out-format", type=str, default="tsv", choices=["txt", "tsv", "csv", "plink"], help="Output format for per-trait files")
    phe_split_p.add_argument("--out-dir", type=str, default=".", help="Output directory")
    phe_split_p.add_argument("--prefix", type=str, default=None, help="Output file prefix (default: input basename)")
    phe_split_p.add_argument("--out-sample-col", type=str, default=None, help="Rename sample column in outputs (non-PLINK)")
    phe_split_p.add_argument("--header", type=int, default=1, help="Whether output contains header (1) or not (0) for non-PLINK")
    phe_split_p.add_argument("--encoding", type=str, default="utf-8", help="File encoding")
    phe_split_p.set_defaults(func=phe_split)

    # phe merge
    phe_merge_p = phe_subparsers.add_parser("merge", help="Merge single-trait files on sample ID into one multi-trait table")
    phe_merge_p.add_argument("--files", type=str, nargs='+', required=True, help="Input single-trait files (TXT/TSV/CSV/PLINK)")
    phe_merge_p.add_argument("--trait-names", type=str, default=None, help="Comma-separated new trait names (default: from inputs)")
    phe_merge_p.add_argument("--sample-col", type=str, default=None, help="Sample ID column name for TXT/TSV/CSV inputs (default: first column)")
    phe_merge_p.add_argument("--out-format", type=str, default="tsv", choices=["txt", "tsv", "csv"], help="Output format (non-PLINK)")
    phe_merge_p.add_argument("--out-dir", type=str, default=".", help="Output directory")
    phe_merge_p.add_argument("--out-name", type=str, default="phe_merged", help="Output file name (without extension)")
    phe_merge_p.add_argument("--out-sample-col", type=str, default=None, help="Rename output sample column (default: sample)")
    phe_merge_p.add_argument("--encoding", type=str, default="utf-8", help="File encoding")
    phe_merge_p.set_defaults(func=phe_merge)

    # phe stat
    phe_stat_p = phe_subparsers.add_parser("stat", help="Compute phenotype statistics and plot distribution")
    phe_stat_p.add_argument("--input", type=str, required=True, help="Input phenotype file (TXT/TSV/CSV/PLINK)")
    phe_stat_p.add_argument("--format-in", type=str, default="auto", choices=["auto", "table", "plink"], help="Force input format (default: auto)")
    phe_stat_p.add_argument("--sep", type=str, default="auto", help="Input separator for table formats")
    phe_stat_p.add_argument("--sample-col", type=str, default=None, help="Sample ID column name for table input (default: first column)")
    phe_stat_p.add_argument("--columns", type=str, default=None, help="Regex to select traits (default: all non-sample columns)")
    phe_stat_p.add_argument("--out-dir", type=str, default=".", help="Output directory")
    phe_stat_p.add_argument("--out-name", type=str, default="phe_stat", help="Output name prefix")
    phe_stat_p.add_argument("--stats-file", type=str, default=None, help="Custom path for stats table (default: <out_dir>/<out_name>.stats.tsv)")
    phe_stat_p.add_argument("--encoding", type=str, default="utf-8", help="File encoding")
    # plotting options
    phe_stat_p.add_argument("--plot-kind", type=str, default="hist", choices=["box", "violin", "hist", "kde"], help="Plot kind for distribution figure")
    phe_stat_p.add_argument("--bins", type=int, default=30, help="Histogram bins")
    phe_stat_p.add_argument("--density", action="store_true", help="Histogram as density")
    phe_stat_p.add_argument("--fit-curve", action="store_true", help="Add fitted curve for histogram")
    phe_stat_p.add_argument("--fit-bins", type=int, default=100, help="Bins for fitted curve")
    phe_stat_p.add_argument("--hist-style", type=str, default="dodge", choices=["stack", "dodge", "side"], help="Histogram layout for multi-columns")
    phe_stat_p.add_argument("--kde-style", type=str, default="stack", choices=["stack", "side"], help="KDE layout for multi-columns")
    phe_stat_p.add_argument("--kde-points", type=int, default=200, help="KDE points")
    phe_stat_p.add_argument("--extend-tail", type=float, default=0.0, help="Extend lower bound (0-1) for KDE")
    phe_stat_p.add_argument("--extend-head", type=float, default=0.0, help="Extend upper bound (0-1) for KDE")
    phe_stat_p.add_argument("--orientation", type=str, default="vertical", choices=["vertical", "horizontal"], help="Plot orientation")
    phe_stat_p.add_argument("--colors", type=str, nargs="+", help="Colors for plots")
    phe_stat_p.add_argument("--alpha", type=float, default=0.7, help="Transparency for plots")
    phe_stat_p.add_argument("--xlabel", type=str, default=None, help="X label")
    phe_stat_p.add_argument("--ylabel", type=str, default=None, help="Y label")
    phe_stat_p.add_argument("--width", type=float, default=8, help="Figure width")
    phe_stat_p.add_argument("--height", type=float, default=6, help="Figure height")
    phe_stat_p.add_argument("--format", type=str, default="png", help="Figure format: png/pdf/svg")
    phe_stat_p.set_defaults(func=phe_stat)

    # scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Analyze GWAS QTL results")
    scan_group = scan_parser.add_mutually_exclusive_group(required=True)
    scan_group.add_argument("--summary", type=str, nargs="+", help="Path(s) to GWAS summary statistics txt file(s). multiple files with space separated are allowed. Required columns: chr, rs, p_wald and ps")
    scan_group.add_argument("--phe", type=str, help="Path to phenotype file (no header, tab-separated, used for GWAS analysis with no summary statistics)")
    scan_parser.add_argument("--vcf", type=str, required=True, help="Path to VCF genotype file")
    scan_parser.add_argument("--out_name", type=str, default="gwas", help="Output file name prefix (default: %(default)s)")
    scan_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    scan_parser.add_argument("--min_snps", type=int, default=3, help="Minimum number of SNPs per QTL block (default: %(default)s)")
    scan_parser.add_argument("--ld_window", type=int, default=9999999, help="Maximum number of SNPs in LD window (default: %(default)s)")
    scan_parser.add_argument("--ld_window_kb", type=int, default=100, help="Maximum number of kb in LD window (default: %(default)s)")
    scan_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD threshold (R^2) for LD window (default: %(default)s)")
    scan_parser.add_argument("--threads", type=int, default=cpu_count(), help="Number of threads for PLINK (default: %(default)s)")
    scan_parser.add_argument(
        "--pvalue",
        type=float,
        nargs="+",
        default=[1.0],
        metavar="P",
        help=(
            "One or two P-value thresholds. If two are provided, the larger value filters GWAS input "
            "and the smaller value evaluates core SNP counts within QTL blocks."
        ),
    )
    scan_parser.add_argument("--gff", type=str, help="Path to GFF file for gene annotation")
    scan_parser.add_argument("--main_type", type=str, default="mRNA", help="Main feature type to extract from GFF file (default: %(default)s)")
    scan_parser.add_argument("--attr_id", type=str, default="ID", help="Attribute ID to extract gene names from GFF file (default: %(default)s)")
    scan_parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Trait names in output order (comma-separated string or path to a newline-delimited file); count must match summaries or phenotypes",
    )
    scan_parser.set_defaults(func=run_scan)

    # gwas subcommand
    gwas_parser = subparsers.add_parser("gwas", help="Run GWAS using GEMMA or EMMAX")
    gwas_parser.add_argument("--vcf", type=str, required=True, help="Path to VCF genotype file")
    gwas_parser.add_argument(
        "--phe",
        type=str,
        required=True,
        help="Phenotype file (tab-separated, no header, first two columns are sample identifiers)",
    )
    gwas_parser.add_argument(
        "--method",
        type=str,
        default="gemma",
        choices=["gemma", "emmax"],
        help="GWAS engine to use (default: %(default)s)",
    )
    gwas_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    gwas_parser.add_argument("--out_name", type=str, default="gwas", help="Output file name prefix (default: %(default)s)")
    gwas_parser.set_defaults(func=run_gwas)

    # QTL Network subcommand
    qtl_network_parser = subparsers.add_parser("network", help="Generate QTL network")
    
    input_group = qtl_network_parser.add_argument_group("Input Options")
    input_ex_group = input_group.add_mutually_exclusive_group(required=True)
    input_ex_group.add_argument("--qtl", type=str, nargs="+", help="Path(s) to QTL blocks csv file(s). Multiple files with space separated are allowed.")
    input_ex_group.add_argument("--edge", type=str, help="Path to existing edge dataframe file (skips LD calculation).")
    input_group.add_argument("--vcf", type=str, help="Path to VCF genotype file (required if --qtl is used)")
    input_group.add_argument("--names", type=str, nargs="+", help="Rename different QTL blocks file(s) for visualization")
    input_group.add_argument(
        "--trait_colors",
        type=str,
        help="Path to a trait\u2013color mapping file (trait and color separated by comma/tab/space per line)"
    )

    layout_group = qtl_network_parser.add_argument_group("Layout Options")
    layout_group.add_argument("--layout", type=str, default="cluster", choices=['spring', 'kamada_kawai', 'fruchterman_reingold', 'forceatlas2'],
                                    help="Layout for QTL network (default: %(default)s)")
    layout_group.add_argument("--k", type=float, default=0.3, help="Spring layout parameter k (default: %(default)s)")
    layout_group.add_argument("--seed", type=int, default=42, help="Random seed for deterministic node layouts (default: %(default)s)")
    layout_group.add_argument("--min_weight", type=float, default=0, help="Minimum weight for edges (default: %(default)s)")

    node_group = qtl_network_parser.add_argument_group("Node Styling")
    node_group.add_argument("--node_size_factor", type=float, default=5, help="Factor to multiply -log10(pvalue) for node size (default: %(default)s)")
    node_group.add_argument("--node_alpha", type=float, default=0.8, help="Node transparency (default: %(default)s)")
    node_group.add_argument("--node_linewidths", type=float, default=1.0, help="Node border width (default: %(default)s)")
    
    edge_group = qtl_network_parser.add_argument_group("Edge Styling")
    edge_group.add_argument("--edge_width_factor", type=float, default=3, help="Factor to multiply LD value for edge width (default: %(default)s)")
    edge_group.add_argument("--edge_alpha", type=float, default=0.5, help="Edge transparency (default: %(default)s)")
    edge_group.add_argument("--edge_color", type=str, default="gray", help="Edge color (default: %(default)s)")
    edge_group.add_argument("--edge_style", type=str, default="solid", choices=['solid', 'dashed', 'dotted', 'dashdot'], help="Edge line style (default: %(default)s)")
    edge_group.add_argument("--edge_arrows", action="store_true", default=True, help="Show edge arrows (default: %(default)s)")
    edge_group.add_argument("--no_edge_arrows", action="store_false", dest="edge_arrows", help="Hide edge arrows")
    edge_group.add_argument("--edge_arrowsize", type=int, default=10, help="Edge arrow size (default: %(default)s)")
    edge_group.add_argument("--edge_connectionstyle", type=str, default="arc3,rad=0.2", help="Edge connection style (default: %(default)s)")

    label_group = qtl_network_parser.add_argument_group("Label Styling")
    label_group.add_argument("--with_labels", action="store_true", help="Whether to show node labels")
    label_group.add_argument("--node_labels_size", type=float, default=6, help="Node label font size (default: %(default)s)")
    label_group.add_argument("--node_labels_color", type=str, default="white", help="Node label color (default: %(default)s)")
    label_group.add_argument("--label_font_weight", type=str, default="normal", help="Label font weight (default: %(default)s)")
    label_group.add_argument("--label_font_family", type=str, default="sans-serif", help="Label font family (default: %(default)s)")

    output_group = qtl_network_parser.add_argument_group("Output Options")
    output_group.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    output_group.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
    output_group.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    output_group.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    output_group.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")

    qtl_network_parser.set_defaults(func=run_network)

    # report subcommand
    report_parser = subparsers.add_parser("report", help="Generate an interactive HTML summary for QTL blocks")
    report_input_group = report_parser.add_mutually_exclusive_group(required=True)
    report_input_group.add_argument(
        "--qtl",
        type=str,
        nargs="+",
        help="Path(s) to QTL blocks csv file(s)"
    )
    report_input_group.add_argument(
        "--qtl_dir",
        type=str,
        help="Directory containing one or more *.blocks.csv files"
    )
    report_parser.add_argument(
        "--names",
        type=str,
        help="Optional comma-separated names or newline-delimited file to rename each QTL dataset",
    )
    report_parser.add_argument("--top_n", type=int, default=20, help="Number of most significant blocks to highlight")
    report_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    report_parser.add_argument("--out_name", type=str, default="qtl_report", help="Output HTML file name prefix (default: %(default)s)")
    report_parser.set_defaults(func=run_report)

    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Visualize QTL results")
    plot_subparsers = plot_parser.add_subparsers(dest="plot_type", help="Plot types")

    # Group for Manhattan and QQ plots
    manhattan_parser = plot_subparsers.add_parser("manhattan", help="Generate Manhattan and QQ plots")
    manhattan_parser.add_argument("--summary", type=str, required=True, help="Path to summary statistics txt file")
    manhattan_parser.add_argument("--chr_unit", type=str, default="mb", help="Unit for x-axis (default: %(default)s)")
    manhattan_parser.add_argument("--chr_colors", type=str, nargs="+", help="Colors for chromosomes")
    manhattan_parser.add_argument("--sig_threshold", type=float, help="Significance p-value threshold for line plot")
    manhattan_parser.add_argument("--point_size", type=float, default=5, help="Point size for Manhattan plot (default: %(default)s)")
    manhattan_parser.add_argument("--qq", action="store_true", help="Whether to plot QQ plot (default: %(default)s)")
    manhattan_parser.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    manhattan_parser.add_argument("--height", type=float, default=3, help="Figure height (default: %(default)s)")
    manhattan_parser.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    manhattan_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    manhattan_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    manhattan_parser.set_defaults(plot_func=plot_manhattan)

    # Group for LD Heatmap
    ldheatmap_parser = plot_subparsers.add_parser("ldheatmap", help="Generate LD heatmap")
    ldheatmap_parser.add_argument("--ld", type=str, required=True, help="Path to LD information txt file")
    ldheatmap_parser.add_argument("--blocks", type=str, required=True, help="Path to QTL blocks csv file")
    ldheatmap_parser.add_argument("--chr", type=str, required=True, help="Target chromosome for plotting")
    ldheatmap_parser.add_argument("--gff", type=str, required=True, help="Path to GFF file for gene annotation")
    ldheatmap_parser.add_argument("--main_type", type=str, default="mRNA", help="Main feature type to extract from GFF file (default: %(default)s)")
    ldheatmap_parser.add_argument("--attr_id", type=str, default="ID", help="Attribute ID to extract gene names from GFF file (default: %(default)s)")
    ldheatmap_parser.add_argument("--unit", type=str, default="mb", help="Unit for x-axis (default: %(default)s)")
    ldheatmap_parser.add_argument("--ft", type=str, nargs='+', default=['exon', 'five_prime_UTR', 'three_prime_UTR'], help="Feature type to plot (default: %(default)s)")
    ldheatmap_parser.add_argument("--fc", type=str, nargs='+', default=['#38638D', 'gray', 'gray'], help="Face color for features (default: %(default)s)")
    ldheatmap_parser.add_argument("--ec", type=str, nargs='+', default=[None, None, None], help="Edge color for features (default: %(default)s)")
    ldheatmap_parser.add_argument("--fh", type=float, nargs='+', default=[0.2, 0.2, 0.2], help="Feature height to plot (default: %(default)s)")
    ldheatmap_parser.add_argument("--fz", type=int, nargs='+', default=[1, 1, 1], help="Feature zorder to plot (default: %(default)s)")
    ldheatmap_parser.add_argument("--th", type=float, default=0.5, help="Track height to plot (default: %(default)s)")
    ldheatmap_parser.add_argument("--plot_value", action="store_true", help="Whether to plot values in LD heatmap (default: %(default)s)")
    ldheatmap_parser.add_argument("--cmap", type=str, default="Reds", help="Colormap for LD heatmap (default: %(default)s)")
    ldheatmap_parser.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    ldheatmap_parser.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
    ldheatmap_parser.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    ldheatmap_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    ldheatmap_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    ldheatmap_parser.set_defaults(plot_func=plot_ldheatmap)

    # Group for QTL Map
    qtlmap_parser = plot_subparsers.add_parser("qtlmap", help="Generate QTL map")
    qtlmap_parser.add_argument("--qtl", type=str, nargs="+", required=True, help="Path(s) to QTL blocks csv file(s). Multiple files with space separated are allowed.")
    qtlmap_parser.add_argument("--genome", type=str, required=True, help="Path to genome txt file contain chromosome name and length information")
    qtlmap_parser.add_argument("--cw", type=float, default=0.8, help="Chromosome width (default: %(default)s)")
    qtlmap_parser.add_argument("--cm", type=float, default=0.2, help="Chromosome margin (default: %(default)s)")
    qtlmap_parser.add_argument("--cs", type=float, default=0.5, help="Chromosome spacing (default: %(default)s)")
    qtlmap_parser.add_argument("--cr", type=float, default=0.3, help="Chromosome rounding (default: %(default)s)")
    qtlmap_parser.add_argument("--fc", type=str, default="lightgrey", help="Chromosome facecolor (default: %(default)s)")
    qtlmap_parser.add_argument("--ec", type=str, default="black", help="Chromosome edgecolor (default: %(default)s)")
    qtlmap_parser.add_argument("--ew", type=float, default=0.8, help="Chromosome edgewidth (default: %(default)s)")
    qtlmap_parser.add_argument("--orientation", type=str, default="vertical", choices=["vertical", "horizontal"], help="Layout orientation for chromosomes (default: %(default)s)")
    qtlmap_parser.add_argument("--unit", type=str, default="mb", choices=["mb", "kb", "bp"], help="Unit for displaying genomic positions (default: %(default)s)")
    qtlmap_parser.add_argument("--colors", type=str, help="Trait color specification: comma-separated list, matplotlib colormap name, or path to trait-color table")
    qtlmap_parser.add_argument("--title", type=str, help="Figure title")
    qtlmap_parser.add_argument("--subtitle", type=str, help="Optional subtitle placed beneath the title")
    qtlmap_parser.add_argument("--legend-title", type=str, default="Traits", help="Legend title (default: %(default)s)")
    qtlmap_parser.add_argument("--legend-loc", type=str, help="Legend location (e.g., upper left)")
    qtlmap_parser.add_argument("--legend-ncol", type=int, help="Number of legend columns")
    qtlmap_parser.add_argument("--legend-anchor", type=str, help="Legend bbox_to_anchor as 'x,y' (optional)")
    qtlmap_parser.add_argument("--font-family", type=str, default="Arial", help="Font family applied to the plot (default: %(default)s)")
    qtlmap_parser.add_argument("--title-fontsize", type=int, default=18, help="Font size for title (default: %(default)s)")
    qtlmap_parser.add_argument("--subtitle-fontsize", type=int, default=12, help="Font size for subtitle (default: %(default)s)")
    qtlmap_parser.add_argument("--label-fontsize", type=int, default=12, help="Font size for axis labels (default: %(default)s)")
    qtlmap_parser.add_argument("--tick-fontsize", type=int, default=10, help="Font size for tick labels (default: %(default)s)")
    qtlmap_parser.add_argument("--annotation", type=str, help="Optional annotation file with columns chr,bp1,bp2,label")
    qtlmap_parser.add_argument("--annotation-sep", type=str, help="Delimiter for annotation file (default: infer from extension)")
    qtlmap_parser.add_argument("--annotation-max-width", type=int, default=28, help="Maximum characters per annotation line before wrapping (default: %(default)s)")
    qtlmap_parser.add_argument("--annotation-line-spacing", type=float, default=1.2, help="Line spacing multiplier for annotation text (default: %(default)s)")
    qtlmap_parser.add_argument("--annotation-arrowstyle", type=str, default="-", help="Arrow style for annotations (default: %(default)s)")
    qtlmap_parser.add_argument("--annotation-boxstyle", type=str, default="round,pad=0.2", help="Box style for annotation text (default: %(default)s)")
    qtlmap_parser.add_argument("--annotation-facecolor", type=str, default="white", help="Face color for annotation boxes (default: %(default)s)")
    qtlmap_parser.add_argument("--annotation-edgecolor", type=str, default="none", help="Edge color for annotation boxes (default: %(default)s)")
    qtlmap_parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: %(default)s)")
    qtlmap_parser.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    qtlmap_parser.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
    qtlmap_parser.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    qtlmap_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    qtlmap_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    qtlmap_parser.set_defaults(plot_func=plot_qtl_map)

    # Group for QTN Map
    qtnmap_parser = plot_subparsers.add_parser("qtnmap", help="Generate QTN genotype heatmap")
    qtnmap_parser.add_argument("--qtl", type=str, nargs="+", required=True, help="Path(s) to QTL blocks csv file(s)")
    qtnmap_parser.add_argument("--vcf", type=str, required=True, help="Path to VCF genotype file")
    qtnmap_parser.add_argument("--samples_file", type=str, help="Path to samples file (2 columns: sample, group)")
    qtnmap_parser.add_argument("--by", type=str, default="trait", choices=["trait", "chr"], help="Group variants by trait or chr (default: %(default)s)")
    qtnmap_parser.add_argument("--sample_metric", type=str, default="count", choices=["mis", "het", "count", "count0", "count1", "count2"], help="Sample metric (default: %(default)s)")
    qtnmap_parser.add_argument("--variant_metric", type=str, default="maf", choices=["mis", "het", "maf", "count", "count0", "count1", "count2"], help="Variant metric (default: %(default)s)")
    qtnmap_parser.add_argument("--width", type=float, default=12, help="Figure width (default: %(default)s)")
    qtnmap_parser.add_argument("--height", type=float, default=10, help="Figure height (default: %(default)s)")
    qtnmap_parser.add_argument("--format", type=str, default="png", help="Output format (default: %(default)s)")
    qtnmap_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    qtnmap_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    qtnmap_parser.set_defaults(plot_func=plot_qtn_map)

    # Group for Distribution Plot
    dist_parser = plot_subparsers.add_parser("dist", help="Generate data distribution plots")
    dist_parser.add_argument("--data", type=str, required=True, help="Path to data file (CSV/TSV format)")
    dist_parser.add_argument("--kind", type=str, default="box", choices=["box", "violin", "hist", "kde"], 
                           help="Kind of distribution plot (default: %(default)s)")
    dist_parser.add_argument("--columns", type=str, help="Comma-separated column names or path to file containing column names. If not provided, all numeric columns will be used")
    dist_parser.add_argument("--colors", type=str, help="Comma-separated color list for each column")
    dist_parser.add_argument("--alpha", type=float, default=0.7, help="Transparency level (0-1) (default: %(default)s)")
    dist_parser.add_argument("--orientation", type=str, default="vertical", choices=["vertical", "horizontal"], 
                           help="Plot orientation (default: %(default)s)")
    dist_parser.add_argument("--bins", type=int, default=30, help="Number of bins for histogram (default: %(default)s)")
    dist_parser.add_argument("--density", action="store_true", help="Show density instead of count for histogram (default: False)")
    dist_parser.add_argument("--fit_curve", action="store_true", help="Add fitted curve for histogram (default: False)")
    dist_parser.add_argument("--fit_bins", type=int, default=100, help="Number of bins for fitted curve in histogram (default: %(default)s)")
    dist_parser.add_argument("--hist_style", type=str, default="dodge", choices=["stack", "dodge", "side"], 
                           help="Style for multiple columns in histogram (default: %(default)s)")
    dist_parser.add_argument("--kde_style", type=str, default="stack", choices=["stack", "side"], 
                           help="Style for multiple columns in KDE plot (default: %(default)s)")
    dist_parser.add_argument("--kde_points", type=int, default=200, 
                           help="Number of points for KDE calculation (default: %(default)s)")
    dist_parser.add_argument("--extend_tail", type=float, default=0.0, 
                           help="Extend factor for smaller values (left boundary) in kde plot, range: 0-1 (default: %(default)s)")
    dist_parser.add_argument("--extend_head", type=float, default=0.0, 
                           help="Extend factor for larger values (right boundary) in kde plot, range: 0-1 (default: %(default)s)")
    dist_parser.add_argument("--xlabel", type=str, help="Custom label for x-axis")
    dist_parser.add_argument("--ylabel", type=str, help="Custom label for y-axis")
    dist_parser.add_argument("--width", type=float, default=4, help="Figure width (default: %(default)s)")
    dist_parser.add_argument("--height", type=float, default=3, help="Figure height (default: %(default)s)")
    dist_parser.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    dist_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    dist_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    dist_parser.set_defaults(plot_func=plot_dist)

    # Parse arguments and execute the corresponding subcommand
    args = parser.parse_args()
    if args.command:
        # Create output directory if it doesn't exist
        os.makedirs(args.out_dir, exist_ok=True)
        if hasattr(args, 'plot_func'):
            args.plot_func(args)
        else:
            args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
