from qtlscan.qtl import QTL, QTLNetwork
from qtlscan.viz import Visualizer
from qtlscan.log import logger
from qtlscan.igs import train as train_impl
from qtlscan.igs import predict as predict_impl
from qtlscan.phe import phe_split, phe_merge, phe_stat

import argparse
import os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from typing import List, Tuple, Dict, Optional


def run_scan(args):
    """Process GWAS subcommand."""

    logger.info("Initializing QTL analysis...")
    qtl = QTL()
    qtl.check_plink()

    # If --phe is provided, run GEMMA to generate GWAS summary files
    if args.phe:
        logger.info("Phenotype file provided. Running GEMMA to generate GWAS summary statistics...")
        summary_files = qtl.run_gemma(args.vcf, args.phe, args.out_dir, args.out_name)
    else:
        # If --summary is provided, use the summary file
        summary_files = args.summary

    # Load GFF file (if provided)
    if args.gff:
        ann_data = qtl.read_gff(args.gff, main_type=args.main_type, attr_id=args.attr_id, extract_gene=True)

    # Process each summary file
    for i, summary_file in enumerate(summary_files):
        logger.info(f"Processing GWAS summary file: {summary_file}")

        # Set output name based on the summary file name
        base_name = os.path.splitext(os.path.basename(summary_file))[0]
        out_name = f"{args.out_name}_{i}_{base_name}"

        # Load GWAS summary statistics
        gwas_data = qtl.read_gwas(summary_file, pvalue_threshold=args.pvalue)

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
            out_dir=args.out_dir
        )

        # Save QTL blocks for each summary file
        qtl.save(qtl_blocks, out_dir=args.out_dir, out_name=str(out_name))

    logger.info("Done!")

def run_qtlnetwork(args):
    logger.info("Initializing QTL network analysis...")
    qtl_network = QTLNetwork()

    # Parameters required: qtl, vcf
    if args.qtl is None or args.vcf is None:
        raise ValueError("--qtl and --vcf are required for plotting QTL network")
    # Load QTL blocks if provided
    qtl_network.load_data(args.qtl, args.names)
    qtl_network.process_data(args.vcf, args.min_weight)
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
    spec = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[1:, :])
    qtl_network.plot(layout=args.layout, 
                    k=args.k, seed=args.seed,
                    node_size_factor=args.node_size_factor, 
                    edge_width_factor=args.edge_width_factor,
                    with_labels=args.with_labels, 
                    export_csv=os.path.join(args.out_dir, f"{args.out_name}.degree.csv"),
                    ax1=ax1, ax2=ax2, ax3=ax3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
    plt.close()

    qtl_network.save(out_dir=args.out_dir, out_name=str(args.out_name))
    logger.info("Done!")

def run_train(args):
    """Train genotype-to-phenotype model(s) and generate report."""
    logger.info("Starting train subcommand...")
    train_impl(args)
    logger.info("Training pipeline completed!")

def run_predict(args):
    """Apply a trained model bundle to new feature table for inference-only."""
    logger.info("Starting predict subcommand...")
    predict_impl(args)
    logger.info("Prediction completed!")

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

def plot_genotype(args):
    """Genotype heatmap and stats from VCF"""
    import importlib
    try:
        pysam = importlib.import_module('pysam')
    except Exception as e:
        raise RuntimeError("pysam 未安装或不可用；该子命令需要 pysam 来读取 VCF。请安装 pysam 后重试。") from e

    logger.info("Starting genotype plot subcommand...")
    # Validate inputs
    if args.vcf is None:
        raise ValueError("--vcf is required for plotting genotype heatmap")

    vcf_path = args.vcf
    if not os.path.isfile(vcf_path):
        raise ValueError(f"VCF not found: {vcf_path}")

    # Figure format
    image_format = (args.format or "png").lower()
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("--format only supports png/pdf/svg")

    # ----- Helper functions (borrowed and adapted from stat_vcf, self-contained) -----
    def get_sample_names(vcf_path: str) -> List[str]:
        with pysam.VariantFile(vcf_path) as vf:
            return list(vf.header.samples)

    def parse_regions(region_args: Optional[List[str]]) -> Optional[Dict[str, List[Tuple[int, int]]]]:
        if not region_args:
            return None
        regions: Dict[str, List[Tuple[int, int]]] = {}
        for item in region_args:
            s = item.strip()
            if ":" not in s:
                chrom = s
                start, end = 1, 2**31 - 1
            else:
                chrom, span = s.split(":", 1)
                if "-" in span:
                    a, b = span.split("-", 1)
                    start = int(a) if a else 1
                    end = int(b) if b else 2**31 - 1
                else:
                    start = int(span)
                    end = start
            regions.setdefault(chrom, []).append((start, end))
        # merge
        for chrom in list(regions.keys()):
            ivals = sorted(regions[chrom])
            merged = []
            for s, e in ivals:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            regions[chrom] = [(s, e) for s, e in merged]
        return regions

    def parse_regions_file(path: Optional[str]) -> Optional[Dict[str, List[Tuple[int, int]]]]:
        if not path:
            return None
        regions: Dict[str, List[Tuple[int, int]]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start0 = int(parts[1])
                    end1 = int(parts[2])
                except ValueError:
                    continue
                start = start0 + 1
                end = end1
                if start > end:
                    continue
                regions.setdefault(chrom, []).append((start, end))
        for chrom in list(regions.keys()):
            ivals = sorted(regions[chrom])
            merged: List[Tuple[int, int]] = []
            for s, e in ivals:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            regions[chrom] = [(s, e) for s, e in merged]
        return regions

    def is_indel_site(ref: str, alts: Optional[Tuple[str, ...]]) -> Optional[bool]:
        if not alts:
            return None
        for a in alts:
            if a == "*" or a.startswith("<"):
                return True
        if any(len(a) != len(ref) for a in alts):
            return True
        if len(ref) == 1 and all(len(a) == 1 for a in alts):
            return False
        return True

    def encode_gt(gt: Optional[object]) -> Tuple[int, bool]:
        if gt is None:
            return -1, False
        if isinstance(gt, (tuple, list)):
            if any(a is None for a in gt):
                return -1, False
            alle = [str(a) for a in gt]
        else:
            s = str(gt)
            if "." in s:
                return -1, False
            alle = s.replace("|", "/").split("/")
        if len(alle) == 0:
            return -1, False
        nonref = any(a != "0" for a in alle)
        if all(a == "0" for a in alle):
            return 0, False
        if len(set(alle)) == 1 and alle[0] != "0":
            return 2, True
        return 1, True

    def in_regions(chrom: str, pos: int, regions: Optional[Dict[str, List[Tuple[int, int]]]]) -> bool:
        if regions is None:
            return True
        if chrom not in regions:
            return False
        for s, e in regions[chrom]:
            if s <= pos <= e:
                return True
        return False

    def iterate_records(vcf_path: str, regions: Optional[Dict[str, List[Tuple[int, int]]]]):
        vf = pysam.VariantFile(vcf_path)
        if not regions:
            for rec in vf:
                yield rec
            return
        try:
            for chrom, ivals in regions.items():
                for s, e in ivals:
                    start0 = max(0, s - 1)
                    for rec in vf.fetch(chrom, start0, e):
                        yield rec
        except (ValueError, OSError):
            vf.close()
            with pysam.VariantFile(vcf_path) as vf2:
                for rec in vf2:
                    if in_regions(rec.chrom, rec.pos, regions):
                        yield rec

    def reservoir_pick_variant_indices(vcf_path: str, regions: Optional[Dict[str, List[Tuple[int, int]]]], k: int, seed: int) -> List[int]:
        rnd = random.Random(seed)
        picked: List[int] = []
        seen = 0
        for _rec in iterate_records(vcf_path, regions):
            if seen < k:
                picked.append(seen)
            else:
                j = rnd.randrange(seen + 1)
                if j < k:
                    picked[j] = seen
            seen += 1
        return sorted(picked)

    def select_samples(all_samples: List[str], names: Optional[List[str]], name_file: Optional[str], random_n: Optional[int], seed: int) -> Tuple[List[str], List[int]]:
        target_names: List[str] = []
        if names:
            for n in names:
                n2 = n.strip()
                if n2:
                    target_names.append(n2)
        if name_file:
            with open(name_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        target_names.append(s)
        if target_names:
            name2idx = {n: i for i, n in enumerate(all_samples)}
            missing = [n for n in target_names if n not in name2idx]
            if missing:
                raise RuntimeError(f"Samples not found: {','.join(missing)}")
            idxs = [name2idx[n] for n in target_names]
            return [all_samples[i] for i in idxs], idxs
        if random_n is not None:
            if random_n <= 0:
                raise RuntimeError("--random-samples must be >0")
            if random_n > len(all_samples):
                raise RuntimeError("Random sample count exceeds available samples in VCF")
            rnd = random.Random(seed)
            idxs = sorted(rnd.sample(range(len(all_samples)), random_n))
            return [all_samples[i] for i in idxs], idxs
        idxs = list(range(len(all_samples)))
        return all_samples[:], idxs

    def parse_groups_file(path: Optional[str], selected_names: List[str]) -> Dict[str, str]:
        if not path:
            return {}
        mp: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                sample, group = parts[0], parts[1]
                mp[sample] = group
        return {s: mp.get(s, "Ungrouped") for s in selected_names}

    def make_random_groups(selected_names: List[str], k: int, seed: int) -> Dict[str, str]:
        k = max(1, min(k, len(selected_names)))
        rnd = random.Random(seed)
        arr = selected_names[:]
        rnd.shuffle(arr)
        n = len(arr)
        base = n // k
        extra = n % k
        mp: Dict[str, str] = {}
        idx = 0
        for i in range(k):
            sz = base + (1 if i < extra else 0)
            gname = f"G{i+1}"
            for _ in range(sz):
                mp[arr[idx]] = gname
                idx += 1
        return mp

    # ----- Prepare inputs -----
    all_samples = get_sample_names(vcf_path)
    samples_arg = [s for s in args.samples.split(",")] if args.samples else None
    selected_names, _selected_idx = select_samples(
        all_samples, samples_arg, args.samples_file, args.random_samples, args.seed
    )
    logger.info(f"Selected {len(selected_names)} samples for plotting")

    regions_a = parse_regions(args.regions)
    regions_b = parse_regions_file(args.regions_file)
    if regions_a and regions_b:
        merged: Dict[str, List[Tuple[int, int]]] = {}
        for src in (regions_a, regions_b):
            for chrom, ivals in src.items():
                merged.setdefault(chrom, []).extend(ivals)
        for chrom in list(merged.keys()):
            ivals = sorted(merged[chrom])
            merged2: List[Tuple[int, int]] = []
            for s, e in ivals:
                if not merged2 or s > merged2[-1][1] + 1:
                    merged2.append([s, e])
                else:
                    merged2[-1][1] = max(merged2[-1][1], e)
            merged[chrom] = [(s, e) for s, e in merged2]
        regions = merged
    else:
        regions = regions_a or regions_b

    picked_local_indices = None
    if args.random_variants is not None:
        if args.random_variants <= 0:
            raise ValueError("--random-variants must be >0")
        picked_local_indices = reservoir_pick_variant_indices(vcf_path, regions, args.random_variants, args.seed)

    if args.groups_file:
        group_map = parse_groups_file(args.groups_file, selected_names)
    elif args.random_groups is not None:
        if args.random_groups <= 0 or args.random_groups > len(selected_names):
            raise ValueError("--random-groups must be within [1, number_of_selected_samples]")
        group_map = make_random_groups(selected_names, args.random_groups, args.seed)
    else:
        group_map = {s: "All" for s in selected_names}

    # ----- Compute stats and build plot rows -----
    stats = [{
        "sample": name,
        "total": 0,
        "called": 0,
        "missing": 0,
        "het": 0,
        "snp": 0,
        "indel": 0,
        "code0": 0,
        "code1": 0,
        "code2": 0,
    } for name in selected_names]

    plot_rows: List[List[int]] = []
    plot_meta: List[Tuple[str, int]] = []
    plot_seen = 0
    rnd = random.Random(args.seed)
    picked_set = set(picked_local_indices) if picked_local_indices is not None else None
    region_local_counter = 0

    for rec in iterate_records(vcf_path, regions):
        chrom = rec.chrom
        pos = rec.pos
        indel_flag = is_indel_site(rec.ref, rec.alts)

        if picked_set is not None and region_local_counter not in picked_set:
            region_local_counter += 1
            continue

        row_codes: List[int] = []
        gt_list: List[Tuple[int, bool]] = []
        for name in selected_names:
            gt_raw = rec.samples[name].get("GT", None)
            code, nonref = encode_gt(gt_raw)
            row_codes.append(code)
            gt_list.append((code, nonref))

        for i in range(len(selected_names)):
            st = stats[i]
            st["total"] += 1
            code, nonref = gt_list[i]
            if code == -1:
                st["missing"] += 1
            else:
                st["called"] += 1
                if code == 1:
                    st["het"] += 1
                if code == 0:
                    st["code0"] += 1
                elif code == 1:
                    st["code1"] += 1
                elif code == 2:
                    st["code2"] += 1
                if indel_flag is not None:
                    if args.count_ref:
                        if indel_flag:
                            st["indel"] += 1
                        else:
                            st["snp"] += 1
                    else:
                        if nonref:
                            if indel_flag:
                                st["indel"] += 1
                            else:
                                st["snp"] += 1

        if picked_set is not None:
            plot_rows.append(row_codes)
            plot_meta.append((chrom, pos))
        else:
            if plot_seen < args.max_plot_variants:
                plot_rows.append(row_codes)
                plot_meta.append((chrom, pos))
            else:
                j = rnd.randrange(plot_seen + 1)
                if j < args.max_plot_variants:
                    plot_rows[j] = row_codes
                    plot_meta[j] = (chrom, pos)
            plot_seen += 1

        region_local_counter += 1

    # ----- Outputs -----
    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, args.out_name)
    stats_path = f"{out_prefix}.stats.tsv"
    with open(stats_path, "w", encoding="utf-8") as w:
        w.write("\t".join([
            "sample", "total_sites", "called", "missing",
            "het_count", "het_rate", "missing_rate",
            "snp_count", "indel_count", "code0", "code1", "code2"
        ]) + "\n")
        for st in stats:
            called = st["called"]
            total = st["total"]
            het_rate = (st["het"] / called) if called > 0 else 0.0
            missing_rate = (st["missing"] / total) if total > 0 else 0.0
            w.write("\t".join([
                st["sample"],
                str(total),
                str(called),
                str(st["missing"]),
                str(st["het"]),
                f"{het_rate:.6f}",
                f"{missing_rate:.6f}",
                str(st["snp"]),
                str(st["indel"]),
                str(st["code0"]),
                str(st["code1"]),
                str(st["code2"]),
            ]) + "\n")

    # plot using visualizer
    if plot_rows:
        visualizer = Visualizer()
        fig_path = f"{out_prefix}.genotypes.{image_format}"
        visualizer.plot_genotype(
            plot_rows=plot_rows,
            plot_meta=plot_meta,
            selected_sample_names=selected_names,
            stats=stats,
            group_map=group_map,
            sample_metric=args.sample_metric,
            variant_metric=args.variant_metric,
            figsize=(args.width, args.height),
            out_path=fig_path,
        )
        logger.info(f"Genotype figure saved to: {fig_path}")


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
        base_name = os.path.splitext(os.path.basename(qtl_file))[0]
        base_name = base_name.split(".")[0]
        qtl_df = qtl.read_block_file(qtl_file)
        qtl_blocks.append(qtl_df.assign(trait=f"{base_name}_{i}"))
    qtl_blocks = pd.concat(qtl_blocks)
    logger.info(f"Loaded {len(qtl_blocks)} QTL blocks from {len(args.qtl)} files.")
    # Load genome file if provided
    genome = qtl.read_genome_file(args.genome)
    logger.info(f"Loaded genome file with {len(genome)} chromosomes.")
    # Create figure
    fig = plt.figure(figsize=(args.width, args.height))
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
    visualizer.plot_qtl_map(qtl_blocks, genome, ax=ax, chrom_style=chrom_style)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.out_name}.{args.format}"), dpi=300, bbox_inches="tight")
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
    scan_parser.add_argument("--ld_window", type=int, default=999999, help="Maximum number of SNPs in LD window (default: %(default)s)")
    scan_parser.add_argument("--ld_window_kb", type=int, default=100, help="Maximum number of kb in LD window (default: %(default)s)")
    scan_parser.add_argument("--ld_window_r2", type=float, default=0.2, help="LD threshold (R^2) for LD window (default: %(default)s)")
    scan_parser.add_argument("--threads", type=int, default=cpu_count(), help="Number of threads for PLINK (default: %(default)s)")
    scan_parser.add_argument("--pvalue", type=float, default=1.0, help="P-value threshold for significant SNPs (default: %(default)s)")
    scan_parser.add_argument("--gff", type=str, help="Path to GFF file for gene annotation")
    scan_parser.add_argument("--main_type", type=str, default="mRNA", help="Main feature type to extract from GFF file (default: %(default)s)")
    scan_parser.add_argument("--attr_id", type=str, default="ID", help="Attribute ID to extract gene names from GFF file (default: %(default)s)")
    scan_parser.set_defaults(func=run_scan)

    # QTL Network subcommand
    qtl_network_parser = subparsers.add_parser("qtlnetwork", help="Generate QTL network")
    qtl_network_parser.add_argument("--qtl", type=str, nargs="+", required=True, help="Path(s) to QTL blocks csv file(s). Multiple files with space separated are allowed.")
    qtl_network_parser.add_argument("--vcf", type=str, required=True, help="Path to VCF genotype file")
    qtl_network_parser.add_argument("--layout", type=str, default="spring", choices=['spring', 'atlas'], 
                                    help="Layout for QTL network (default: %(default)s)")
    qtl_network_parser.add_argument("--k", type=float, default=0.3, help="Spring layout parameter k (default: %(default)s)")
    qtl_network_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic node layouts (default: %(default)s)")
    qtl_network_parser.add_argument("--names", type=str, nargs="+", help="Rename different QTL blocks file(s) for visualization")
    qtl_network_parser.add_argument("--min_weight", type=float, default=0, help="Minimum weight for edges (default: %(default)s)")
    qtl_network_parser.add_argument("--with_labels", action="store_true", help="Whether to show node labels")
    qtl_network_parser.add_argument("--node_size_factor", type=float, default=10, help="Factor to multiply -log10(pvalue) for node size (default: %(default)s)")
    qtl_network_parser.add_argument("--edge_width_factor", type=float, default=3, help="Factor to multiply LD value for edge width (default: %(default)s)")
    qtl_network_parser.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    qtl_network_parser.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
    qtl_network_parser.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    qtl_network_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    qtl_network_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    qtl_network_parser.set_defaults(func=run_qtlnetwork)

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
    manhattan_parser.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
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
    qtlmap_parser.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    qtlmap_parser.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
    qtlmap_parser.add_argument("--format", type=str, default="png", help="Output format, e.g., pdf or png (default: %(default)s)")
    qtlmap_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    qtlmap_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    qtlmap_parser.set_defaults(plot_func=plot_qtl_map)

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

    # Group for Genotype Heatmap from VCF
    geno_parser = plot_subparsers.add_parser("genotype", help="Generate genotype heatmap and per-sample/variant metrics from VCF")
    geno_parser.add_argument("--vcf", type=str, required=True, help="Path to VCF/BCF file (supports .gz)")
    geno_parser.add_argument("--samples", default=None, help="Comma-separated sample names")
    geno_parser.add_argument("--samples_file", default=None, help="File with one sample name per line")
    geno_parser.add_argument("--random_samples", type=int, default=None, help="Number of random samples to select")
    geno_parser.add_argument("--regions", nargs="+", default=None, help="Genomic regions like chr1:1-100000; can be provided multiple times")
    geno_parser.add_argument("--regions_file", default=None, help="BED file with regions: chrom start end")
    geno_parser.add_argument("--random_variants", type=int, default=None, help="Randomly sample N variants after region filtering")
    geno_parser.add_argument("--count_ref", action="store_true", help="Count SNP/INDEL per site for non-missing calls (default: count only ALT carriers)")
    geno_parser.add_argument("--max_plot_variants", type=int, default=2000, help="Max variants to plot via reservoir sampling when --random-variants is not set")
    geno_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    geno_parser.add_argument("--groups_file", default=None, help="Sample grouping file: two columns 'sample group', whitespace/tab/comma separated")
    geno_parser.add_argument("--random_groups", type=int, default=None, help="Number of random groups when no --groups_file is provided")
    geno_parser.add_argument("--sample_metric", default="mis",
                       choices=["mis", "het", "count", "count0", "count1", "count2"],
                       help="Top subplot metric for samples: mis/het/count or count0/count1/count2")
    geno_parser.add_argument("--variant_metric", default="maf", choices=["mis", "het", "maf"], help="Right subplot metric for variants")
    geno_parser.add_argument("--width", type=float, default=10, help="Figure width (default: %(default)s)")
    geno_parser.add_argument("--height", type=float, default=8, help="Figure height (default: %(default)s)")
    geno_parser.add_argument("--format", type=str, default="png", help="Image format: png/pdf/svg (default: %(default)s)")
    geno_parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    geno_parser.add_argument("--out_name", type=str, default="output", help="Output file name prefix (default: %(default)s)")
    geno_parser.set_defaults(plot_func=plot_genotype)

    # ML subcommands group
    ml_parser = subparsers.add_parser("ml", help="Machine learning workflows: train models and run predictions")
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command", help="ML subcommands")

    # ml train
    ml_train = ml_subparsers.add_parser("train", help="Train genotype-to-phenotype model(s) from VCF + phenotype and generate report")
    # inputs
    ml_train.add_argument("--vcf", type=str, required=True, help="Path to VCF/VCF.GZ genotype file")
    ml_train.add_argument("--phe", type=str, required=True, help="Path to phenotype file (CSV/TSV)")
    ml_train.add_argument("--sample-col", type=str, default=None, help="Sample ID column name in phenotype file (default: auto-detect first column)")
    ml_train.add_argument("--traits", type=str, default=None, help="Comma separated trait names; default: all non-sample columns")
    ml_train.add_argument("--target-type", type=str, default="reg", choices=["reg", "clf"], help="Task type: regression or classification (default: %(default)s)")
    # variant selection & QC
    ml_train.add_argument("--variant-selection", type=str, default="all", choices=["all", "random", "ids"], help="Variant selection strategy (default: %(default)s)")
    ml_train.add_argument("--random-variants", type=int, default=None, help="When variant-selection=random, number of variants to sample")
    ml_train.add_argument("--variant-ids-file", type=str, default=None, help="When variant-selection=ids, file containing variant IDs (one per line)")
    ml_train.add_argument("--maf-min", type=float, default=0.01, help="Minimum MAF to keep a variant (default: %(default)s)")
    ml_train.add_argument("--max-missing", type=float, default=0.2, help="Maximum missing rate to keep a variant (default: %(default)s)")
    # preprocessing & split
    ml_train.add_argument("--impute", type=str, default="mean", choices=["mean", "median", "most_frequent"], help="Imputation strategy (default: %(default)s)")
    ml_train.add_argument("--scale", type=str, default="none", choices=["none", "standard", "minmax"], help="Feature scaling (default: %(default)s)")
    ml_train.add_argument("--test-size", type=float, default=0.2, help="Test size ratio (default: %(default)s)")
    ml_train.add_argument("--cv", type=int, default=5, help="CV folds (default: %(default)s)")
    ml_train.add_argument("--eval-cv", action="store_true", help="Also run K-fold evaluation on the training split and report mean/std metrics (uses --cv folds)")
    ml_train.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s)")
    # models & tuning
    ml_train.add_argument("--model", type=str, default="lgb", choices=["svm", "lgb", "xgb", "rf", "lasso", "knn"], help="Model type (default: %(default)s)")
    ml_train.add_argument("--auto-params", action="store_true", help="Enable heuristic auto-parameterization based on sample size and feature count; merged with --params (user values take precedence)")
    ml_train.add_argument("--params", type=str, default=None, help="Path to JSON file of model parameters to override defaults")
    ml_train.add_argument("--tune", type=str, default="none", choices=["none", "grid", "random", "bayes"], help="Hyperparameter tuning strategy (default: %(default)s)")
    ml_train.add_argument("--n-iter", type=int, default=30, help="Iterations for random/bayes search (default: %(default)s)")
    ml_train.add_argument("--param-grid", type=str, default=None, help="Path to JSON file for GridSearchCV param_grid")
    ml_train.add_argument("--param-space", type=str, default=None, help="Path to JSON file for Random/Bayes param space")
    ml_train.add_argument("--scoring", type=str, default=None, help="Custom scoring; default r2 for reg, roc_auc/accuracy for clf")
    # importance & SHAP
    ml_train.add_argument("--importance", type=str, default="shap", choices=["auto", "shap", "permutation", "none"], help="Feature importance method (default: %(default)s)")
    ml_train.add_argument("--importance-topk", type=int, default=10, help="TopK features to plot in bar chart (default: %(default)s)")
    ml_train.add_argument("--shap-background", type=int, default=200, help="Number of background samples for SHAP (default: %(default)s)")
    ml_train.add_argument("--shap-dependence-topk", type=int, default=10, help="TopK features for SHAP dependence plots (default: %(default)s)")
    ml_train.add_argument("--importance-fast", action="store_true", help="Use native importance for tree models to speed up (fallback from SHAP)")
    # report & output
    ml_train.add_argument("--report", dest="report", action="store_true", help="Enable HTML report (default: on)")
    ml_train.add_argument("--no-report", dest="report", action="store_false", help="Disable HTML report")
    ml_train.set_defaults(report=True)
    ml_train.add_argument("--report-title", type=str, default="QTLScan Prediction Report", help="HTML report title (default: %(default)s)")
    # Template is fixed and always used for report rendering; no per-run template flags
    ml_train.add_argument("--width", type=float, default=6, help="Figure width (default: %(default)s)")
    ml_train.add_argument("--height", type=float, default=5, help="Figure height (default: %(default)s)")
    ml_train.add_argument("--format", type=str, default="png", help="Plot image format (default: %(default)s)")
    ml_train.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    ml_train.add_argument("--out_name", type=str, default="train", help="Output name prefix (default: %(default)s)")
    ml_train.set_defaults(func=run_train)

    # ml predict
    ml_pred = ml_subparsers.add_parser("predict", help="Apply a trained bundle to new features for prediction")
    ml_pred.add_argument("--bundle", type=str, required=True, help="Path to a *.bundle.json saved by train")
    ml_pred.add_argument("--features", type=str, required=True, help="Path to feature CSV/TSV for new samples (rows=samples; columns=features; first column=sample if --sample-col not set)")
    ml_pred.add_argument("--sample-col", type=str, default=None, help="Sample ID column name (default: first column)")
    ml_pred.add_argument("--trait", type=str, default=None, help="Optional trait name label for outputs")
    ml_pred.add_argument("--out_dir", type=str, default=".", help="Output directory (default: %(default)s)")
    ml_pred.add_argument("--out_name", type=str, default="predict", help="Output name prefix (default: %(default)s)")
    ml_pred.set_defaults(func=run_predict)
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
