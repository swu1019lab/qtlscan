import copy
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib.patches import Polygon, RegularPolygon, Rectangle, Patch, FancyBboxPatch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.transforms import Affine2D

from qtlscan.log import logger
from typing import List, Tuple, Dict, Optional



class Visualizer:
    def __init__(self):
        pass

    def plot_genotype(
        self,
        plot_rows: List[List[int]],
        plot_meta: List[Tuple[str, int]],
        selected_sample_names: List[str],
        stats: List[Dict[str, int]],
        group_map: Dict[str, str],
        sample_metric: str = "mis",
        variant_metric: str = "maf",
        figsize: Tuple[float, float] = (10.0, 8.0),
        out_path: Optional[str] = None,
    ):
        """
        Plot genotype heatmap with top sample-metric bar and right variant-metric bar.

        Inputs:
        - plot_rows: list of variant rows; each row is a list of genotype codes per sample (-1,0,1,2)
        - plot_meta: list of (chrom, pos) matching plot_rows
        - selected_sample_names: list of sample names aligned to columns of plot_rows
        - stats: per-sample stats dicts (total/called/missing/het/code0/1/2 ...)
        - group_map: mapping of sample -> group name
        - sample_metric: top bar metric: mis/het/count or count0/count1/count2
        - variant_metric: right bar metric: mis/het/maf
        - figsize: figure size (w,h)
        - out_path: if provided, save figure to this path; otherwise return the figure

        Returns: matplotlib Figure if out_path is None, else None (saved to disk)
        """
        if not plot_rows:
            logger.warning("No genotypes to plot (plot_rows is empty)")
            return None

        def map_code(c: int) -> int:
            if c == -1:
                return 0
            if c == 0:
                return 1
            if c == 1:
                return 2
            if c == 2:
                return 3
            return 0

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(["lightgray", "steelblue", "gold", "firebrick"])
        norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)

        # group rows by chromosome and stitch in chrom order
        grouped: Dict[str, List[Tuple[int, List[int]]]] = {}
        for i, (chrom, pos) in enumerate(plot_meta):
            grouped.setdefault(chrom, []).append((pos, plot_rows[i]))
        for chrom in grouped:
            grouped[chrom].sort(key=lambda x: x[0])

        stitched: List[List[int]] = []
        chrom_order: List[str] = []
        group_sizes_y: List[int] = []
        for chrom in sorted(grouped.keys(), key=lambda c: (c,)):
            rows = [row for _pos, row in grouped[chrom]]
            if not rows:
                continue
            chrom_order.append(chrom)
            group_sizes_y.append(len(rows))
            stitched.extend(rows)

        # order columns by group_map
        gmap = {n: group_map.get(n, "All") for n in selected_sample_names}
        group_names = sorted({gmap[n] for n in selected_sample_names})
        col_order: List[int] = []
        group_col_sizes: List[int] = []
        ordered_sample_names: List[str] = []
        for g in group_names:
            idxs = [i for i, n in enumerate(selected_sample_names) if gmap[n] == g]
            if not idxs:
                continue
            col_order.extend(idxs)
            group_col_sizes.append(len(idxs))
            ordered_sample_names.extend([selected_sample_names[i] for i in idxs])

        stitched_re = [[row[i] for i in col_order] for row in stitched]
        mapped = [[map_code(c) for c in row] for row in stitched_re]

        fig_w, fig_h = figsize
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, constrained_layout=True)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 0.15], height_ratios=[0.25, 1], wspace=0.02, hspace=0.02, figure=fig)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        ax_main = fig.add_subplot(gs[1, 0])

        im = ax_main.imshow(mapped, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

        ncols = len(ordered_sample_names)
        nrows = len(mapped)
        ax_main.set_xlabel("Samples")
        ax_main.set_ylabel("Variants")
        ax_main.set_xticks(range(ncols))
        if ncols > 50:
            step = max(1, ncols // 50)
            ticks = list(range(0, ncols, step))
            ax_main.set_xticks(ticks)
            ax_main.set_xticklabels([ordered_sample_names[i] for i in ticks], rotation=90, fontsize=6)
        else:
            ax_main.set_xticklabels(ordered_sample_names, rotation=90, fontsize=7)

        boundaries_y: List[int] = []
        acc = 0
        for sz in group_sizes_y:
            acc += sz
            boundaries_y.append(acc)
        for b in boundaries_y[:-1]:
            ax_main.hlines(b - 0.5, -0.5, ncols - 0.5, colors="white", linewidth=1.0)

        yticks = []
        ylabels = []
        start = 0
        for chrom, sz in zip(chrom_order, group_sizes_y):
            mid = start + sz / 2 - 0.5
            yticks.append(mid)
            ylabels.append(chrom)
            start += sz
        ax_main.set_yticks(yticks)
        ax_main.set_yticklabels(ylabels, fontsize=7)

        boundaries_x: List[int] = []
        accx = 0
        for sz in group_col_sizes:
            accx += sz
            boundaries_x.append(accx)
        for b in boundaries_x[:-1]:
            ax_main.vlines(b - 0.5, -0.5, nrows - 0.5, colors="white", linewidth=1.0)

        # top subplot: per-sample metric
        if sample_metric == "count":
            c0 = [stats[i].get("code0", 0) for i in col_order]
            c1 = [stats[i].get("code1", 0) for i in col_order]
            c2 = [stats[i].get("code2", 0) for i in col_order]
            bottom_c1 = c0[:]
            bottom_c2 = [c0[j] + c1[j] for j in range(ncols)]
            ax_top.bar(range(ncols), c0, color="steelblue", width=1.0, label="0")
            ax_top.bar(range(ncols), c1, bottom=bottom_c1, color="gold", width=1.0, label="1")
            ax_top.bar(range(ncols), c2, bottom=bottom_c2, color="firebrick", width=1.0, label="2")
            ax_top.set_ylabel("count")
        else:
            def sample_value(st: Dict[str, int]) -> float:
                if sample_metric == "mis":
                    total = st["total"]
                    return (st["missing"] / total) if total > 0 else 0.0
                if sample_metric == "het":
                    called = st["called"]
                    return (st["het"] / called) if called > 0 else 0.0
                if sample_metric == "count0":
                    return float(st.get("code0", 0))
                if sample_metric == "count1":
                    return float(st.get("code1", 0))
                if sample_metric == "count2":
                    return float(st.get("code2", 0))
                return 0.0

            sample_vals = [sample_value(stats[i]) for i in col_order]
            ax_top.bar(range(ncols), sample_vals, color="gray", width=1.0)
            ax_top.set_ylabel(sample_metric)

        ax_top.set_xlim([-0.5, ncols - 0.5])
        ax_top.set_xticks([])
        ymin, ymax = ax_top.get_ylim()
        for b in boundaries_x[:-1]:
            ax_top.vlines(b - 0.5, ymin, ymax, colors="white", linewidth=1.0)

        # right subplot: per-variant metric
        def variant_value(row: List[int]) -> float:
            n = len(row)
            called = [c for c in row if c != -1]
            if variant_metric == "mis":
                return (n - len(called)) / n if n > 0 else 0.0
            if variant_metric == "het":
                return (sum(1 for c in called if c == 1) / len(called)) if called else 0.0
            if variant_metric == "maf":
                alt = sum(1 if c == 1 else 2 if c == 2 else 0 for c in called)
                denom = 2 * len(called)
                f = (alt / denom) if denom > 0 else 0.0
                return min(f, 1.0 - f)
            return 0.0

        var_vals = [variant_value(row) for row in stitched_re]
        ax_right.barh(range(nrows), var_vals, color="gray", height=1.0)
        ax_right.set_ylim([-0.5, nrows - 0.5])
        ax_right.set_yticks([])
        ax_right.set_xlabel(variant_metric)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color="lightgray", label="Missing"),
            Patch(color="steelblue", label="Hom-REF"),
            Patch(color="gold", label="HET"),
            Patch(color="firebrick", label="Hom-ALT"),
        ]
        ax_main.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=True)

        if out_path:
            fig.savefig(out_path)
            plt.close(fig)
            return None
        return fig

    def plot_ld_heatmap(
            self, 
            ld_df: pd.DataFrame,
            blocks: pd.DataFrame = None,
            plot_value: bool = False, 
            cmap=None, 
            ax=None):
        """
        Plot LD heatmap and optionally highlight blocks.

        :param ld_df: DataFrame containing LD information (columns: CHR_A, BP_A, SNP_A, CHR_B, BP_B, SNP_B, R2).
        :param blocks: DataFrame containing block information (columns: chr, bp1, bp2).
        :param plot_value: Whether to plot values in heatmap.
        :param cmap: Colormap for LD heatmap.
        :param ax: Matplotlib Axes object for plotting.
        """
        logger.info("Plotting LD heatmap...")
        if ax is None:
            raise ValueError("Please provide a valid Matplotlib Axes object for plotting.")
        
        # Extract SNP positions and p-values
        snp_positions = sorted(set(ld_df["BP_A"].tolist() + ld_df["BP_B"].tolist()))
        snp_index = {pos: i for i, pos in enumerate(snp_positions)}

        # Create LD matrix
        n_snps = len(snp_positions)
        ld_matrix = np.zeros((n_snps, n_snps))
        for _, row in ld_df.iterrows():
            i = snp_index[row["BP_A"]]
            j = snp_index[row["BP_B"]]
            ld_matrix[i, j] = row["R2"]
            ld_matrix[j, i] = row["R2"]  # Fill upper triangle

        # Set colormap
        if cmap is None:
            cmap = 'Reds'
        cmap = mpl.colormaps.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        # Get patch collection to plot
        patches = []
        values = []
        start = 1
        stop = n_snps
        for i in np.arange(start, stop):
            diag_values = np.diag(ld_matrix, -i)
            values.extend(diag_values)
            for j in np.arange(0.5, len(diag_values) + 0.5):
                patches.append(RegularPolygon((j + i * 0.5, (n_snps - i) / 2), numVertices=4, radius=0.5))

        patch_collection = PatchCollection(patches)
        patch_collection.set_array(values)
        patch_collection.set_cmap(cmap)
        patch_collection.set_norm(norm)

        ax.add_collection(patch_collection)
        ax.set_aspect('equal')
        ax.set_xlim(start * 0.5, stop - start * 0.5)
        ax.set_ylim(-0.1, (n_snps - start) / 2 + 0.5 + 0.1)
        ax.set_axis_off()

        # Add color bar
        cax = ax.inset_axes([0.8, 0.01, 0.03, 0.5])
        ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, shrink=.5, label=r"$R^2$")

        # Add text annotations if plot_value is True
        if plot_value:
            logger.info("Adding text annotations to the heatmap...")
            text_colors = ("black", "white")
            color_array = patch_collection.get_array()
            threshold = patch_collection.norm(color_array.max()) / 2
            for i, p in enumerate(patches):
                text = ax.text(p.xy[0], p.xy[1], "{:.2f}".format(values[i]),
                            ha="center", va="center",
                            color=text_colors[int(values[i] > threshold)])
                patch_bbox = p.get_window_extent()
                text_width = text.get_window_extent().transformed(ax.transData.inverted()).width
                font_size = text.get_fontsize()
                while font_size > 1 and text_width > patch_bbox.width / 2:
                    font_size -= 1
                    text.set_fontsize(font_size)
                    text_width = text.get_window_extent().transformed(ax.transData.inverted()).width

        # Add LD blocks if provided
        if blocks is not None:
            logger.info("Adding LD blocks to the heatmap...")
            for _, row in blocks.iterrows():
                bp1 = row["bp1"]
                bp2 = row["bp2"]
                if bp1 in snp_index and bp2 in snp_index:
                    idx1 = snp_index[bp1]
                    idx2 = snp_index[bp2]
                    nsnps = idx2 - idx1 + 1
                    # Create a polygon for the QTL block
                    polygon = Polygon(
                        [
                            (idx1 + 0.5, (n_snps - 1) / 2),
                            (idx2 + 0.5, (n_snps - 1) / 2),
                            ((idx1 + idx2 + 1) / 2, (n_snps - nsnps) / 2),
                        ],
                        closed=True,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=1,
                        alpha=1,
                        zorder=10,
                    )
                    ax.add_patch(polygon)

    def plot_snp_positions(
        self,
        snp_positions: None,
        ax: None):
        """
        Add SNP locations to the heatmap.

        :param snp_positions: List of SNP positions.
        :param ax: Matplotlib Axes object for plotting.
        """
        # Add SNP locations
        logger.info("Adding SNP locations to the heatmap...")
        snp_loc = np.asarray(snp_positions, dtype=np.int64)
        start = 1
        stop = len(snp_loc)
        sx = (stop - start) / (np.max(snp_loc) - np.min(snp_loc))
        scale_loc = Affine2D(). \
            translate(-np.min(snp_loc), ty=0). \
            scale(sx=sx, sy=1). \
            translate(tx=start * 0.5, ty=0). \
            transform(np.column_stack([snp_loc, [1] * len(snp_loc)]))

        line_collection = LineCollection([[[a[0], 1], [i + 0.5, 0]] for i, a in enumerate(scale_loc)], linewidths=.5)
        line_collection.set_color('black')
        ax.add_collection(line_collection)
        ax.set_xlim(start * 0.5, stop - start * 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines.bottom.set_visible(False)

    def plot_pvalue(
        self,
        snp_positions: None,
        pvalues: np.array,
        chrom: str = None,
        title: str = None,
        ax: axes.Axes = None):
        """
        Add P-values to the heatmap.

        :param snp_positions: List of SNP positions.
        :param pvalues: List of P-values.
        """
        logger.info("Adding P-value scatter plot to the heatmap...")
        pvalues = np.asarray(pvalues, dtype=np.float64)
        snp_loc = np.asarray(snp_positions, dtype=np.int64)
        start = 1
        stop = len(snp_loc)
        sx = (stop - start) / (np.max(snp_loc) - np.min(snp_loc))
        scale_loc = Affine2D(). \
            translate(-np.min(snp_loc), ty=0). \
            scale(sx=sx, sy=1). \
            translate(tx=start * 0.5, ty=0). \
            transform(np.column_stack([snp_loc, [1] * len(snp_loc)]))

        ax.scatter([a[0] for a in scale_loc], -np.log10(pvalues), color='#38638D', s=10)
        ax.set_xlim(start * 0.5, stop - start * 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        if chrom is not None:
            ax.set_xlabel(f'Chromosome {str(chrom).replace("scaffold", "")} (Mb)')
        ax.set_ylabel(r"$\mathrm{-log}_{10}(\mathit{p})$")
        if title is not None and chrom is not None:
            ax.set_title(f"{title} (Chr {chrom})")
        # Add ticks and labels using ticker.FuncFormatter
        snp_loc_mb = snp_loc / 1e6  # Transform to Mb
        ax.set_xticks(scale_loc[:, 0][[0, -1]], np.round(snp_loc_mb[[0, -1]], 2))

    def plot_manhattan(self, df, point_size=5,
                       chr_unit='mb', chr_gap=0, chr_colors=None,
                       sig_threshold=None, sig_line_style=None, 
                       xlabel=None, ylabel=None, title=None,
                       ax=None, plot_type='gwas', plot_style='scatter'):
        """
        Plot Manhattan plot for GWAS/Fst/Pi summary statistics.
        
        :param df: DataFrame containing summary statistics
        :param point_size: Point size for Manhattan plot
        :param chr_unit: Position unit, one of ['mb', 'kb', 'bp']
        :param chr_gap: Gap between chromosomes in the unit specified
        :param chr_colors: List of colors for chromosomes. if the length of the list is less than the number of chromosomes, the colors will be cycled.
        :param sig_threshold: Significance threshold p-value, default is 1/n, n is the number of SNPs, can be a float or a list of floats
        :param sig_line_style: Style of the significance line
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param title: Title of the plot
        :param ax: Matplotlib Axes object for plotting
        :param plot_type: Type of data: 'gwas', 'fst', 'pi', 'pi_ratio'
        :param plot_style: Plot style: 'scatter', 'vlines'
        """
        logger.info("Plotting Manhattan plot...")
        # Unit conversion factors
        unit_factors = {'mb': 1e-6, 'kb': 1e-3, 'bp': 1}
        factor = unit_factors.get(chr_unit.lower(), 1e-6)

        # convert data based on plot type and prepare data format
        if plot_type.lower() == 'gwas':
            # For GWAS data from GEMMA: expects columns chr, ps, p_wald
            df["plot_value"] = -np.log10(df["p_wald"])
            default_ylabel = r"$-\log_{10}(p)$"
        elif plot_type.lower() == 'fst':
            # For Fst data from vcftools: expects columns CHROM, BIN_START, BIN_END, WEIGHTED_FST
            # Convert to standard format
            if 'CHROM' in df.columns:
                df['chr'] = df['CHROM']
                df['ps'] = (df['BIN_START'] + df['BIN_END']) / 2
            df["plot_value"] = df["WEIGHTED_FST"]
            default_ylabel = "Fst"
        elif plot_type.lower() == 'pi':
            # For Pi data from vcftools: expects columns CHROM, BIN_START, BIN_END, PI
            # Convert to standard format
            if 'CHROM' in df.columns:
                df['chr'] = df['CHROM']
                df['ps'] = (df['BIN_START'] + df['BIN_END']) / 2
            df["plot_value"] = df["PI"]
            default_ylabel = "Pi"
        elif plot_type.lower() == 'pi_ratio':
            # For Pi ratio data: expects columns CHROM, BIN_START, BIN_END, PI_RATIO
            # Convert to standard format
            if 'CHROM' in df.columns:
                df['chr'] = df['CHROM']
                df['ps'] = (df['BIN_START'] + df['BIN_END']) / 2
            df["plot_value"] = df["PI_RATIO"]
            default_ylabel = "Pi Ratio"
        else:
            # Default: assume it's p-value data
            df["plot_value"] = -np.log10(df["p_wald"])
            default_ylabel = r"$-\log_{10}(p)$"
        
        # remove scaffold prefix from chr column
        df["chr"] = df["chr"].str.replace("scaffold", "")

        # sort by chromosome and position
        df = df.sort_values(by=["chr", "ps"])
        
        df["ps"] = df["ps"] * factor

        # Set chromosome colors
        if chr_colors is None:
            chr_colors = ["#38638D", "#4F94CD"]

        # Calculate cumulative positions for each chromosome
        chrom_start = {}
        chrom_center = {}
        current_pos = 0
        for chrom, group in df.groupby("chr"):
            chrom_start[chrom] = current_pos
            chrom_center[chrom] = current_pos + group["ps"].max() / 2
            current_pos += group["ps"].max() + chr_gap

        # Plot data points
        chrom_index = 0
        for chrom, group in df.groupby("chr"):
            if plot_style.lower() == 'scatter':
                ax.scatter(group["ps"] + chrom_start[chrom], group["plot_value"], color=chr_colors[chrom_index % len(chr_colors)], s=point_size)
            elif plot_style.lower() == 'vlines':
                ax.vlines(group["ps"] + chrom_start[chrom], 0, group["plot_value"], color=chr_colors[chrom_index % len(chr_colors)], linewidth=0.5)
            else:
                ax.scatter(group["ps"] + chrom_start[chrom], group["plot_value"], color=chr_colors[chrom_index % len(chr_colors)], s=point_size)
            chrom_index += 1

        # Set default parameters for significance threshold line
        default_sig_params = {
            'color': 'C3',
            'linestyle': '--',
            'linewidth': 1,
        }

        if sig_line_style:
            default_sig_params.update(sig_line_style)

        # Draw significance threshold line based on plot type
        if sig_threshold is not None:
            if plot_type.lower() == 'gwas':
                # For p-value data: convert threshold to -log10 scale
                if isinstance(sig_threshold, dict):
                    # Handle threshold dictionary from _calculate_thresholds
                    for conf_level, threshold in sig_threshold.items():
                        ax.axhline(-np.log10(threshold), **default_sig_params, label=f"{conf_level} threshold")
                elif isinstance(sig_threshold, list):
                    for threshold in sig_threshold:
                        ax.axhline(-np.log10(threshold), **default_sig_params)
                else:
                    ax.axhline(-np.log10(sig_threshold), **default_sig_params)
            else:
                # For Fst/Pi/Pi_ratio data: use threshold directly
                if isinstance(sig_threshold, dict):
                    # Handle threshold dictionary from _calculate_thresholds
                    for conf_level, threshold in sig_threshold.items():
                        ax.axhline(threshold, **default_sig_params, label=f"{conf_level} threshold")
                elif isinstance(sig_threshold, list):
                    for threshold in sig_threshold:
                        ax.axhline(threshold, **default_sig_params)
                else:
                    ax.axhline(sig_threshold, **default_sig_params)
        elif plot_type.lower() == 'gwas':
            # Default threshold for p-value data
            default_threshold = 1 / len(df)
            ax.axhline(-np.log10(default_threshold), **default_sig_params)

        # Set x-axis ticks and labels
        ax.set_xticks(list(chrom_center.values()), list(chrom_center.keys()))
        ax.tick_params(axis='x', which='major', rotation=90)

        # turn off top and right spines
        ax.spines[['top', 'right']].set_visible(False)

        # Set x-axis label
        ax.set_xlabel(xlabel if xlabel is not None else f"Chromosome Position ({chr_unit.upper()})")

        # Set y-axis label
        ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)

        # Set x-axis and y-axis limits
        ax.set_xlim(0, current_pos)
        ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1])

        # Set title
        ax.set_title(title if title is not None else "Manhattan Plot")

        # Add legend
        ax.legend(loc="upper right", frameon=False)

    def plot_qq(self, df, point_size=5, xlabel=None, ylabel=None, title=None, ax=None):
        """
        Plot QQ plot for GWAS or eQTL summary statistics.
        
        :param df: DataFrame containing summary statistics (columns: p_wald).
        :param point_size: Point size for QQ plot
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param title: Title of the plot
        :param ax: Matplotlib Axes object for plotting
        """
        logger.info("Plotting QQ plot...")
        observed = -np.log10(np.sort(df["p_wald"]))
        expected = -np.log10(np.linspace(1 / len(df), 1, len(df)))

        # Plot
        ax.scatter(expected, observed, s=point_size, color="#38638D")
        ax.plot([0, max(expected)], [0, max(expected)], color="red", linestyle="--", linewidth=1)

        # Format plot
        ax.set_xlabel(xlabel if xlabel is not None else r"Expected $-\log_{10}(p)$")
        ax.set_ylabel(ylabel if ylabel is not None else r"Observed $-\log_{10}(p)$")
        ax.set_title(title if title is not None else "QQ Plot")
        plt.tight_layout()

    def plot_qtl_map(
        self,
        blocks: pd.DataFrame,
        genome: pd.DataFrame,
        chrom_names=None,
        chrom_style: dict = None,
        ann_data: pd.DataFrame = None,
        ann_style: dict = None,
        colors=None,
        orientation: str = "vertical",
        unit: str = "mb",
        ax=None):
        """
        Plot genome-wide QTL distribution map
        
        :param blocks: QTL data (must contain trait, chr, bp1, bp2 columns)
        :param genome: Chromosome length data (must contain chr, len columns)
        :param chrom_names: Specified chromosomes to display
        :param chrom_style: Chromosome style parameters
        :param ann_data: Annotation data (must contain chr, bp1, bp2, label columns)
        :param ann_style: Annotation style parameters
        :param colors: List of colors for traits
        :param orientation: Layout direction (horizontal/vertical)
        :param unit: Genomic unit (mb/kb/bp)
        :param ax: matplotlib axis object
        """

        blocks = copy.deepcopy(blocks)
        genome = copy.deepcopy(genome)

        # Data validation
        for df, cols in zip([blocks, genome], [["trait", "chr", "bp1", "bp2"], ["chr", "len"]]):
            missing = [col for col in cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        # Filter chromosomes
        if chrom_names is not None:
            chrom_names = [chrom_names] if isinstance(chrom_names, str) else chrom_names
            blocks = blocks[blocks["chr"].isin(chrom_names)].copy()
            genome = genome[genome["chr"].isin(chrom_names)].copy()
            if genome.empty:
                raise ValueError("No chromosomes remain after filtering")

        # Unit conversion
        unit_factors = {'mb': 1e6, 'kb': 1e3, 'bp': 1}
        if unit not in unit_factors:
            raise ValueError(f"Invalid unit: {unit}, choose from 'mb', 'kb', 'bp'")
        
        factor = unit_factors[unit]
        blocks["bp1"] = blocks["bp1"] / factor
        blocks["bp2"] = blocks["bp2"] / factor
        genome["len"] = genome["len"] / factor

        # Parameter settings for chromosomes
        chrom_style = chrom_style or {}
        default_chrom_style = {
            "width": 0.8, 
            "margin": 0.2,
            "spacing": 0.5,
            "rounding": 0.3,
            "facecolor": "lightgrey",
            "edgecolor": "black",
            "edge_width": 0.8
        }
        for k, v in default_chrom_style.items():
            chrom_style.setdefault(k, v)

        # Color mapping for traits
        traits = blocks["trait"].unique()
        if colors is None and len(traits) <= 10:
            cmap = mpl.colormaps.get_cmap("tab10")
            colors = cmap(range(len(traits)))
        elif colors is None and len(traits) > 10:
            cmap = mpl.colormaps.get_cmap("tab20")
            colors = cmap(range(len(traits)))
        trait_color_map = dict(zip(traits, colors))

        # Select layout mode
        def _draw_horizontal_layout(blocks, genome, params, color_map, unit, ax):
            """Horizontal layout drawing logic"""
            # Calculate chromosome positions
            genome = genome.sort_values("chr").reset_index(drop=True)
            positions = []
            current_pos = params["margin"]
            
            for _ in range(len(genome)):
                positions.append(current_pos)
                current_pos += params["width"] + params["spacing"]

            # Draw chromosome background (rounded corners)
            max_len = genome["len"].max()
            for x, (_, chrom_row) in zip(positions, genome.iterrows()):
                chrom = chrom_row["chr"]
                length = chrom_row["len"]
                
                # Chromosome outline (rounded)
                ax.add_patch(FancyBboxPatch(
                    (x, 0), params["width"], length,
                    boxstyle=f"round,pad=0,rounding_size={params['rounding']}",
                    facecolor=params["facecolor"],
                    edgecolor=params["edgecolor"],
                    linewidth=params["edge_width"],
                    alpha=0.7
                ))
                
                # Draw QTLs (rectangles)
                chrom_qtls = blocks[blocks["chr"] == chrom]
                if chrom_qtls.empty:
                    continue
                for _, qtl in chrom_qtls.iterrows():
                    ax.add_patch(Rectangle(
                        (x, qtl["bp1"]), params["width"], qtl["bp2"]-qtl["bp1"],
                        facecolor=color_map[qtl["trait"]],
                        edgecolor=color_map[qtl["trait"]],
                        linewidth=0.5,
                        alpha=0.9
                    ))

            # Configure axes
            ax.set_xlim(0, positions[-1] + params["width"] + params["spacing"])
            ax.set_ylim(0, max_len*1.02)
            ax.spines[['top', 'bottom', 'right']].set_visible(False)
            ax.set_xticks([x + params["width"]/2 for x in positions])
            ax.set_xticklabels(genome["chr"])
            ax.tick_params(axis='x', which='major', color='white')
            ax.set_xlabel("Chromosome")
            ax.set_ylabel(f"Genomic Position ({unit.upper()})")

        def _draw_vertical_layout(blocks, genome, params, color_map, unit, ax):
            """Vertical layout drawing logic"""
            # Calculate chromosome positions
            genome = genome.sort_values("chr").reset_index(drop=True)
            positions = []
            current_pos = params["margin"]
            
            for _ in range(len(genome)):
                positions.append(current_pos)
                current_pos += params["width"] + params["spacing"]

            # Draw chromosome background (rounded corners)
            max_len = genome["len"].max()
            for y, (_, chrom_row) in zip(positions, genome.iterrows()):
                chrom = chrom_row["chr"]
                length = chrom_row["len"]
                
                # Chromosome outline (rounded)
                ax.add_patch(FancyBboxPatch(
                    (0, y), length, params["width"],
                    boxstyle=f"round,pad=0,rounding_size={params['rounding']}",
                    facecolor=params["facecolor"],
                    edgecolor=params["edgecolor"],
                    linewidth=params["edge_width"],
                    alpha=0.7
                ))
                
                # Draw QTLs (rectangles)
                chrom_qtls = blocks[blocks["chr"] == chrom]
                for _, qtl in chrom_qtls.iterrows():
                    ax.add_patch(Rectangle(
                        (qtl["bp1"], y), qtl["bp2"]-qtl["bp1"], params["width"],
                        facecolor=color_map[qtl["trait"]],
                        edgecolor=color_map[qtl["trait"]],
                        linewidth=0.5,
                        alpha=0.9
                    ))

            # Configure axes
            ax.set_ylim(0, positions[-1] + params["width"] + params["spacing"])
            ax.set_xlim(0, max_len*1.02)
            ax.spines[['top', 'left', 'right']].set_visible(False)
            ax.set_yticks([y + params["width"]/2 for y in positions])
            ax.set_yticklabels(genome["chr"])
            ax.tick_params(axis='y', which='major', color='white')
            ax.set_ylabel("Chromosome")
            ax.set_xlabel(f"Genomic Position ({unit.upper()})")

        if orientation == "horizontal":
            _draw_horizontal_layout(blocks, genome, chrom_style, trait_color_map, unit, ax)
        elif orientation == "vertical":
            _draw_vertical_layout(blocks, genome, chrom_style, trait_color_map, unit, ax)
        else:
            raise ValueError("Invalid orientation, choose 'horizontal' or 'vertical'")

        # Add legend
        legend_patches = [Patch(color=c, label=t) for t, c in trait_color_map.items()]
        ax.legend(handles=legend_patches, 
                bbox_to_anchor=(1.05, 1), 
                loc="upper left",
                frameon=False,
                title="Traits")

        # Handle annotation blocks
        if ann_data is not None and not ann_data.empty:
            # Data validation
            if not all(col in ann_data.columns for col in ["chr", "bp1", "bp2", "label"]):
                raise ValueError("ann_data need chr, bp1, bp2, label columns")
            
            # Unit conversion
            ann_data = ann_data.copy()
            ann_data["bp1"] = ann_data["bp1"] / factor
            ann_data["bp2"] = ann_data["bp2"] / factor
            
            # Set default annotation style
            ann_style = ann_style or {}
            default_ann_style = {
                'fontsize': 10,
                'color': 'black',
                'ha': 'left' if orientation == 'horizontal' else 'center',
                'va': 'center' if orientation == 'horizontal' else 'bottom',
                'alpha': 0.9,
                'clip_on': False
            }
            for k, v in default_ann_style.items():
                ann_style.setdefault(k, v)
            
            # Process annotation positions based on orientation
            genome_sorted = genome.sort_values('chr').reset_index(drop=True)
            if orientation == 'horizontal':
                # Calculate x positions for each chromosome
                positions = []
                current_pos = chrom_style['margin']
                for _ in range(len(genome_sorted)):
                    positions.append(current_pos)
                    current_pos += chrom_style['width'] + chrom_style['spacing']
                chrom_to_x = {row.chr: x for x, row in zip(positions, genome_sorted.itertuples())}
                
                # Process annotations for each chromosome
                for chrom, group in ann_data.groupby('chr'):
                    if chrom not in chrom_to_x:
                        continue
                    x_chrom = chrom_to_x[chrom]
                    x_pos = x_chrom + chrom_style['width'] + chrom_style['margin'] * 0.3  # Right margin
                    
                    # Calculate initial y positions (midpoint) and text heights
                    group = group.copy()
                    group['y'] = (group['bp1'] + group['bp2']) / 2
                    group_sorted = group.sort_values('y')
                    
                    # Calculate text height (in data units)
                    y_min, y_max = ax.get_ylim()
                    fig = ax.get_figure()
                    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    height_in = bbox.height
                    data_per_inch = (y_max - y_min) / height_in
                    line_height = ann_style['fontsize'] / 72 * data_per_inch * 1.2
                    
                    text_heights = [
                        (label.count('\n') + 1) * line_height
                        for label in group_sorted['label']
                    ]
                    
                    # Adjust y positions to avoid overlap
                    adjusted_y = []
                    min_spacing = line_height * 0.5
                    for i, y in enumerate(group_sorted['y']):
                        if not adjusted_y:
                            adjusted_y.append(y)
                            continue
                        
                        current_height = text_heights[i]
                        prev_top = adjusted_y[-1] + text_heights[i-1]/2
                        current_bottom = y - current_height/2
                        
                        if current_bottom < prev_top + min_spacing:
                            y = prev_top + min_spacing + current_height/2
                        adjusted_y.append(y)
                    
                    # Draw annotations
                    for (idx, row), y in zip(group_sorted.iterrows(), adjusted_y):
                        ax.annotate(row['label'], 
                                    xy=(x_chrom + chrom_style['width'], group_sorted['y'][idx]), xycoords='data',
                                    xytext=(x_pos, y), textcoords='data',
                                    arrowprops=dict(arrowstyle='-', connectionstyle="arc3"),
                                    **ann_style)

            else:  # Vertical layout
                # Calculate y positions for each chromosome
                positions = []
                current_pos = chrom_style['margin']
                for _ in range(len(genome_sorted)):
                    positions.append(current_pos)
                    current_pos += chrom_style['width'] + chrom_style['spacing']
                chrom_to_y = {row.chr: y for y, row in zip(positions, genome_sorted.itertuples())}
                
                # Process annotations for each chromosome
                for chrom, group in ann_data.groupby('chr'):
                    if chrom not in chrom_to_y:
                        continue
                    y_chrom = chrom_to_y[chrom]
                    y_pos = y_chrom + chrom_style['width'] + chrom_style['margin'] * 0.3  # Top margin
                    
                    # Calculate initial x positions (midpoint) and text widths
                    group = group.copy()
                    group['x'] = (group['bp1'] + group['bp2']) / 2
                    group_sorted = group.sort_values('x')
                    
                    # Calculate text width (in data units)
                    x_min, x_max = ax.get_xlim()
                    fig = ax.get_figure()
                    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    width_in = bbox.width
                    data_per_inch = (x_max - x_min) / width_in
                    char_width = ann_style['fontsize'] / 72 * data_per_inch * 0.6
                    
                    text_widths = [
                        max(len(line) for line in label.split('\n')) * char_width
                        for label in group_sorted['label']
                    ]
                    
                    # Adjust x positions to avoid overlap
                    adjusted_x = []
                    min_spacing = char_width * 2
                    for i, x in enumerate(group_sorted['x']):
                        if not adjusted_x:
                            adjusted_x.append(x)
                            continue
                        
                        current_width = text_widths[i]
                        prev_right = adjusted_x[-1] + text_widths[i-1]/2
                        current_left = x - current_width/2
                        
                        if current_left < prev_right + min_spacing:
                            x = prev_right + min_spacing + current_width/2
                        adjusted_x.append(x)
                    
                    # Draw annotations
                    for (idx, row), x in zip(group_sorted.iterrows(), adjusted_x):
                        ax.annotate(row['label'],
                                    xy=(group_sorted['x'][idx], y_chrom + chrom_style['width']), xycoords='data',
                                    xytext=(x, y_pos), textcoords='data',
                                    arrowprops=dict(arrowstyle='-', connectionstyle="arc3"),
                                    **ann_style)

    def plot_dist(self, df, kind='box', columns=None, colors=None, 
                  alpha=0.7, orientation='vertical', bins=30, density=False,
                  fit_curve=False, fit_bins=100, hist_style='dodge', kde_style='stack',
                  kde_points=200, extend_tail=0.0, extend_head=0.0, 
                  xlabel=None, ylabel=None, ax=None):
        """
        Plot distribution of data in DataFrame columns using different visualization methods.
        
        :param df: DataFrame containing data to plot
        :param kind: Type of plot ('box', 'violin', 'hist', 'kde')
        :param columns: List of column names to plot. If None, use all numeric columns
        :param colors: List of colors for each column. If None, use default matplotlib colors
        :param alpha: Transparency level for plots (0-1)
        :param orientation: Plot orientation ('vertical' or 'horizontal')
        :param bins: Number of bins for histogram (default: 30)
        :param density: Whether to show density instead of count for histogram (default: False)
        :param fit_curve: Whether to add fitted curve for histogram (default: False)
        :param fit_bins: Number of bins for fitted curve (default: 100)
        :param hist_style: Style for multiple columns in histogram ('stack', 'dodge', 'side')
        :param kde_style: Style for multiple columns in KDE ('stack', 'side')
        :param kde_points: Number of points for KDE calculation (default: 200)
        :param extend_tail: Extend factor for smaller values (left boundary), range: 0-1 (default: 0.0)
        :param extend_head: Extend factor for larger values (right boundary), range: 0-1 (default: 0.0)
        :param xlabel: Custom label for x-axis. If None, use default labels
        :param ylabel: Custom label for y-axis. If None, use default labels
        :param ax: Matplotlib Axes object for plotting
        """
        logger.info(f"Plotting distribution using {kind} plot...")
        
        if ax is None:
            raise ValueError("Please provide a valid Matplotlib Axes object for plotting.")
        
        # Select columns
        if columns is None:
            # Use all numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate columns exist in DataFrame
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        if not columns:
            raise ValueError("No numeric columns found in DataFrame")
        
        # Set colors
        if colors is None:
            colors = [f"C{i}" for i in range(len(columns))]
        elif len(colors) < len(columns):
            # Cycle colors if not enough provided
            colors = (colors * ((len(columns) // len(colors)) + 1))[:len(columns)]
        
        # Remove missing values
        plot_data = df[columns].dropna()
        
        if kind == 'box':
            self.plot_box(plot_data, columns, colors, alpha, orientation, ax)
        elif kind == 'violin':
            self.plot_violin(plot_data, columns, colors, alpha, orientation, ax)
        elif kind == 'hist':
            self.plot_hist(plot_data, columns, colors, alpha, orientation, 
                                       bins, density, fit_curve, fit_bins, hist_style, ax)
        elif kind == 'kde':
            self.plot_kde(plot_data, columns, colors, alpha, orientation, 
                                      kde_style, ax, kde_points, extend_tail, extend_head)
        else:
            raise ValueError(f"Invalid kind: {kind}. Choose from 'box', 'violin', 'hist', 'kde'")
        
        # Set labels and title
        if orientation == 'vertical':
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        ax.set_title(f'{kind.capitalize()} Plot of Distribution')
        
        # Turn off top and right spines
        ax.spines[['top', 'right']].set_visible(False)
        
        # Add legend for multi-column plots
        if len(columns) > 1 and kind in ['hist', 'kde']:
            ax.legend(frameon=False)

    def plot_box(self, data, columns, colors, alpha, orientation, ax):
        """Plot box plot distribution"""
        if orientation == 'vertical':
            box_plot = ax.boxplot([data[col].values for col in columns], 
                                 tick_labels=columns, patch_artist=True)
            # Set colors
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)
        else:
            box_plot = ax.boxplot([data[col].values for col in columns], 
                                 tick_labels=columns, vert=False, patch_artist=True)
            # Set colors
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)

    def plot_violin(self, data, columns, colors, alpha, orientation, ax):
        """Plot violin plot distribution"""
        if orientation == 'vertical':
            parts = ax.violinplot([data[col].values for col in columns], 
                                 positions=range(1, len(columns) + 1))
            # Set colors
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
            ax.set_xticks(range(1, len(columns) + 1))
            ax.set_xticklabels(columns)
        else:
            # For horizontal violin plots, we need to transpose the data approach
            parts = ax.violinplot([data[col].values for col in columns], 
                                 positions=range(1, len(columns) + 1), vert=False)
            # Set colors
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
            ax.set_yticks(range(1, len(columns) + 1))
            ax.set_yticklabels(columns)

    def plot_hist(self, data, columns, colors, alpha, orientation, 
                               bins, density, fit_curve, fit_bins, hist_style, ax):
        """Plot histogram distribution"""
        from scipy.stats import shapiro, normaltest, norm, sem

        def _perform_normality_tests_and_confidence_intervals(data, columns):
            """
            Perform normality tests and calculate confidence intervals for each column
            
            :param data: DataFrame containing the data
            :param columns: List of column names to analyze
            """
            for col in columns:
                col_data = data[col].dropna().values
                if len(col_data) < 3:
                    logger.warning(f"Column '{col}': Insufficient data for analysis (n={len(col_data)})")
                    continue
                
                logger.info(f"Analysis for column: {col}")
                logger.info(f"Sample size: {len(col_data)}")
                logger.info(f"Mean: {np.mean(col_data):.4f}")
                logger.info(f"Standard deviation: {np.std(col_data, ddof=1):.4f}")
                logger.info(f"Minimum: {np.min(col_data):.4f}")
                logger.info(f"Maximum: {np.max(col_data):.4f}")

                # Perform normality tests
                logger.info("Normality Tests:")
                
                # Shapiro-Wilk test (recommended for small to medium samples)
                if len(col_data) <= 5000:  # Shapiro-Wilk is most reliable for smaller samples
                    try:
                        shapiro_stat, shapiro_p = shapiro(col_data)
                        logger.info(f"Shapiro-Wilk Test:")
                        logger.info(f"  Statistic: {shapiro_stat:.6f}")
                        logger.info(f"  P-value: {shapiro_p:.6f}")
                        if shapiro_p > 0.05:
                            logger.info(f"  Result: Data appears to be normally distributed (p > 0.05)")
                        else:
                            logger.info(f"  Result: Data does NOT appear to be normally distributed (p ≤ 0.05)")
                    except Exception as e:
                        logger.error(f"  Shapiro-Wilk Test failed: {e}")
                
                # D'Agostino and Pearson's test (good for larger samples)
                if len(col_data) >= 8:  # Minimum sample size for this test
                    try:
                        dagostino_stat, dagostino_p = normaltest(col_data)
                        logger.info(f"D'Agostino-Pearson Test:")
                        logger.info(f"  Statistic: {dagostino_stat:.6f}")
                        logger.info(f"  P-value: {dagostino_p:.6f}")
                        if dagostino_p > 0.05:
                            logger.info(f"  Result: Data appears to be normally distributed (p > 0.05)")
                        else:
                            logger.info(f"  Result: Data does NOT appear to be normally distributed (p ≤ 0.05)")
                    except Exception as e:
                        logger.error(f"  D'Agostino-Pearson Test failed: {e}")
                
                # calculate confidence intervals
                logger.info("Confidence Intervals:")
                
                try:
                    # 95% confidence interval
                    ci_95_lower, ci_95_upper = norm.interval(0.95, loc=np.mean(col_data), scale=np.std(col_data, ddof=1))

                    # 99% confidence interval
                    ci_99_lower, ci_99_upper = norm.interval(0.99, loc=np.mean(col_data), scale=np.std(col_data, ddof=1))
                    
                    # Count data points within confidence intervals
                    data_in_95 = np.sum((col_data >= ci_95_lower) & (col_data <= ci_95_upper))
                    data_in_99 = np.sum((col_data >= ci_99_lower) & (col_data <= ci_99_upper))
                    
                    logger.info(f"95% Confidence Interval:")
                    logger.info(f"  Range: [{ci_95_lower:.4f}, {ci_95_upper:.4f}]")
                    logger.info(f"  Data points within interval: {data_in_95}/{len(col_data)} ({100*data_in_95/len(col_data):.1f}%)")
                    
                    logger.info(f"99% Confidence Interval:")
                    logger.info(f"  Range: [{ci_99_lower:.4f}, {ci_99_upper:.4f}]")
                    logger.info(f"  Data points within interval: {data_in_99}/{len(col_data)} ({100*data_in_99/len(col_data):.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Confidence intervals calculation failed: {e}")

        # Perform normality tests and calculate confidence intervals for each column
        _perform_normality_tests_and_confidence_intervals(data, columns)

        # Plot histogram
        if len(columns) == 1:
            # Single column histogram
            col = columns[0]
            if orientation == 'vertical':
                n, bins_edges, patches = ax.hist(data[col].values, bins=bins, 
                                                density=density, alpha=alpha, 
                                                color=colors[0], label=col)
                if fit_curve:
                    mu, sigma = norm.fit(data[col].values)
                    x = np.linspace(data[col].min(), data[col].max(), fit_bins)
                    y = norm.pdf(x, mu, sigma)
                    if not density:
                        y = y * len(data[col]) * (bins_edges[1] - bins_edges[0])
                    ax.plot(x, y, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[0])
            else:
                n, bins_edges, patches = ax.hist(data[col].values, bins=bins, 
                                                density=density, alpha=alpha, 
                                                orientation='horizontal',
                                                color=colors[0], label=col)
                if fit_curve:
                    mu, sigma = norm.fit(data[col].values)
                    y = np.linspace(data[col].min(), data[col].max(), fit_bins)
                    x = norm.pdf(y, mu, sigma)
                    if not density:
                        x = x * len(data[col]) * (bins_edges[1] - bins_edges[0])
                    ax.plot(x, y, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[0])
        else:
            # Multiple columns histogram
            if hist_style == 'stack':
                if orientation == 'vertical':
                    n, bins_edges, patches = ax.hist([data[col].values for col in columns], bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, stacked=True)
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            mu, sigma = norm.fit(data[col].values)
                            x = np.linspace(data[col].min(), data[col].max(), fit_bins)
                            y = norm.pdf(x, mu, sigma)
                            if not density:
                                y = y * len(data[col]) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, label=f'{col} (μ={mu:.2f}, σ={sigma:.2f})', color=colors[i])
                else:
                    n, bins_edges, patches = ax.hist([data[col].values for col in columns], bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, orientation='horizontal', stacked=True)
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            mu, sigma = norm.fit(data[col].values)
                            y = np.linspace(data[col].min(), data[col].max(), fit_bins)
                            x = norm.pdf(y, mu, sigma)
                            if not density:
                                x = x * len(data[col]) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[i])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
            elif hist_style == 'dodge':
                if orientation == 'vertical':
                    n, bins_edges, patches = ax.hist([data[col].values for col in columns], bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, stacked=False)
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            mu, sigma = norm.fit(data[col].values)
                            x = np.linspace(data[col].min(), data[col].max(), fit_bins)
                            y = norm.pdf(x, mu, sigma)
                            if not density:
                                y = y * len(data[col]) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[i])
                else:
                    n, bins_edges, patches = ax.hist([data[col].values for col in columns], bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, orientation='horizontal', stacked=False)
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            mu, sigma = norm.fit(data[col].values)
                            y = np.linspace(data[col].min(), data[col].max(), fit_bins)
                            x = norm.pdf(y, mu, sigma)
                            if not density:
                                x = x * len(data[col]) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[i])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
            elif hist_style == 'side':
                # Create side bar plots
                bin_edges = np.histogram_bin_edges(data[columns].values.flatten(), bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bar_width = (bin_edges[1] - bin_edges[0]) / (len(columns) + 1)
                
                if orientation == 'vertical':
                    bottom = 0.0
                    y_locations = []
                    for i, col in enumerate(columns):
                        hist, _ = np.histogram(data[col].values, bins=bin_edges, density=density)
                        ax.bar(bin_centers + i * bar_width - (len(columns)-1) * bar_width / 2, 
                            hist, width=bar_width, alpha=alpha, color=colors[i], label=col, bottom=bottom)
                        if fit_curve:
                            mu, sigma = norm.fit(data[col].values)
                            x = np.linspace(data[col].min(), data[col].max(), fit_bins)
                            y = norm.pdf(x, mu, sigma)
                            if not density:
                                y = y * len(data[col]) * (bin_edges[1] - bin_edges[0])
                            ax.plot(x, y + bottom, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[i])
                        bottom += np.max(hist)
                        y_locations.append(bottom - np.max(hist) / 2)
                    ax.set_yticks(y_locations, labels=columns)
                else:
                    left = 0.0
                    x_locations = []
                    for i, col in enumerate(columns):
                        hist, _ = np.histogram(data[col].values, bins=bin_edges, density=density)
                        ax.barh(bin_centers + i * bar_width - (len(columns)-1) * bar_width / 2, 
                            hist, height=bar_width, alpha=alpha, color=colors[i], label=col, left=left)
                        if fit_curve:
                            mu, sigma = norm.fit(data[col].values)
                            y = np.linspace(data[col].min(), data[col].max(), fit_bins)
                            x = norm.pdf(y, mu, sigma)
                            if not density:
                                x = x * len(data[col]) * (bin_edges[1] - bin_edges[0])
                            ax.plot(x + left, y, '--', linewidth=2, label=f'{col} (mean={mu:.2f}, std={sigma:.2f})', color=colors[i])
                        left += np.max(hist)
                        x_locations.append(left - np.max(hist) / 2)
                    ax.set_xticks(x_locations, labels=columns)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)

    def plot_kde(self, data, columns, colors, alpha, orientation, 
                              kde_style, ax, kde_points=200, extend_tail=0.0, extend_head=0.0):
        """
        Plot KDE distribution with mean lines and extended boundaries
        
        For each column, plots KDE curve and adds a dashed line indicating the mean value
        using the same color as the KDE curve. The mean line length is based on the KDE 
        density value at the mean position.
        
        :param kde_points: Number of points for KDE calculation (default: 200)
        :param extend_tail: Extend factor for smaller values (left boundary), range: 0-1 (default: 0.0)
        :param extend_head: Extend factor for larger values (right boundary), range: 0-1 (default: 0.0)
        """
        if len(columns) == 1:
            # Single column KDE
            col = columns[0]
            values = data[col].dropna().values
            if len(values) > 1:
                kde = gaussian_kde(values)
                
                # Calculate extended boundaries
                data_min, data_max = values.min(), values.max()
                data_range = data_max - data_min
                
                # Apply boundary extensions
                extend_tail_amount = data_range * min(max(extend_tail, 0.0), 1.0)  # Clamp to [0, 1]
                extend_head_amount = data_range * min(max(extend_head, 0.0), 1.0)  # Clamp to [0, 1]
                
                x_min = data_min - extend_tail_amount
                x_max = data_max + extend_head_amount
                x_range = np.linspace(x_min, x_max, kde_points)
                
                y = kde(x_range)
                mean_val = values.mean()
                
                # Get KDE value at mean position
                mean_kde_val = kde(mean_val)[0]
                
                if orientation == 'vertical':
                    ax.plot(x_range, y, color=colors[0], alpha=alpha, linewidth=2, label=col)
                    ax.fill_between(x_range, y, alpha=alpha*0.3, color=colors[0])
                    # Add mean line with length based on KDE value at mean
                    ax.plot([mean_val, mean_val], [0, mean_kde_val], color=colors[0], 
                           linestyle='--', alpha=0.8, linewidth=1.5)
                    # Add mean value text annotation
                    ax.text(mean_val, mean_kde_val * 1.05, f'{mean_val:.2f}', 
                           color=colors[0], ha='center', va='bottom', fontsize=9, fontweight='bold')
                else:
                    ax.plot(y, x_range, color=colors[0], alpha=alpha, linewidth=2, label=col)
                    ax.fill_betweenx(x_range, y, alpha=alpha*0.3, color=colors[0])
                    # Add mean line with length based on KDE value at mean
                    ax.plot([0, mean_kde_val], [mean_val, mean_val], color=colors[0], 
                           linestyle='--', alpha=0.8, linewidth=1.5)
                    # Add mean value text annotation
                    ax.text(mean_kde_val * 1.05, mean_val, f'{mean_val:.2f}', 
                           color=colors[0], ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            # Multiple columns KDE
            if kde_style == 'stack':
                # Stacked KDE plots - use unified extended range for all columns
                all_values = []
                for col in columns:
                    col_values = data[col].dropna().values
                    if len(col_values) > 1:
                        all_values.extend(col_values)
                
                if all_values:
                    # Calculate unified extended boundaries for all columns
                    data_min, data_max = min(all_values), max(all_values)
                    data_range = data_max - data_min
                    
                    # Apply boundary extensions
                    extend_tail_amount = data_range * min(max(extend_tail, 0.0), 1.0)  # Clamp to [0, 1]
                    extend_head_amount = data_range * min(max(extend_head, 0.0), 1.0)  # Clamp to [0, 1]
                    
                    x_min = data_min - extend_tail_amount
                    x_max = data_max + extend_head_amount
                    unified_x_range = np.linspace(x_min, x_max, kde_points)
                
                for i, col in enumerate(columns):
                    values = data[col].dropna().values
                    if len(values) > 1:
                        kde = gaussian_kde(values)
                        y = kde(unified_x_range)
                        mean_val = values.mean()
                        
                        # Get KDE value at mean position
                        mean_kde_val = kde(mean_val)[0]
                        
                        if orientation == 'vertical':
                            ax.plot(unified_x_range, y, color=colors[i], alpha=alpha, 
                                   linewidth=2, label=col)
                            ax.fill_between(unified_x_range, y, alpha=alpha*0.3, color=colors[i])
                            # Add mean line with length based on KDE value at mean
                            ax.plot([mean_val, mean_val], [0, mean_kde_val], color=colors[i], 
                                   linestyle='--', alpha=0.8, linewidth=1.5)
                            # Add mean value text annotation
                            ax.text(mean_val, mean_kde_val * (1.05 + i * 0.15), f'{mean_val:.2f}', 
                                   color=colors[i], ha='center', va='bottom', fontsize=8, fontweight='bold')
                        else:
                            ax.plot(y, unified_x_range, color=colors[i], alpha=alpha, 
                                   linewidth=2, label=col)
                            ax.fill_betweenx(unified_x_range, y, alpha=alpha*0.3, color=colors[i])
                            # Add mean line with length based on KDE value at mean
                            ax.plot([0, mean_kde_val], [mean_val, mean_val], color=colors[i], 
                                   linestyle='--', alpha=0.8, linewidth=1.5)
                            # Add mean value text annotation
                            ax.text(mean_kde_val * (1.05 + i * 0.15), mean_val, f'{mean_val:.2f}', 
                                   color=colors[i], ha='left', va='center', fontsize=8, fontweight='bold')
            elif kde_style == 'side':
                n_cols = len(columns)
                if orientation == 'vertical':
                    # Vertical orientation: divide Y-axis into independent regions
                    kde_data = []
                    mean_values = []
                    for col in columns:
                        values = data[col].dropna().values
                        if len(values) > 1:
                            kde = gaussian_kde(values)
                            
                            # Calculate extended boundaries for each column independently
                            data_min, data_max = values.min(), values.max()
                            data_range = data_max - data_min
                            
                            # Apply boundary extensions
                            extend_tail_amount = data_range * min(max(extend_tail, 0.0), 1.0)  # Clamp to [0, 1]
                            extend_head_amount = data_range * min(max(extend_head, 0.0), 1.0)  # Clamp to [0, 1]
                            
                            x_min = data_min - extend_tail_amount
                            x_max = data_max + extend_head_amount
                            x_range = np.linspace(x_min, x_max, kde_points)
                            
                            y = kde(x_range)
                            kde_data.append((kde, x_range, y))
                            mean_values.append(values.mean())
                        else:
                            kde_data.append((None, None, None))
                            mean_values.append(None)
                    
                    # Plot each KDE in its independent Y region
                    for i, (col, (kde, x_range, y), mean_val) in enumerate(zip(columns, kde_data, mean_values)):
                        if x_range is not None:
                            # Normalize Y values to [0,1] then map to region [i, i+1]
                            y_normalized = y / y.max()  # Normalize to [0,1]
                            y_region = y_normalized + i  # Map to region [i, i+1]
                            
                            ax.plot(x_range, y_region, color=colors[i], 
                                   alpha=alpha, linewidth=2, label=col)
                            ax.fill_between(x_range, i, y_region, 
                                          alpha=alpha*0.3, color=colors[i])
                            
                            # Add mean line for this region with KDE-based length
                            mean_kde_val = kde(mean_val)[0]
                            mean_kde_normalized = mean_kde_val / y.max()  # Normalize to match region
                            ax.plot([mean_val, mean_val], [i, i + mean_kde_normalized], color=colors[i], 
                                   linestyle='--', alpha=0.8, linewidth=1.5)
                            # Add mean value text annotation
                            ax.text(mean_val, i + mean_kde_normalized * 1.1, f'{mean_val:.2f}', 
                                   color=colors[i], ha='center', va='bottom', fontsize=8, fontweight='bold')
                    
                    # Set Y-axis ticks and labels for each region
                    ax.set_yticks([i + 0.5 for i in range(n_cols)])
                    ax.set_yticklabels(columns)
                    ax.set_ylim(-0.1, n_cols + 0.1)
                    
                else:
                    # Horizontal orientation: divide X-axis into independent regions
                    kde_data = []
                    mean_values = []
                    for col in columns:
                        values = data[col].dropna().values
                        if len(values) > 1:
                            kde = gaussian_kde(values)
                            
                            # Calculate extended boundaries for each column independently
                            data_min, data_max = values.min(), values.max()
                            data_range = data_max - data_min
                            
                            # Apply boundary extensions
                            extend_tail_amount = data_range * min(max(extend_tail, 0.0), 1.0)  # Clamp to [0, 1]
                            extend_head_amount = data_range * min(max(extend_head, 0.0), 1.0)  # Clamp to [0, 1]
                            
                            y_min = data_min - extend_tail_amount
                            y_max = data_max + extend_head_amount
                            y_range = np.linspace(y_min, y_max, kde_points)
                            
                            x = kde(y_range)
                            kde_data.append((kde, x, y_range))
                            mean_values.append(values.mean())
                        else:
                            kde_data.append((None, None, None))
                            mean_values.append(None)
                    
                    # Plot each KDE in its independent X region
                    for i, (col, (kde, x, y_range), mean_val) in enumerate(zip(columns, kde_data, mean_values)):
                        if x is not None:
                            # Normalize X values to [0,1] then map to region [i, i+1]
                            x_normalized = x / x.max()  # Normalize to [0,1]
                            x_region = x_normalized + i  # Map to region [i, i+1]
                            
                            ax.plot(x_region, y_range, color=colors[i], 
                                   alpha=alpha, linewidth=2, label=col)
                            ax.fill_betweenx(y_range, i, x_region, 
                                           alpha=alpha*0.3, color=colors[i])
                            
                            # Add mean line for this region with KDE-based length
                            mean_kde_val = kde(mean_val)[0]
                            mean_kde_normalized = mean_kde_val / x.max()  # Normalize to match region
                            ax.plot([i, i + mean_kde_normalized], [mean_val, mean_val], color=colors[i], 
                                   linestyle='--', alpha=0.8, linewidth=1.5)
                            # Add mean value text annotation
                            ax.text(i + mean_kde_normalized * 1.1, mean_val, f'{mean_val:.2f}', 
                                   color=colors[i], ha='left', va='center', fontsize=8, fontweight='bold')
                    
                    # Set X-axis ticks and labels for each region
                    ax.set_xticks([i + 0.5 for i in range(n_cols)])
                    ax.set_xticklabels(columns)
                    ax.set_xlim(-0.1, n_cols + 0.1)

