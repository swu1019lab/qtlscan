import copy
import re
import textwrap
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
import matplotlib.ticker as ticker
import random
import os

from qtlscan.log import logger
from typing import List, Tuple, Dict, Optional



class Visualizer:
    def __init__(self):
        pass

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

    def plot_gene_structure(self, genes, gff_df=None, snp_df=None, ann_df=None, ax=None, main_type='mRNA', 
                            feature_params=None, track_params=None, attr_id='ID', attr_parent='Parent',
                            unit='mb', xlim=None, marker_num=20, marker_size=8):
        """
        Plot gene structures based on GFF file and gene names.
        :param genes: List of gene names of interest.
        :param gff_df: GFF DataFrame containing gene annotation information and sub-features.
        :param snp_df: SNP DataFrame containing SNP positions and haplotype information, with 4+N columns: chrom, pos, ref, alt, hap1, hap2, ...
        :param ann_df: Annotation DataFrame containing SNP positions and annotation information, with 3 columns: chrom, pos, anno
        :param ax: Matplotlib Axes object for plotting.
        :param main_type: Main feature type (e.g., 'gene' or 'mRNA'), default is 'mRNA'.
        :param feature_params: Dictionary defining drawing parameters for different feature types.
        :param track_params: Dictionary defining drawing parameters for feature tracks.
        :param attr_id: Unique identifier for parsing genes or features, default is 'ID'.
        :param attr_parent: Parent attribute for parsing sub-features, default is 'Parent'.
        :param unit: Unit for genomic position ('mb', 'kb', or 'bp'), default is 'mb'.
        :param xlim: X-axis limits for the plot. must be a tuple (start, end) with unit bp.
        :param marker_num: Number of markers for drawing gene strands.
        :param marker_size: Marker size for drawing gene strands.
        """

        def assign_tracks(gene_intervals, track_height=0.4):
            """
            Assign tracks (y-axis positions) to genes based on overlap rules.
            :param gene_intervals: List of tuples (gene_name, start, end).
            :return: Dictionary with gene names as keys and assigned track numbers as values.
            """
            # Sort by start position
            sorted_genes = sorted(gene_intervals, key=lambda x: x[1])

            # Initialize variables
            gene_track = {}  # Store track assignments
            current_track = 0  # Current track number
            last_end = -1  # End position of the last gene

            for gene in sorted_genes:
                name, start, end = gene
                if start >= last_end:  # If the current gene does not overlap with the previous one
                    current_track = 0  # Reset track number to 0
                else:  # If overlapping
                    current_track += track_height  # Increment track number
                gene_track[name] = current_track  # Assign track
                last_end = end  # Update the end position of the last gene

            return gene_track

        # Convert positions to the specified unit
        def convert_position(x):
            if unit == 'mb':
                return int(x) / 1e6
            elif unit == 'kb':
                return int(x) / 1e3
            else:  # bp
                return int(x)

        # Default feature_params
        if feature_params is None:
            feature_params = {
                'type': ['exon', 'five_prime_UTR', 'three_prime_UTR'],
                'height': 0.05,
                'facecolor': ['#38638D', 'gray', 'gray'],
                'edgecolor': ['#38638D', 'gray', 'gray'],
                'zorder': 1,
                'label': True,
            }

        # Default track_params
        if track_params is None:
            track_params = {
                'height': 0.5,
                'link_height': 0.1,
                'margin': 0.1,
                'fontsize': 10,
            }

        # Parse gene information
        genes_info = {}
        for gene_name in genes:
            # print(f"Processing gene: {gene_name}")
            # Find entries of the specified main_type in the GFF file
            # Use exact match for gene name to avoid matching similar gene names
            feature_entry = gff_df[(gff_df['type'] == main_type) &
                                    (gff_df['attributes'].str.contains(f'{attr_id}={gene_name}(?:[;,]|$)', regex=True))]
            if not feature_entry.empty:
                seqid = feature_entry['chr'].values[0]  # Chromosome name
                feature_start = feature_entry['start'].values[0]  # Start position
                feature_end = feature_entry['end'].values[0]  # End position
                feature_strand = feature_entry['strand'].values[0]  # Strand direction

                # Filter sub-features based on chromosome, start, and end positions
                # consider multiple transcripts
                # Use exact match for gene name to avoid matching similar transcript names
                features = gff_df[
                    (gff_df['chr'] == seqid) &
                    (gff_df['start'] <= feature_end) &
                    (gff_df['end'] >= feature_start) &
                    (gff_df['attributes'].str.contains(f'{attr_parent}={gene_name}(?:[;,]|$)', regex=True))
                    ]

                # Extract required columns (type, start, end)
                features_df = features[['type', 'start', 'end']].copy()

                # Filter out types not defined in feature_params
                if 'type' in feature_params:
                    features_df = features_df[features_df['type'].isin(feature_params['type'])]

                # Filter SNPs based on chromosome and gene positions
                snps = None
                if snp_df is not None:
                    snps = snp_df[
                        (snp_df['chrom'] == seqid) & 
                        (snp_df['pos'] >= feature_start - 2000) &
                        (snp_df['pos'] <= feature_end + 2000)
                    ]
                    snps = snps.sort_values('pos')
                    snps = snps.reset_index(drop=True)

                # Filter SNPs annotation based on chromosome and gene positions
                anno = None
                if ann_df is not None:
                    anno = ann_df[
                        (ann_df['chrom'] == seqid) & 
                        (ann_df['pos'] >= feature_start - 2000) &
                        (ann_df['pos'] <= feature_end + 2000)
                    ]
                    anno = anno.sort_values('pos')
                    anno = anno.reset_index(drop=True)

                # Store gene information
                genes_info[gene_name] = {
                    'seqid': seqid,
                    'start': feature_start,
                    'end': feature_end,
                    'strand': feature_strand,
                    'features': features_df,
                    'snps': snps,
                    'anno': anno
                }
            else:
                raise ValueError(f"Gene {gene_name} not found in the GFF file with main type {main_type}.")

        # Check if genes are from the same chromosome
        chromosomes = set(info['seqid'] for info in genes_info.values())
        if len(chromosomes) > 1:
            raise ValueError(
                f"Genes are from different chromosomes: {chromosomes}. Please select genes from the same chromosome.")
        elif not chromosomes:
            raise ValueError("No chromosome information found! Please check the gene names.")
        else:
            logger.info(f"Check {len(genes_info)} genes are from the same chromosome: {chromosomes}")

        # Convert gene positions
        for gene_name in genes_info:
            genes_info[gene_name]['start'] = convert_position(genes_info[gene_name]['start'])
            genes_info[gene_name]['end'] = convert_position(genes_info[gene_name]['end'])
            genes_info[gene_name]['features']['start'] = genes_info[gene_name]['features']['start'].apply(convert_position)
            genes_info[gene_name]['features']['end'] = genes_info[gene_name]['features']['end'].apply(convert_position)

        # Assign tracks to genes
        gene_track = assign_tracks([(name, info['start'], info['end']) for name, info in genes_info.items()], track_params['height'])

        # Set axis labels and limits
        chromosome = next(iter(chromosomes))  # Get chromosome name
        chromo_start = min(info['start'] for info in genes_info.values()) if xlim is None else convert_position(xlim[0])
        chromo_end = max(info['end'] for info in genes_info.values()) if xlim is None else convert_position(xlim[1])
        # print(chromo_start, chromo_end)

        # Plot each gene
        for gene_name, info in genes_info.items():
            start = info['start']
            end = info['end']
            strand = info['strand']
            features = info['features']
            snps = info['snps']
            anno = info['anno']
            y_offset = gene_track[gene_name]  # Use assigned track as y-axis position

            # Plot gene line
            ax.plot([start, end], [y_offset, y_offset], color='black', linewidth=1, zorder=1)
            ax.plot(np.linspace(start, end, int(marker_num)), [y_offset] * int(marker_num), color='black',
                linewidth=.5, zorder=1, marker='4' if strand == '+' else '3', markersize=marker_size)
            
            # Plot gene name
            if feature_params['label']:
                center = (start + end) / 2
                height = max(feature_params['height']) if isinstance(feature_params['height'], list) else feature_params['height']
                ax.annotate(gene_name, xy=(center, y_offset + height / 2), xytext=(0, 8),
                            textcoords='offset points', fontsize=10, ha='center', va='center')

            # Plot features (exon, CDS, UTR, etc.)
            for _, row in features.iterrows():
                feature_type = row['type']
                feature_start = row['start']
                feature_end = row['end']

                # Get the index of the current feature type
                type_index = feature_params['type'].index(feature_type)

                # Get drawing parameters
                height = feature_params['height'][type_index] if isinstance(feature_params['height'], list) else \
                feature_params['height']
                facecolor = feature_params['facecolor'][type_index] if isinstance(feature_params['facecolor'], list) else \
                feature_params['facecolor']
                edgecolor = feature_params['edgecolor'][type_index] if isinstance(feature_params['edgecolor'], list) else \
                feature_params['edgecolor']
                zorder = feature_params['zorder'][type_index] if isinstance(feature_params['zorder'], list) else \
                feature_params['zorder']

                # Plot rectangle
                ax.add_patch(
                    Rectangle((feature_start, y_offset - height / 2), feature_end - feature_start,
                                    height, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder)
                )

            # Plot SNPs and haplotypes
            if snps is not None:
                num_haps = snps.shape[1] - 4
                num_snps = snps.shape[0]
                pos = snps['pos'].apply(convert_position)
                start = min(start, min(pos))
                end = max(end, max(pos))
                chromo_start = min(chromo_start, min(pos))
                chromo_end = max(chromo_end, max(pos))
                track_height = track_params['height']
                link_height = track_params['link_height']
                margin = track_params['margin'] # SNPs will be plotted in the middle of the gene track with a margin of 10% on both sides
                fontsize = track_params['fontsize']
                # cell_height = (track_height - height/2 - link_height) / num_haps
                cell_height = (track_height/2 - height/2 - link_height) / num_haps
                cell_width = (end - start) * (1 - 2 * margin) / num_snps
                cell_color = {0: 'C0', 1: 'gray', 2: 'C3'}

                if link_height > track_height:
                    logger.warning(f"Link height {link_height} is larger than track height {track_height}. "
                                    f"Please adjust the parameters.")

                for i in range(num_snps):
                    ref, alt = snps.iloc[i, 2], snps.iloc[i, 3]
                    texts = {0: ref + ref, 1: ref + alt, 2: alt + alt}
                    for j in range(num_haps):
                        ax.add_patch(
                            Rectangle((start + (end - start) * margin + cell_width * i, y_offset - height/2 - link_height - cell_height * (j + 1)),
                                        cell_width, cell_height, 
                                        facecolor=cell_color[snps.iloc[i, j + 4]], 
                                        edgecolor='white', zorder=2)
                        )
                        cell_text = texts.get(snps.iloc[i, j + 4], '')
                        ax.text(start + (end - start) * margin + cell_width * (i + 0.5), y_offset - height/2 - link_height - cell_height * (j + 0.5),
                                cell_text, fontsize=fontsize, ha='center', va='center', color = 'white')
                    # Draw links between SNPs and haplotypes
                    ax.plot([pos[i], pos[i], start + (end - start) * margin + cell_width * (i + 0.5)], 
                            [y_offset - height / 2, y_offset - height/2 - link_height / 2, y_offset - height/2 - link_height], 
                            color='gray', linewidth=.5, linestyle='-', zorder=1)

                for j in range(num_haps):
                    ax.annotate(snps.columns[j + 4], xy=(start + (end - start) * margin, y_offset - height/2 - link_height - cell_height * (j + 0.5)),
                                xytext=(-8, 0), textcoords='offset points', fontsize=fontsize, ha='right', va='center')

                # Add legend for reference and alternative alleles
                ax.legend(handles=[
                    Patch(facecolor='C0', edgecolor='white', label='Reference'),
                    Patch(facecolor='C3', edgecolor='white', label='Alternative')
                ], fontsize=fontsize + 4, bbox_to_anchor=(0., 1.02, 1., .11), loc='lower center', ncols=2, borderaxespad=0.)

            # Plot variant annotations
            if anno is not None and len(anno) > 0:
                logger.info(f"Plotting {len(anno)} variant annotations for gene {gene_name}...")
                
                # Get annotation positions and texts
                anno_pos = anno['pos'].apply(convert_position).tolist()
                anno_texts = anno['anno'].tolist()
                
                # Combine position and text for sorting and overlap detection
                anno_data = list(zip(anno_pos, anno_texts))
                anno_data.sort(key=lambda x: x[0])  # Sort by position
                
                # Define annotation plotting parameters
                track_height = track_params['height']
                height = max(feature_params['height']) if isinstance(feature_params['height'], list) else feature_params['height']
                anno_base_y = y_offset + track_height/5 # Base Y position for annotations
                anno_fontsize = max(6, fontsize - 2)  # Smaller font for annotations
                anno_colors = ['#E74C3C', '#8E44AD', '#3498DB', '#E67E22', '#27AE60']  # Distinct colors
                
                # Calculate text positions to avoid overlap
                text_pos = []
                min_distance = anno_fontsize * 0.00001  # Minimum distance threshold (approximately one character width)
                
                for i, (pos, text) in enumerate(anno_data):
                    if i == 0:
                        # First annotation uses original position
                        text_pos.append(pos)
                    else:
                        # Check distance to previous annotation
                        prev_text_pos = text_pos[i-1]
                        if pos - prev_text_pos < min_distance:
                            # Adjust position to avoid overlap
                            adjusted_pos = prev_text_pos + min_distance
                            text_pos.append(adjusted_pos)
                        else:
                            # Use original position
                            text_pos.append(pos)
                
                # Plot annotations
                for i, ((pos, text), adjusted_pos) in enumerate(zip(anno_data, text_pos)):
                    color = anno_colors[i % len(anno_colors)]

                    # Draw annotation text at adjusted position
                    annotation = ax.annotate(
                        text, 
                        xy=(pos, y_offset + height/2),  # Point to original gene position
                        xytext=(adjusted_pos, anno_base_y),  # Text at adjusted position
                        fontsize=anno_fontsize,
                        ha='center', 
                        va='bottom',
                        rotation='vertical',
                        color=color,
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                edgecolor=color, 
                                alpha=0.8),
                        arrowprops=dict(
                            arrowstyle='-', 
                            color=color, 
                            alpha=0.7,
                        )
                    )
                    
                    # Add a small marker at the original variant position
                    ax.plot(pos, y_offset + height/2, 
                           marker='v', 
                           markersize=4, 
                           color=color, 
                           markeredgecolor='white',
                           markeredgewidth=0.5,
                           zorder=5)


        ax.set_xlabel(f'Position of {chromosome} ({unit.upper()})', fontsize=14)
        ax.set_xlim(chromo_start, chromo_end)  # Set x-axis limits
        
        # Adjust y-axis limits to accommodate annotations
        y_min = min(gene_track.values()) - track_params['height'] * 0.75
        y_max = max(gene_track.values()) + track_params['height'] / 2
        ax.set_ylim(y_min, y_max)  # Set y-axis limits
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))  # Format x-axis labels
        ax.set_yticks([])  # Hide y-axis ticks
        ax.spines[['top', 'right', 'left']].set_visible(False)  # Hide top, right, and left spines
        ax.tick_params(axis='x', labelsize=12)  # Set x-axis label size

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
            chr_colors = ["#B8B0C3", "#D0E2DF"]  # Default colors

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
            'color': 'gray',
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
                        ax.axhline(-np.log10(threshold), **default_sig_params, label=f"Threshold {threshold:.2e}")
                else:
                    ax.axhline(-np.log10(sig_threshold), **default_sig_params, label=f"Threshold {sig_threshold:.2e}")
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
        else:
            logger.info("No significance threshold provided; skipping threshold line.")

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
        ax.scatter(expected, observed, s=point_size, color="#B8B0C3")
        ax.plot([0, max(expected)], [0, max(expected)], color="gray", linestyle="--", linewidth=1)

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
        ax=None,
        figsize: Tuple[float, float] = (10.0, 6.0),
        dpi: int = 300,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        chrom_order: Optional[List[str]] = None,
        legend_title: str = "Traits",
        legend_kwargs: Optional[Dict[str, object]] = None,
        text_style: Optional[Dict[str, object]] = None):
        """Plot a publication-ready genome-wide QTL distribution map.

        Parameters mirror the previous implementation with additional styling
        controls for figure creation, fonts, legend placement, and annotation
        aesthetics suitable for publication-quality figures.
        """

        # Parameter settings for text
        text_style = (text_style or {}).copy()
        default_text_style = {
            "font_family": "Arial",
            "title_fontsize": 18,
            "subtitle_fontsize": 12,
            "label_fontsize": 12,
            "tick_fontsize": 10
        }
        for key, value in default_text_style.items():
            text_style.setdefault(key, value)
            
        font_family = text_style["font_family"]
        title_fontsize = text_style["title_fontsize"]
        subtitle_fontsize = text_style["subtitle_fontsize"]
        label_fontsize = text_style["label_fontsize"]
        tick_fontsize = text_style["tick_fontsize"]

        with mpl.rc_context({'font.family': font_family}):
            blocks = copy.deepcopy(blocks)
            genome = copy.deepcopy(genome)

            # Data validation
            for df, cols in zip([blocks, genome], [["trait", "chr", "bp1", "bp2"], ["chr", "len"]]):
                missing = [col for col in cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")

            if blocks.empty:
                raise ValueError("No QTL blocks provided for plotting.")

            # Filter chromosomes if requested
            if chrom_names is not None:
                chrom_names = [chrom_names] if isinstance(chrom_names, str) else chrom_names
                chrom_names = [str(c) for c in chrom_names]
                blocks = blocks[blocks["chr"].astype(str).isin(chrom_names)].copy()
                genome = genome[genome["chr"].astype(str).isin(chrom_names)].copy()
                if genome.empty:
                    raise ValueError("No chromosomes remain after filtering")

            unit_factors = {'mb': 1e6, 'kb': 1e3, 'bp': 1}
            if unit not in unit_factors:
                raise ValueError(f"Invalid unit: {unit}, choose from 'mb', 'kb', 'bp'")

            factor = unit_factors[unit]
            blocks["chr"] = blocks["chr"].astype(str)
            genome["chr"] = genome["chr"].astype(str)
            blocks["bp1"] = blocks["bp1"] / factor
            blocks["bp2"] = blocks["bp2"] / factor
            genome["len"] = genome["len"] / factor

            # Determine chromosome order
            def _natural_sort_key(value: str):
                parts = re.split(r'(\d+)', value)
                return [int(p) if p.isdigit() else p.lower() for p in parts]

            available_chroms = genome["chr"].unique().tolist()
            if chrom_order is not None:
                chrom_order = [str(c) for c in chrom_order if str(c) in available_chroms]
                if not chrom_order:
                    raise ValueError("Provided chrom_order does not match any chromosomes in genome data")
                order = chrom_order
                genome = genome[genome["chr"].isin(order)]
            else:
                order = sorted(available_chroms, key=_natural_sort_key)

            cat_type = pd.api.types.CategoricalDtype(order, ordered=True)
            genome["chr"] = genome["chr"].astype(cat_type)
            genome = genome.sort_values("chr").reset_index(drop=True)
            blocks = blocks[blocks["chr"].isin(order)].copy()
            blocks["chr"] = blocks["chr"].astype(cat_type)
            blocks = blocks.sort_values(["chr", "bp1", "bp2"]).reset_index(drop=True)

            if blocks.empty:
                raise ValueError("No QTL blocks remain after chromosome filtering")

            # Parameter settings for chromosomes
            chrom_style = (chrom_style or {}).copy()
            default_chrom_style = {
                "width": 0.8,
                "margin": 0.2,
                "spacing": 0.5,
                "rounding": 0.3,
                "facecolor": "#ECECEC",
                "edgecolor": "#303030",
                "edge_width": 0.8,
                "inner_padding_ratio": 0.2
            }
            for key, value in default_chrom_style.items():
                chrom_style.setdefault(key, value)

            # Resolve trait colors
            traits = list(pd.unique(blocks["trait"].astype(str)))

            def _resolve_trait_colors(traits_list, color_spec):
                if not traits_list:
                    return {}
                if isinstance(color_spec, dict):
                    return {trait: color_spec.get(trait, mpl.colormaps.get_cmap("tab20")(i % 20))
                            for i, trait in enumerate(traits_list)}
                if isinstance(color_spec, (list, tuple)):
                    palette = list(color_spec)
                    if len(palette) < len(traits_list):
                        cmap = mpl.colormaps.get_cmap("tab20")
                        palette.extend([cmap(i / max(1, len(traits_list) - 1))
                                        for i in range(len(palette), len(traits_list))])
                    return {trait: palette[i] for i, trait in enumerate(traits_list)}
                cmap_name = color_spec if isinstance(color_spec, str) else ("Set2" if len(traits_list) <= 8 else "tab20")
                cmap = mpl.colormaps.get_cmap(cmap_name)
                if len(traits_list) == 1:
                    return {traits_list[0]: cmap(0.5)}
                return {trait: cmap(i / (len(traits_list) - 1)) for i, trait in enumerate(traits_list)}

            trait_color_map = _resolve_trait_colors(traits, colors)

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            else:
                fig = ax.figure

            # Layout helpers assume ordered genome
            def _draw_horizontal_layout(blocks_df, genome_df, params, color_map, unit_label, axis):
                positions = []
                current_pos = params["margin"]
                for _ in range(len(genome_df)):
                    positions.append(current_pos)
                    current_pos += params["width"] + params["spacing"]

                max_len = genome_df["len"].max()
                
                inner_padding_ratio = params.get("inner_padding_ratio", 0.2)
                inner_width = params["width"] * (1 - 2 * inner_padding_ratio)
                inner_offset = params["width"] * inner_padding_ratio

                for x_pos, (_, chrom_row) in zip(positions, genome_df.iterrows()):
                    chrom = chrom_row["chr"]
                    length = chrom_row["len"]
                    
                    # Outer chromosome boundary
                    axis.add_patch(FancyBboxPatch(
                        (x_pos, 0), params["width"], length,
                        boxstyle=f"round,pad=0,rounding_size={params['rounding']}",
                        facecolor="none",
                        edgecolor=params["edgecolor"],
                        linewidth=params["edge_width"],
                        alpha=1.0
                    ))
                    
                    # Inner track
                    axis.add_patch(FancyBboxPatch(
                        (x_pos + inner_offset, 0), inner_width, length,
                        boxstyle=f"round,pad=0,rounding_size={params['rounding'] * (1 - inner_padding_ratio)}",
                        facecolor=params["facecolor"],
                        edgecolor="none",
                        linewidth=0,
                        alpha=0.75
                    ))

                    chrom_qtls = blocks_df[blocks_df["chr"] == chrom]
                    for _, qtl in chrom_qtls.iterrows():
                        axis.add_patch(Rectangle(
                            (x_pos + inner_offset, qtl["bp1"]), inner_width, max(qtl["bp2"] - qtl["bp1"], 1e-6),
                            facecolor=color_map.get(str(qtl["trait"]), "#38638D"),
                            edgecolor=color_map.get(str(qtl["trait"]), "#38638D"),
                            linewidth=0.6,
                            alpha=0.9
                        ))

                axis.set_xlim(0, positions[-1] + params["width"] + params["spacing"])
                axis.set_ylim(0, max_len * 1.02)
                axis.spines[['top', 'bottom', 'right']].set_visible(False)
                axis.set_xticks([x + params["width"] / 2 for x in positions])
                axis.set_xticklabels(genome_df["chr"].astype(str), fontsize=tick_fontsize)
                axis.tick_params(axis='x', which='major', color='white', length=0)
                axis.set_xlabel("Chromosome", fontsize=label_fontsize)
                axis.set_ylabel(f"Genomic Position ({unit_label.upper()})", fontsize=label_fontsize)

            def _draw_vertical_layout(blocks_df, genome_df, params, color_map, unit_label, axis):
                positions = []
                current_pos = params["margin"]
                for _ in range(len(genome_df)):
                    positions.append(current_pos)
                    current_pos += params["width"] + params["spacing"]

                max_len = genome_df["len"].max()
                
                inner_padding_ratio = params.get("inner_padding_ratio", 0.2)
                inner_width = params["width"] * (1 - 2 * inner_padding_ratio)
                inner_offset = params["width"] * inner_padding_ratio

                for y_pos, (_, chrom_row) in zip(positions, genome_df.iterrows()):
                    chrom = chrom_row["chr"]
                    length = chrom_row["len"]
                    
                    # Outer chromosome boundary
                    axis.add_patch(FancyBboxPatch(
                        (0, y_pos), length, params["width"],
                        boxstyle=f"round,pad=0,rounding_size={params['rounding']}",
                        facecolor="none",
                        edgecolor=params["edgecolor"],
                        linewidth=params["edge_width"],
                        alpha=1.0
                    ))
                    
                    # Inner track
                    axis.add_patch(FancyBboxPatch(
                        (0, y_pos + inner_offset), length, inner_width,
                        boxstyle=f"round,pad=0,rounding_size={params['rounding'] * (1 - inner_padding_ratio)}",
                        facecolor=params["facecolor"],
                        edgecolor="none",
                        linewidth=0,
                        alpha=0.75
                    ))

                    chrom_qtls = blocks_df[blocks_df["chr"] == chrom]
                    for _, qtl in chrom_qtls.iterrows():
                        axis.add_patch(Rectangle(
                            (qtl["bp1"], y_pos + inner_offset), max(qtl["bp2"] - qtl["bp1"], 1e-6), inner_width,
                            facecolor=color_map.get(str(qtl["trait"]), "#38638D"),
                            edgecolor=color_map.get(str(qtl["trait"]), "#38638D"),
                            linewidth=0.6,
                            alpha=0.9
                        ))

                axis.set_ylim(0, positions[-1] + params["width"] + params["spacing"])
                axis.set_xlim(0, max_len * 1.02)
                axis.spines[['top', 'left', 'right']].set_visible(False)
                axis.set_yticks([y + params["width"] / 2 for y in positions])
                axis.set_yticklabels(genome_df["chr"].astype(str), fontsize=tick_fontsize)
                axis.tick_params(axis='y', which='major', color='white', length=0)
                axis.set_ylabel("Chromosome", fontsize=label_fontsize)
                axis.set_xlabel(f"Genomic Position ({unit_label.upper()})", fontsize=label_fontsize)

            if orientation == "horizontal":
                _draw_horizontal_layout(blocks, genome, chrom_style, trait_color_map, unit, ax)
            elif orientation == "vertical":
                _draw_vertical_layout(blocks, genome, chrom_style, trait_color_map, unit, ax)
            else:
                raise ValueError("Invalid orientation, choose 'horizontal' or 'vertical'")

            ax.tick_params(axis='both', labelsize=tick_fontsize)

            if title:
                ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=18)
            if subtitle:
                ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, fontsize=subtitle_fontsize,
                        ha='center', va='bottom')

            if trait_color_map:
                legend_patches = [Patch(color=color, label=str(trait)) for trait, color in trait_color_map.items()]
                legend_params = {
                    "bbox_to_anchor": (1.02, 1),
                    "loc": "upper left",
                    "frameon": False,
                    "borderaxespad": 0.0,
                    "prop": {"size": tick_fontsize}
                }
                if legend_kwargs:
                    legend_params.update(legend_kwargs)
                legend = ax.legend(handles=legend_patches, title=legend_title, **legend_params)
                if legend and legend.get_title() is not None:
                    legend.get_title().set_fontsize(tick_fontsize)

            # Annotation handling
            if ann_data is not None and not ann_data.empty:
                required = {"chr", "bp1", "bp2", "label"}
                if not required.issubset(ann_data.columns):
                    missing = required - set(ann_data.columns)
                    raise ValueError(f"ann_data missing required columns: {sorted(missing)}")

                ann_data = ann_data.copy()
                ann_data["chr"] = ann_data["chr"].astype(str)
                ann_data["bp1"] = ann_data["bp1"] / factor
                ann_data["bp2"] = ann_data["bp2"] / factor
                ann_data = ann_data[ann_data["chr"].isin(order)]
                if not ann_data.empty:
                    ann_style_local = (ann_style or {}).copy()
                    default_ann_style = {
                        'fontsize': 10,
                        'color': 'black',
                        'ha': 'left' if orientation == 'horizontal' else 'center',
                        'va': 'center' if orientation == 'horizontal' else 'bottom',
                        'alpha': 0.9,
                        'clip_on': False,
                        'max_width': 28,
                        'line_spacing': 1.2,
                        'arrowstyle': "-",
                        'boxstyle': "round,pad=0.2",
                        'facecolor': "white",
                        'edgecolor': "none"
                    }
                    for key, value in default_ann_style.items():
                        ann_style_local.setdefault(key, value)

                    arrowprops = ann_style_local.pop('arrowprops', None) or {}
                    arrowprops.setdefault('arrowstyle', ann_style_local.pop('arrowstyle'))
                    arrowprops.setdefault('color', ann_style_local.get('color', 'black'))
                    arrowprops.setdefault('linewidth', 0.8)
                    arrowprops.setdefault('alpha', ann_style_local.get('alpha', 0.9))

                    bbox_style = ann_style_local.pop('bbox', None) or {}
                    bbox_style.setdefault('boxstyle', ann_style_local.pop('boxstyle'))
                    bbox_style.setdefault('facecolor', ann_style_local.pop('facecolor'))
                    bbox_style.setdefault('edgecolor', ann_style_local.pop('edgecolor'))
                    bbox_style.setdefault('linewidth', 0.6)
                    bbox_style.setdefault('alpha', max(0.2, ann_style_local.get('alpha', 0.9) * 0.7))
                    
                    annotation_max_width = ann_style_local.pop('max_width')
                    annotation_line_spacing = ann_style_local.pop('line_spacing')

                    genome_sorted = genome.reset_index(drop=True)
                    if orientation == 'horizontal':
                        positions = []
                        current_pos = chrom_style['margin']
                        for _ in range(len(genome_sorted)):
                            positions.append(current_pos)
                            current_pos += chrom_style['width'] + chrom_style['spacing']
                        chrom_to_x = {row.chr: pos for pos, row in zip(positions, genome_sorted.itertuples())}

                        for chrom, group in ann_data.groupby('chr'):
                            if chrom not in chrom_to_x:
                                continue
                            x_base = chrom_to_x[chrom]
                            x_text = x_base + chrom_style['width'] + chrom_style['margin'] * 0.35
                            group = group.copy()
                            group['y'] = (group['bp1'] + group['bp2']) / 2
                            group['label_wrapped'] = group['label'].apply(lambda text: textwrap.fill(str(text), width=annotation_max_width))
                            group_sorted = group.sort_values('y')

                            y_min, y_max = ax.get_ylim()
                            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                            height_in = bbox.height if bbox.height > 0 else 1
                            data_per_inch = (y_max - y_min) / height_in
                            line_height = ann_style_local['fontsize'] / 72 * data_per_inch * annotation_line_spacing

                            text_heights = [
                                (label.count('\n') + 1) * line_height
                                for label in group_sorted['label_wrapped']
                            ]

                            adjusted_y = []
                            min_spacing = line_height * 0.5
                            for i, y_val in enumerate(group_sorted['y']):
                                if not adjusted_y:
                                    adjusted_y.append(y_val)
                                    continue
                                prev_top = adjusted_y[-1] + text_heights[i - 1] / 2
                                current_bottom = y_val - text_heights[i] / 2
                                if current_bottom < prev_top + min_spacing:
                                    y_val = prev_top + min_spacing + text_heights[i] / 2
                                adjusted_y.append(y_val)

                            for (_, row), y_adj in zip(group_sorted.iterrows(), adjusted_y):
                                ax.annotate(
                                    row['label_wrapped'],
                                    xy=(x_base + chrom_style['width'], row['y']), xycoords='data',
                                    xytext=(x_text, y_adj), textcoords='data',
                                    arrowprops=arrowprops,
                                    bbox=bbox_style,
                                    **ann_style_local
                                )
                    else:
                        positions = []
                        current_pos = chrom_style['margin']
                        for _ in range(len(genome_sorted)):
                            positions.append(current_pos)
                            current_pos += chrom_style['width'] + chrom_style['spacing']
                        chrom_to_y = {row.chr: pos for pos, row in zip(positions, genome_sorted.itertuples())}

                        for chrom, group in ann_data.groupby('chr'):
                            if chrom not in chrom_to_y:
                                continue
                            y_base = chrom_to_y[chrom]
                            y_text = y_base + chrom_style['width'] + chrom_style['margin'] * 0.35
                            group = group.copy()
                            group['x'] = (group['bp1'] + group['bp2']) / 2
                            group['label_wrapped'] = group['label'].apply(lambda text: textwrap.fill(str(text), width=annotation_max_width))
                            group_sorted = group.sort_values('x')

                            x_min, x_max = ax.get_xlim()
                            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                            width_in = bbox.width if bbox.width > 0 else 1
                            data_per_inch = (x_max - x_min) / width_in
                            char_width = ann_style_local['fontsize'] / 72 * data_per_inch * 0.6

                            text_widths = [
                                max(len(line) for line in label.split('\n')) * char_width
                                for label in group_sorted['label_wrapped']
                            ]

                            adjusted_x = []
                            min_spacing = char_width * 2
                            for i, x_val in enumerate(group_sorted['x']):
                                if not adjusted_x:
                                    adjusted_x.append(x_val)
                                    continue
                                prev_right = adjusted_x[-1] + text_widths[i - 1] / 2
                                current_left = x_val - text_widths[i] / 2
                                if current_left < prev_right + min_spacing:
                                    x_val = prev_right + min_spacing + text_widths[i] / 2
                                adjusted_x.append(x_val)

                            for (_, row), x_adj in zip(group_sorted.iterrows(), adjusted_x):
                                ax.annotate(
                                    row['label_wrapped'],
                                    xy=(row['x'], y_base + chrom_style['width']), xycoords='data',
                                    xytext=(x_adj, y_text), textcoords='data',
                                    arrowprops=arrowprops,
                                    bbox=bbox_style,
                                    **ann_style_local
                                )

            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(False)

            return ax

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
        :param ax: Matplotlib Axes object for plotting.
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
        # plot_data = df[columns].dropna() # Don't dropna globally
        plot_data = df[columns]
        
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
        
        ax.set_title(f'{kind.capitalize()} Plot')
        
        # Turn off top and right spines
        ax.spines[['top', 'right']].set_visible(False)
        
        # Add legend for multi-column plots
        if len(columns) > 1 and kind in ['hist', 'kde']:
            ax.legend(frameon=False)

    def plot_box(self, data, columns, colors, alpha, orientation, ax):
        """Plot box plot distribution"""
        # Define mean point properties: white diamond with black edge
        meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=5)
        
        # Prepare data by dropping NaNs for each column individually
        plot_data = [data[col].dropna().values for col in columns]
        
        if orientation == 'vertical':
            box_plot = ax.boxplot(plot_data, 
                                 tick_labels=columns, patch_artist=True,
                                 showmeans=True, meanprops=meanprops)
            # Set colors
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)
            
            # Rotate x-axis labels if too many columns
            if len(columns) > 5:
                ax.tick_params(axis='x', rotation=45)
        else:
            box_plot = ax.boxplot(plot_data, 
                                 tick_labels=columns, vert=False, patch_artist=True,
                                 showmeans=True, meanprops=meanprops)
            # Set colors
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)

    def plot_violin(self, data, columns, colors, alpha, orientation, ax):
        """Plot violin plot distribution"""
        # Prepare data by dropping NaNs for each column individually
        plot_data = [data[col].dropna().values for col in columns]
        
        if orientation == 'vertical':
            parts = ax.violinplot(plot_data, 
                                 positions=range(1, len(columns) + 1),
                                 showmedians=True)
            # Set colors
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
            
            # Set style for lines (median, min, max, bars)
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in parts:
                    parts[partname].set_edgecolor('gray')
                    parts[partname].set_linewidth(1)
            
            ax.set_xticks(range(1, len(columns) + 1))
            ax.set_xticklabels(columns)
            
            # Rotate x-axis labels if too many columns
            if len(columns) > 5:
                ax.tick_params(axis='x', rotation=45)
        else:
            # For horizontal violin plots, we need to transpose the data approach
            parts = ax.violinplot(plot_data, 
                                 positions=range(1, len(columns) + 1), vert=False,
                                 showmedians=True)
            # Set colors
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
            
            # Set style for lines (median, min, max, bars)
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in parts:
                    parts[partname].set_edgecolor('gray')
                    parts[partname].set_linewidth(1)

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

        # Prepare data by dropping NaNs for each column individually
        plot_data = [data[col].dropna().values for col in columns]

        # Plot histogram
        if len(columns) == 1:
            # Single column histogram
            col = columns[0]
            col_data = plot_data[0]
            if orientation == 'vertical':
                n, bins_edges, patches = ax.hist(col_data, bins=bins, 
                                                density=density, alpha=alpha, 
                                                color=colors[0], label=col, edgecolor='white')
                if fit_curve:
                    mu, sigma = norm.fit(col_data)
                    x = np.linspace(col_data.min(), col_data.max(), fit_bins)
                    y = norm.pdf(x, mu, sigma)
                    if not density:
                        y = y * len(col_data) * (bins_edges[1] - bins_edges[0])
                    ax.plot(x, y, '--', linewidth=2, color=colors[0], zorder=10)
            else:
                n, bins_edges, patches = ax.hist(col_data, bins=bins, 
                                                density=density, alpha=alpha, 
                                                orientation='horizontal',
                                                color=colors[0], label=col, edgecolor='white')
                if fit_curve:
                    mu, sigma = norm.fit(col_data)
                    y = np.linspace(col_data.min(), col_data.max(), fit_bins)
                    x = norm.pdf(y, mu, sigma)
                    if not density:
                        x = x * len(col_data) * (bins_edges[1] - bins_edges[0])
                    ax.plot(x, y, '--', linewidth=2, color=colors[0], zorder=10)
        else:
            # Multiple columns histogram
            if hist_style == 'stack':
                if orientation == 'vertical':
                    n, bins_edges, patches = ax.hist(plot_data, bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, stacked=True, edgecolor='white')
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            col_data = plot_data[i]
                            if len(col_data) < 2: continue
                            mu, sigma = norm.fit(col_data)
                            x = np.linspace(col_data.min(), col_data.max(), fit_bins)
                            y = norm.pdf(x, mu, sigma)
                            if not density:
                                y = y * len(col_data) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, color=colors[i], zorder=10)
                else:
                    n, bins_edges, patches = ax.hist(plot_data, bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, orientation='horizontal', stacked=True, edgecolor='white')
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            col_data = plot_data[i]
                            if len(col_data) < 2: continue
                            mu, sigma = norm.fit(col_data)
                            y = np.linspace(col_data.min(), col_data.max(), fit_bins)
                            x = norm.pdf(y, mu, sigma)
                            if not density:
                                x = x * len(col_data) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, color=colors[i], zorder=10)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
            elif hist_style == 'dodge':
                if orientation == 'vertical':
                    n, bins_edges, patches = ax.hist(plot_data, bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, stacked=False, edgecolor='white')
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            col_data = plot_data[i]
                            if len(col_data) < 2: continue
                            mu, sigma = norm.fit(col_data)
                            x = np.linspace(col_data.min(), col_data.max(), fit_bins)
                            y = norm.pdf(x, mu, sigma)
                            if not density:
                                y = y * len(col_data) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, color=colors[i], zorder=10)
                else:
                    n, bins_edges, patches = ax.hist(plot_data, bins=bins, 
                           density=density, alpha=alpha, color=colors, 
                           label=columns, orientation='horizontal', stacked=False, edgecolor='white')
                    # add fit curve for each column
                    if fit_curve:
                        for i, col in enumerate(columns):
                            col_data = plot_data[i]
                            if len(col_data) < 2: continue
                            mu, sigma = norm.fit(col_data)
                            y = np.linspace(col_data.min(), col_data.max(), fit_bins)
                            x = norm.pdf(y, mu, sigma)
                            if not density:
                                x = x * len(col_data) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y, '--', linewidth=2, color=colors[i], zorder=10)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
            elif hist_style == 'side':
                # Create side bar plots
                # Use flattened data for bin edges calculation
                all_data = np.concatenate(plot_data)
                bin_edges = np.histogram_bin_edges(all_data, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bar_width = (bin_edges[1] - bin_edges[0]) / (len(columns) + 1)
                
                if orientation == 'vertical':
                    bottom = 0.0
                    y_locations = []
                    for i, col in enumerate(columns):
                        col_data = plot_data[i]
                        hist, _ = np.histogram(col_data, bins=bin_edges, density=density)
                        ax.bar(bin_centers + i * bar_width - (len(columns)-1) * bar_width / 2, 
                            hist, width=bar_width, alpha=alpha, color=colors[i], label=col, bottom=bottom, edgecolor='white')
                        if fit_curve and len(col_data) >= 2:
                            mu, sigma = norm.fit(col_data)
                            x = np.linspace(col_data.min(), col_data.max(), fit_bins)
                            y = norm.pdf(x, mu, sigma)
                            if not density:
                                y = y * len(col_data) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x, y + bottom, '--', linewidth=2, color=colors[i], zorder=10)
                        bottom += np.max(hist)
                        y_locations.append(bottom - np.max(hist) / 2)
                    ax.set_yticks(y_locations, labels=columns)
                else:
                    left = 0.0
                    x_locations = []
                    for i, col in enumerate(columns):
                        col_data = plot_data[i]
                        hist, _ = np.histogram(col_data, bins=bin_edges, density=density)
                        ax.barh(bin_centers + i * bar_width - (len(columns)-1) * bar_width / 2, 
                            hist, height=bar_width, alpha=alpha, color=colors[i], label=col, left=left, edgecolor='white')
                        if fit_curve and len(col_data) >= 2:
                            mu, sigma = norm.fit(col_data)
                            y = np.linspace(col_data.min(), col_data.max(), fit_bins)
                            x = norm.pdf(y, mu, sigma)
                            if not density:
                                x = x * len(col_data) * (bins_edges[1] - bins_edges[0])
                            ax.plot(x + left, y, '--', linewidth=2, color=colors[i], zorder=10)
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
                            ax.fill_betweenx(y_range, x_region, alpha=alpha*0.3, color=colors[i])
                            
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

    def plot_qtn_map(
        self,
        vcf_path: str,
        qtl_blocks: pd.DataFrame,
        samples_file: Optional[str] = None,
        by: str = "trait",
        sample_metric: str = "mis",
        variant_metric: str = "maf",
        figsize: Tuple[float, float] = (12.0, 10.0),
        out_path: Optional[str] = None
    ):
        """
        Visualize genotype data for significant QTL sites using a heatmap.

        :param vcf_path: Path to VCF file.
        :param qtl_blocks: DataFrame containing QTL blocks information.
        :param samples_file: Path to samples file (2 columns: sample, group).
        :param by: Group variants by 'trait' or 'chr'.
        :param sample_metric: Metric for sample statistics ('mis', 'het', 'count', 'count0', 'count1', 'count2').
        :param variant_metric: Metric for variant statistics ('mis', 'het', 'maf', 'count', 'count0', 'count1', 'count2').
        :param figsize: Figure size.
        :param out_path: Output path for the figure.
        """
        import pysam
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.gridspec import GridSpec

        logger.info("Starting QTN map plotting...")

        # 1. Process Samples
        vf_temp = pysam.VariantFile(vcf_path)
        all_samples = list(vf_temp.header.samples)
        vf_temp.close()
        
        sample_groups = {}
        phenotype_data = pd.DataFrame()

        if samples_file:
            if not os.path.exists(samples_file):
                raise FileNotFoundError(f"Samples file not found: {samples_file}")
            
            # Read samples file with header
            try:
                df_samples = pd.read_csv(samples_file, sep=r'\s+')
            except Exception as e:
                raise ValueError(f"Error reading samples file: {e}")

            if df_samples.empty:
                raise ValueError("Samples file is empty.")

            # First column is sample name
            sample_col = df_samples.columns[0]
            # Second column is group (optional)
            group_col = df_samples.columns[1] if len(df_samples.columns) > 1 else None
            # Remaining columns are phenotypes
            phe_cols = df_samples.columns[2:] if len(df_samples.columns) > 2 else []

            # Create map for groups
            if group_col:
                # Ensure strings
                sample_groups = dict(zip(df_samples[sample_col].astype(str), df_samples[group_col].astype(str)))
            else:
                sample_groups = {str(s): "All" for s in df_samples[sample_col]}
            
            # Store phenotype data
            if len(phe_cols) > 0:
                df_samples[sample_col] = df_samples[sample_col].astype(str)
                phenotype_data = df_samples.set_index(sample_col)[phe_cols]

            # Filter samples present in VCF
            valid_samples = [s for s in sample_groups.keys() if s in all_samples]
            if not valid_samples:
                raise ValueError("No matching samples found in VCF.")
            
            # Sort samples by group then name
            valid_samples.sort(key=lambda x: (sample_groups[x], x))
            ordered_samples = valid_samples
            
            # Reorder phenotype data
            if not phenotype_data.empty:
                phenotype_data = phenotype_data.reindex(ordered_samples)
                
                # Ensure numeric types
                phenotype_data = phenotype_data.apply(pd.to_numeric, errors='coerce')
                
                # Normalize phenotypes to 0-1
                min_vals = phenotype_data.min()
                max_vals = phenotype_data.max()
                range_vals = max_vals - min_vals
                
                # Avoid division by zero where max == min
                range_vals[range_vals == 0] = 1.0
                
                phenotype_data = (phenotype_data - min_vals) / range_vals
        else:
            ordered_samples = sorted(all_samples)
            sample_groups = {s: "All" for s in ordered_samples}

        # 2. Process Variants (QTL Blocks)
        if "ids" not in qtl_blocks.columns:
            raise ValueError("QTL blocks must contain 'ids' column.")
        
        if by not in qtl_blocks.columns:
             raise ValueError(f"Column '{by}' not found in QTL blocks for grouping.")

        variant_list = []
        for _, row in qtl_blocks.iterrows():
            group_val = row[by]
            chrom = str(row["chr"])
            ids = str(row["ids"]).split(";")
            
            # Try to get positions if available
            if "bps" in row:
                bps = str(row["bps"]).split(";")
            else:
                # Fallback if bps not present (should not happen based on requirements)
                bps = ["0"] * len(ids)
            
            # Zip ids and bps. If lengths mismatch, zip truncates, which is safer than crashing but might lose data.
            # Assuming data integrity from upstream.
            for snp_id, snp_pos in zip(ids, bps):
                if snp_id.strip():
                    try:
                        pos_int = int(snp_pos)
                    except ValueError:
                        pos_int = 0
                        
                    variant_list.append({
                        "group": group_val, 
                        "id": snp_id.strip(),
                        "_chrom": chrom,
                        "_pos": pos_int
                    })
        
        if not variant_list:
            raise ValueError("No variants found in QTL blocks.")

        variants_df = pd.DataFrame(variant_list)
        # Remove duplicates within groups if any, but keep duplicates across groups if a SNP is in multiple traits
        variants_df = variants_df.drop_duplicates(subset=["group", "id"])
        
        # Ensure group is string for sorting
        variants_df["group"] = variants_df["group"].astype(str)
        
        # Sort
        variants_df = variants_df.sort_values(by=["group", "_chrom", "_pos"])
        
        ordered_variants = variants_df.to_dict("records")

        # 3. Fetch Genotypes

        # Prepare matrix
        n_variants = len(ordered_variants)
        n_samples = len(ordered_samples)
        genotype_matrix = np.full((n_variants, n_samples), -1, dtype=int) # -1 for missing

        # Cache VCF access
        vf = pysam.VariantFile(vcf_path)
        
        logger.info(f"Fetching genotypes for {n_variants} variants and {n_samples} samples...")
        
        # Create a map of ID -> list of row indices
        id_to_rows = {}
        for idx, v in enumerate(ordered_variants):
            id_to_rows.setdefault(v["id"], []).append(idx)

        # Build fetch targets from ordered_variants directly
        fetch_targets = {} # (chrom, pos) -> list of ids
        
        for v in ordered_variants:
            chrom = v["_chrom"]
            pos = v["_pos"]
            fetch_targets.setdefault((chrom, pos), []).append(v["id"])

        # Fetch
        # Group by chromosome to minimize seek
        chrom_groups = {}
        for (chrom, pos), ids in fetch_targets.items():
            chrom_groups.setdefault(chrom, []).append(pos)

        for chrom, positions in chrom_groups.items():
            positions.sort()
            if not positions:
                continue
            
            # Fetch region covering all
            # We iterate through positions to fetch specific sites
            try:
                for pos in positions:
                    # pysam fetch(chrom, start, end) is 0-based, half-open [start, end)
                    # VCF positions are 1-based.
                    # To fetch a specific 1-based position P, we use fetch(chrom, P-1, P)
                    if pos <= 0:
                        continue
                        
                    for rec in vf.fetch(chrom, pos - 1, pos):
                        # Check if this record matches any of our IDs
                        # We check if the record ID matches one of our target IDs
                        # OR if the position matches (which it should)
                        
                        # Find which IDs match this record
                        matched_ids = fetch_targets.get((chrom, pos), [])
                        
                        # Filter matched_ids by checking if rec.id matches or if we just trust position
                        # The user requested "Directly extract by variant ID", but pysam fetches by position.
                        # If we have multiple variants at same position, we need to distinguish.
                        # If the VCF record has an ID, we can check it.
                        # If the VCF record ID is '.', we might have to rely on position.
                        
                        # If rec.id is None or '.', we can't match by ID.
                        # But we are fetching by position.
                        
                        current_rec_id = rec.id
                        
                        for snp_id in matched_ids:
                            # If record has ID and it matches, or if we just match by position (loose matching)
                            # Strict matching:
                            if current_rec_id and current_rec_id != "." and current_rec_id != snp_id:
                                continue
                                
                            # Fill matrix
                            row_indices = id_to_rows[snp_id]
                            
                            for i, sample in enumerate(ordered_samples):
                                if sample in rec.samples:
                                    gt = rec.samples[sample]["GT"]
                                    
                                    # Inline encoding
                                    code = -1
                                    if gt is not None and not any(x is None for x in gt):
                                        if len(gt) == 2:
                                            if gt[0] == gt[1]:
                                                if gt[0] == 0:
                                                    code = 0 # Hom-Ref
                                                else:
                                                    code = 2 # Hom-Alt
                                            else:
                                                code = 1 # Het
                                        else:
                                            # Handle non-diploid or other cases
                                            # For now treat as missing or try to infer
                                            pass
                                    
                                    # code: 0 (ref), 1 (het), 2 (alt), -1 (missing)
                                    for row_idx in row_indices:
                                        genotype_matrix[row_idx, i] = code
            except ValueError as e:
                logger.warning(f"Error fetching region {chrom} around {positions[0]}: {e}")

        vf.close()

        # 4. Calculate Metrics
        logger.info("Calculating metrics...")
        
        # Sample metrics (columns)
        sample_stats = []
        for i in range(n_samples):
            col = genotype_matrix[:, i]
            total = len(col)
            missing = np.sum(col == -1)
            called = total - missing
            c0 = np.sum(col == 0)
            c1 = np.sum(col == 1)
            c2 = np.sum(col == 2)
            
            stat = {
                "mis": missing / total if total > 0 else 0,
                "het": c1 / called if called > 0 else 0,
                "count": total, # Total sites
                "count0": c0,
                "count1": c1,
                "count2": c2
            }
            sample_stats.append(stat)

        # Variant metrics (rows)
        variant_stats = []
        for i in range(n_variants):
            row = genotype_matrix[i, :]
            total = len(row)
            missing = np.sum(row == -1)
            called = total - missing
            c0 = np.sum(row == 0)
            c1 = np.sum(row == 1)
            c2 = np.sum(row == 2)
            
            # MAF
            # Allele counts: Ref = 2*c0 + c1, Alt = 2*c2 + c1
            ref_alleles = 2*c0 + c1
            alt_alleles = 2*c2 + c1
            total_alleles = ref_alleles + alt_alleles
            maf = min(ref_alleles, alt_alleles) / total_alleles if total_alleles > 0 else 0

            stat = {
                "mis": missing / total if total > 0 else 0,
                "het": c1 / called if called > 0 else 0,
                "maf": maf,
                "count": total,
                "count0": c0,
                "count1": c1,
                "count2": c2
            }
            variant_stats.append(stat)

        # Save metrics to CSV
        if out_path:
            base_out = os.path.splitext(out_path)[0]
            # Remove extension if it's .png/.pdf etc to get base name
            # Actually out_path usually has extension.
            # Let's just append suffix.
            # If out_path is "dir/output.png", base is "dir/output"
            
            # Save sample stats
            sample_stats_df = pd.DataFrame(sample_stats)
            sample_stats_df.insert(0, "sample", ordered_samples)
            sample_stats_df.insert(1, "group", [sample_groups[s] for s in ordered_samples])
            sample_stats_df.to_csv(f"{base_out}.sample_stats.csv", index=False)
            
            # Save variant stats
            variant_stats_df = pd.DataFrame(variant_stats)
            variant_stats_df.insert(0, "id", [v["id"] for v in ordered_variants])
            variant_stats_df.insert(1, "group", [v["group"] for v in ordered_variants])
            variant_stats_df.insert(2, "chr", [v["_chrom"] for v in ordered_variants])
            variant_stats_df.insert(3, "pos", [v["_pos"] for v in ordered_variants])
            variant_stats_df.to_csv(f"{base_out}.variant_stats.csv", index=False)
            
            logger.info(f"Saved statistics to {base_out}.sample_stats.csv and {base_out}.variant_stats.csv")

        # 5. Plotting
        logger.info("Plotting...")
        
        # Define colors for groups
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
        
        def get_group_colors(items, group_key):
            groups = [item[group_key] for item in items]
            unique_groups = sorted(list(set(groups)))
            # Use tab20
            cmap = plt.get_cmap("tab20")
            colors = {}
            for i, g in enumerate(unique_groups):
                colors[g] = mcolors.to_hex(cmap(i % 20))
            return colors, unique_groups

        # Sample Groups Colors
        sample_group_colors, unique_sample_groups = get_group_colors([{"g": sample_groups[s]} for s in ordered_samples], "g")
        
        # Variant Groups Colors
        variant_group_colors, unique_variant_groups = get_group_colors(ordered_variants, "group")

        fig = plt.figure(figsize=figsize)
        
        # Layout:
        #       | Top Plot |
        # Var Bar | Heatmap | Right Plot
        #       | Sample Bar|
        #       | Phe Plots | (Optional)
        
        n_phe = len(phenotype_data.columns)
        height_ratios = [0.2, 1, 0.025] + [0.1] * n_phe
        
        gs = GridSpec(3 + n_phe, 3, width_ratios=[0.025, 1, 0.2], height_ratios=height_ratios, wspace=0.05, hspace=0.05)
        
        ax_top = fig.add_subplot(gs[0, 1])
        ax_variant_bar = fig.add_subplot(gs[1, 0])
        ax_main = fig.add_subplot(gs[1, 1], sharex=ax_top, sharey=ax_variant_bar)
        ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
        ax_sample_bar = fig.add_subplot(gs[2, 1], sharex=ax_main)
        
        phe_axes = []
        for i in range(n_phe):
            ax = fig.add_subplot(gs[3 + i, 1], sharex=ax_main)
            phe_axes.append(ax)
        
        # Colors: Missing(-1)=LightGray, 0=#50BBC3, 1=#90EE90, 2=#1F72B4
        plot_matrix = genotype_matrix.copy()
        plot_matrix = plot_matrix + 1  # Shift to 0,1,2,3  
        
        cmap = ListedColormap(["#E0E0E0", "#50BBC3", "#90EE90", "#1F72B4"])
        norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
        
        ax_main.imshow(plot_matrix, aspect="auto", cmap=cmap, interpolation="nearest", norm=norm)
        
        # Grid lines (White separators)
        # Sample Groups
        current_group = sample_groups[ordered_samples[0]]
        for i, sample in enumerate(ordered_samples):
            grp = sample_groups[sample]
            if grp != current_group:
                ax_main.axvline(x=i - 0.5, color='white', linewidth=1)
                ax_top.axvline(x=i - 0.5, color='white', linewidth=1)
                ax_sample_bar.axvline(x=i - 0.5, color='white', linewidth=1)
                current_group = grp
        
        # Variant Groups
        current_group = ordered_variants[0]["group"]
        for i, v in enumerate(ordered_variants):
            grp = v["group"]
            if grp != current_group:
                ax_main.axhline(y=i - 0.5, color='white', linewidth=1)
                ax_right.axhline(y=i - 0.5, color='white', linewidth=1)
                ax_variant_bar.axhline(y=i - 0.5, color='white', linewidth=1)
                current_group = grp

        # Plot Sample Group Bar
        sample_indices = [unique_sample_groups.index(sample_groups[s]) for s in ordered_samples]
        sample_cmap = ListedColormap([sample_group_colors[g] for g in unique_sample_groups])
        ax_sample_bar.imshow([sample_indices], aspect='auto', cmap=sample_cmap, interpolation='nearest')
        ax_sample_bar.set_axis_off()
        
        # Plot Variant Group Bar
        variant_indices = [unique_variant_groups.index(v["group"]) for v in ordered_variants]
        variant_cmap = ListedColormap([variant_group_colors[g] for g in unique_variant_groups])
        ax_variant_bar.imshow(np.array([variant_indices]).T, aspect='auto', cmap=variant_cmap, interpolation='nearest')
        ax_variant_bar.set_axis_off()

        # Plot Phenotypes
        if n_phe > 0:
            # Use tab10 and cycle if needed
            cmap = plt.get_cmap("tab10")
            phe_colors = [cmap(i % 10) for i in range(n_phe)]
            x_range = range(len(ordered_samples))
            
            for i, col in enumerate(phenotype_data.columns):
                ax = phe_axes[i]
                vals = phenotype_data[col].values
                ax.plot(x_range, vals, color=phe_colors[i], linewidth=1)
                ax.fill_between(x_range, vals, color=phe_colors[i], alpha=0.3)
                ax.set_ylabel(col, rotation=0, ha='right', va='center', fontsize=8)
                ax.set_ylim(0, 1)
                
                # Set Y-axis ticks and labels to show original range
                vmin = min_vals[col]
                vmax = max_vals[col]
                ax.set_yticks([0, 1])
                ax.set_yticklabels([f"{vmin:.1f}", f"{vmax:.1f}"], fontsize=6)
                
                ax.spines[['top', 'right', 'bottom']].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
                
                # Add grid lines for groups
                current_group = sample_groups[ordered_samples[0]]
                for j, sample in enumerate(ordered_samples):
                    grp = sample_groups[sample]
                    if grp != current_group:
                        ax.axvline(x=j - 0.5, color='white', linewidth=1, linestyle='--')
                        current_group = grp

        # Remove ticks from main heatmap
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        
        # Top Plot (Sample Metrics)
        x = range(n_samples)
        if sample_metric == "count":
            c0 = [s["count0"] for s in sample_stats]
            c1 = [s["count1"] for s in sample_stats]
            c2 = [s["count2"] for s in sample_stats]
            ax_top.bar(x, c0, color="#50BBC3", width=1.0, label="Hom-Ref")
            ax_top.bar(x, c1, bottom=c0, color="#90EE90", width=1.0, label="Het")
            ax_top.bar(x, c2, bottom=np.array(c0)+np.array(c1), color="#1F72B4", width=1.0, label="Hom-Alt")
        else:
            vals = [s[sample_metric] for s in sample_stats]
            ax_top.bar(x, vals, color="gray", width=1.0)
        
        ax_top.set_ylabel(sample_metric)
        ax_top.set_xlim(-0.5, n_samples - 0.5)
        ax_top.set_ylim(bottom=0)
        ax_top.tick_params(axis='x', labelbottom=False, bottom=False)
        ax_top.spines[['top', 'right', 'bottom']].set_visible(False)
        
        # Right Plot (Variant Metrics)
        y = range(n_variants)
        if variant_metric == "count":
            c0 = [s["count0"] for s in variant_stats]
            c1 = [s["count1"] for s in variant_stats]
            c2 = [s["count2"] for s in variant_stats]
            ax_right.barh(y, c0, color="#50BBC3", height=1.0, label="Hom-Ref")
            ax_right.barh(y, c1, left=c0, color="#90EE90", height=1.0, label="Het")
            ax_right.barh(y, c2, left=np.array(c0)+np.array(c1), color="#1F72B4", height=1.0, label="Hom-Alt")
        else:
            vals = [s[variant_metric] for s in variant_stats]
            ax_right.barh(y, vals, color="gray", height=1.0)
            
        ax_right.set_xlabel(variant_metric)
        ax_right.set_ylim(-0.5, n_variants - 0.5)
        ax_right.set_xlim(left=0)
        ax_right.invert_yaxis()
        ax_right.tick_params(axis='y', labelleft=False, left=False)
        ax_right.spines[['top', 'right', 'left']].set_visible(False)
        
        # Legends
        # 1. Genotype Legend (Top Right)
        geno_legend_elements = [
            Patch(facecolor='#50BBC3', edgecolor='white', label='Hom-Ref'),
            Patch(facecolor='#90EE90', edgecolor='white', label='Het'),
            Patch(facecolor='#1F72B4', edgecolor='white', label='Hom-Alt'),
            Patch(facecolor='#E0E0E0', edgecolor='white', label='Missing')
        ]
        # Place in the empty subplot at top right (gs[0, 2])
        ax_legend_tr = fig.add_subplot(gs[0, 2])
        ax_legend_tr.axis('off')
        ax_legend_tr.legend(handles=geno_legend_elements, loc='center', ncol=1, frameon=False, title="Genotype", fontsize=8)
        
        # 2. Sample Group Legend (Below Sample Bar or Last Phenotype)
        sample_legend_elements = [Patch(facecolor=sample_group_colors[g], edgecolor='white', label=g) for g in unique_sample_groups]
        
        n_samples_groups = len(unique_sample_groups)
        ncol_sample = min(n_samples_groups, 6)
        
        if n_phe > 0:
            target_ax = phe_axes[-1]
            bbox_y = -0.8
        else:
            target_ax = ax_sample_bar
            bbox_y = -3.0
            
        target_ax.legend(handles=sample_legend_elements, loc='upper center', bbox_to_anchor=(0.5, bbox_y), ncol=ncol_sample, frameon=False, title="Sample Groups")

        # 3. Variant Group Legend (Far Left)
        variant_legend_elements = [Patch(facecolor=variant_group_colors[g], edgecolor='white', label=g) for g in unique_variant_groups]
        # Place to the left of the variant bar
        # We can attach it to ax_variant_bar but place it outside to the left
        ax_variant_bar.legend(handles=variant_legend_elements, loc='center right', bbox_to_anchor=(-0.5, 0.5), frameon=False, title="Loci Groups")
        
        # Adjust layout to make room for legends
        # Left margin for Variant Legend, Bottom margin for Sample Legend
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        
        if out_path:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()