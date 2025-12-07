import os
import re
import math
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from qtlscan.log import logger
from qtlscan.viz import Visualizer
from scipy import stats as sstats


# ------------------------
# Helpers
# ------------------------

def _infer_sep_from_ext(path: str, fallback: str = "\t") -> str:
	lower = (path or "").lower()
	if lower.endswith(".csv"):
		return ","
	# default treat .tsv/.txt as tab
	return "\t"


def _normalize_sep(sep: Optional[str], path: Optional[str]) -> str:
	if sep in (None, "auto"):
		return _infer_sep_from_ext(path or "")
	if sep.lower() in {"csv", ","}:
		return ","
	if sep.lower() in {"tsv", "tab", "\t"}:
		return "\t"
	# allow custom single-char
	return sep


def _read_table(path: str, sep: Optional[str] = None, header: bool = True, encoding: str = "utf-8") -> pd.DataFrame:
	use_sep = _normalize_sep(sep, path)
	try:
		df = pd.read_csv(path, sep=use_sep, header=0 if header else None, encoding=encoding)
	except Exception as e:
		raise ValueError(f"Failed to read table: {path} ({e})")
	return df


def _write_table(df: pd.DataFrame, path: str, sep: Optional[str] = None, header: bool = True, index: bool = False, encoding: str = "utf-8"):
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	use_sep = _normalize_sep(sep, path)
	try:
		df.to_csv(path, sep=use_sep, header=header, index=index, encoding=encoding, line_terminator="\n")
	except TypeError:
		# for older pandas without line_terminator
		df.to_csv(path, sep=use_sep, header=header, index=index, encoding=encoding)


def _write_plink_single(sample_series: pd.Series, value_series: pd.Series, out_path: str):
	"""Write PLINK phenotype: 2 ID columns (FID, IID) and a single phenotype column; tab-delimited; no header.
	FID and IID are both set to the sample id string.
	"""
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as w:
		for s, v in zip(sample_series.astype(str), value_series):
			val = "" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
			w.write(f"{s}\t{s}\t{val}\n")


def _select_trait_columns(df: pd.DataFrame, sample_col: str, traits_regex: Optional[str]) -> List[str]:
	cols = [c for c in df.columns if c != sample_col]
	if traits_regex:
		pat = re.compile(traits_regex)
		cols = [c for c in cols if pat.search(str(c))]
	return cols


# ------------------------
# split
# ------------------------

def phe_split(args):
	"""Split a multi-trait table into per-trait single-column files.

	Input: TXT/TSV/CSV with header; first column or --sample-col is sample id; remaining columns are traits.
	Output formats: txt/tsv/csv or plink (two ID cols + one phenotype; no header).
	"""
	in_path = args.input
	if not os.path.isfile(in_path):
		raise ValueError(f"Input not found: {in_path}")

	df = _read_table(in_path, sep=args.sep, header=True, encoding=args.encoding)
	if df.shape[1] < 2:
		raise ValueError("Input must contain at least 2 columns: sample + >=1 trait")

	sample_col = args.sample_col or df.columns[0]
	if sample_col not in df.columns:
		raise ValueError(f"--sample-col '{args.sample_col}' not found in columns: {list(df.columns)}")

	trait_cols = _select_trait_columns(df, sample_col, args.traits)
	if not trait_cols:
		raise ValueError("No trait columns selected; check --traits pattern")

	out_fmt = args.out_format.lower()
	if out_fmt not in {"txt", "tsv", "csv", "plink"}:
		raise ValueError("--out-format must be one of txt/tsv/csv/plink")

	prefix = args.prefix or os.path.splitext(os.path.basename(in_path))[0]
	out_dir = args.out_dir or "."
	os.makedirs(out_dir, exist_ok=True)

	for trait in trait_cols:
		sub = df[[sample_col, trait]].copy()
		# write
		if out_fmt == "plink":
			ext = "phe"
			out_path = os.path.join(out_dir, f"{prefix}.{trait}.{ext}")
			_write_plink_single(sub[sample_col], sub[trait], out_path)
			logger.info(f"Wrote PLINK phenotype: {out_path}")
		else:
			ext = {"txt": "txt", "tsv": "tsv", "csv": "csv"}[out_fmt]
			out_path = os.path.join(out_dir, f"{prefix}.{trait}.{ext}")
			header = bool(args.header)
			_write_table(sub.rename(columns={sample_col: args.out_sample_col or sample_col}), out_path, sep={"txt": "\t", "tsv": "\t", "csv": ","}[out_fmt], header=header)
			logger.info(f"Wrote phenotype: {out_path}")

	logger.info(f"Split {len(trait_cols)} traits from {in_path}")


# ------------------------
# merge
# ------------------------

def _read_single_trait_file(path: str, sample_col: Optional[str], encoding: str, trait_name: Optional[str]) -> Tuple[pd.DataFrame, str]:
	"""Return (df with columns ['sample', trait_name], trait_name). Detect format by extension.
	- TXT/TSV/CSV: expects header; if multiple non-sample columns, use the first as the trait.
	- PLINK: no header; tab; expect 3 columns: FID IID PHENO; sample taken from IID; trait_name must be provided or derived from filename.
	"""
	lower = path.lower()
	is_plink = lower.endswith(".phe") or lower.endswith(".plink")
	if is_plink:
		dfp = _read_table(path, sep="\t", header=False, encoding=encoding)
		if dfp.shape[1] < 3:
			raise ValueError(f"PLINK file requires at least 3 columns (FID IID PHENO): {path}")
		phen = dfp.iloc[:, 2]
		samp = dfp.iloc[:, 1]  # IID
		tname = trait_name or os.path.splitext(os.path.basename(path))[0]
		out = pd.DataFrame({"sample": samp.astype(str), tname: phen})
		return out, tname
	# table with header
	df = _read_table(path, sep=None, header=True, encoding=encoding)
	s_col = sample_col or df.columns[0]
	if s_col not in df.columns:
		raise ValueError(f"Sample column '{s_col}' not found in {path}")
	non_sample = [c for c in df.columns if c != s_col]
	if not non_sample:
		raise ValueError(f"No trait column found in {path}")
	use_trait = non_sample[0]
	tname = trait_name or use_trait or os.path.splitext(os.path.basename(path))[0]
	out = pd.DataFrame({"sample": df[s_col].astype(str), tname: df[use_trait]})
	return out, tname


def phe_merge(args):
	files: List[str] = args.files
	if not files:
		raise ValueError("--files required")
	for f in files:
		if not os.path.isfile(f):
			raise ValueError(f"File not found: {f}")

	trait_names = None
	if args.trait_names:
		trait_names = [s.strip() for s in args.trait_names.split(",") if s.strip()]
		if trait_names and len(trait_names) != len(files):
			raise ValueError("--trait-names must match number of --files")

	dfs: List[pd.DataFrame] = []
	colnames: List[str] = []
	for i, path in enumerate(files):
		tname = trait_names[i] if trait_names else None
		sub, t_actual = _read_single_trait_file(path, sample_col=args.sample_col, encoding=args.encoding, trait_name=tname)
		dfs.append(sub)
		colnames.append(t_actual)

	# outer-join on sample
	merged = dfs[0]
	for sub in dfs[1:]:
		merged = pd.merge(merged, sub, on="sample", how="outer")

	# write
	out_fmt = args.out_format.lower()
	if out_fmt not in {"txt", "tsv", "csv"}:
		raise ValueError("merge only supports txt/tsv/csv outputs")
	out_dir = args.out_dir or "."
	os.makedirs(out_dir, exist_ok=True)
	out_name = args.out_name or "phe_merged"
	ext = {"txt": "txt", "tsv": "tsv", "csv": "csv"}[out_fmt]
	out_path = os.path.join(out_dir, f"{out_name}.{ext}")
	sample_out_col = args.out_sample_col or "sample"
	merged = merged.rename(columns={"sample": sample_out_col})
	_write_table(merged, out_path, sep={"txt": "\t", "tsv": "\t", "csv": ","}[out_fmt], header=True)
	logger.info(f"Merged {len(files)} files into: {out_path}")


# ------------------------
# stat
# ------------------------

def _summarize_series(s: pd.Series) -> dict:
	s_num = pd.to_numeric(s, errors="coerce")
	n = int(s_num.shape[0])
	n_miss = int(s_num.isna().sum())
	n_notna = n - n_miss
	desc = s_num.describe(percentiles=[0.25, 0.5, 0.75])
	mean = float(desc["mean"]) if "mean" in desc and not math.isnan(desc["mean"]) else float("nan")
	std = float(desc["std"]) if "std" in desc and not math.isnan(desc["std"]) else float("nan")
	q1 = float(desc.get("25%", float("nan")))
	med = float(desc.get("50%", float("nan")))
	q3 = float(desc.get("75%", float("nan")))
	mn = float(desc.get("min", float("nan")))
	mx = float(desc.get("max", float("nan")))
	uniq = int(s_num.nunique(dropna=True))
	skew = float(s_num.skew()) if n_notna > 2 else float("nan")
	kurt = float(s_num.kurt()) if n_notna > 3 else float("nan")

	# additional robust statistics
	s_clean = s_num.dropna()
	n_clean = int(s_clean.shape[0])
	iqr = float((s_clean.quantile(0.75) - s_clean.quantile(0.25))) if n_clean >= 2 else float("nan")
	mad = float((s_clean - s_clean.median()).abs().median()) if n_clean >= 1 else float("nan")
	mad_scaled = float(1.4826 * mad) if not math.isnan(mad) else float("nan")
	cv = float(std / abs(mean)) if (not math.isnan(std) and not math.isnan(mean) and abs(mean) > 1e-12) else float("nan")

	# normality tests
	# D’Agostino’s K^2 requires n >= 8
	if n_clean >= 8:
		try:
			nt_stat, nt_p = sstats.normaltest(s_clean.values)
			normaltest_stat = float(nt_stat)
			normaltest_p = float(nt_p)
		except Exception:
			normaltest_stat = float("nan")
			normaltest_p = float("nan")
	else:
		normaltest_stat = float("nan")
		normaltest_p = float("nan")

	# Shapiro-Wilk: practical range 3 <= n <= 5000 to avoid warnings
	if 3 <= n_clean <= 5000:
		try:
			sh_stat, sh_p = sstats.shapiro(s_clean.values)
			shapiro_stat = float(sh_stat)
			shapiro_p = float(sh_p)
		except Exception:
			shapiro_stat = float("nan")
			shapiro_p = float("nan")
	else:
		shapiro_stat = float("nan")
		shapiro_p = float("nan")

	# Jarque-Bera
	if n_clean >= 2:
		try:
			jb_stat, jb_p = sstats.jarque_bera(s_clean.values)
			jb_stat = float(jb_stat)
			jb_p = float(jb_p)
		except Exception:
			jb_stat = float("nan")
			jb_p = float("nan")
	else:
		jb_stat = float("nan")
		jb_p = float("nan")

	return {
		"count": int(n),
		"non_missing": int(n_notna),
		"missing": int(n_miss),
		"missing_rate": (n_miss / n) if n > 0 else float("nan"),
		"mean": mean,
		"std": std,
		"cv": cv,
		"min": mn,
		"q1": q1,
		"median": med,
		"q3": q3,
		"iqr": iqr,
		"mad": mad,
		"mad_scaled": mad_scaled,
		"max": mx,
		"unique": uniq,
		"skew": skew,
		"kurtosis": kurt,
		"normaltest_stat": normaltest_stat,
		"normaltest_p": normaltest_p,
		"shapiro_stat": shapiro_stat,
		"shapiro_p": shapiro_p,
		"jb_stat": jb_stat,
		"jb_p": jb_p,
	}


def phe_stat(args):
	in_path = args.input
	if not os.path.isfile(in_path):
		raise ValueError(f"Input not found: {in_path}")

	# read input: support regular table or PLINK
	is_plink = (args.format_in == "plink") or in_path.lower().endswith((".phe", ".plink"))
	if is_plink:
		dfp = _read_table(in_path, sep="\t", header=False, encoding=args.encoding)
		if dfp.shape[1] < 3:
			raise ValueError("PLINK file must have at least 3 columns (FID IID PHENO[...])")
		sample = dfp.iloc[:, 1].astype(str)
		traits_df = dfp.iloc[:, 2:]
		traits_df.columns = [f"trait_{i+1}" for i in range(traits_df.shape[1])]
		df = pd.concat([sample.rename("sample"), traits_df], axis=1)
		sample_col = "sample"
	else:
		df = _read_table(in_path, sep=args.sep, header=True, encoding=args.encoding)
		sample_col = args.sample_col or df.columns[0]
		if sample_col not in df.columns:
			raise ValueError(f"--sample-col '{sample_col}' not found in columns")

	trait_cols = _select_trait_columns(df, sample_col, args.columns)
	if not trait_cols:
		raise ValueError("No trait columns selected for statistics")

	# build stats table
	rows = []
	for col in trait_cols:
		stat = _summarize_series(df[col])
		rows.append({"trait": col, **stat})
	stat_df = pd.DataFrame(rows)

	# write stats file
	out_dir = args.out_dir or "."
	os.makedirs(out_dir, exist_ok=True)
	out_name = args.out_name or "phe_stat"
	stats_path = args.stats_file or os.path.join(out_dir, f"{out_name}.stats.tsv")
	_write_table(stat_df, stats_path, sep="\t", header=True)
	logger.info(f"Phenotype statistics saved to: {stats_path}")

	# plot
	try:
		viz = Visualizer()
		import matplotlib.pyplot as plt
		fig = plt.figure(figsize=(args.width, args.height))
		ax = fig.add_subplot(111)
		viz.plot_dist(
			df=df[[c for c in trait_cols]],
			kind=args.plot_kind,
			columns=None,  # use all selected trait_cols
			colors=args.colors,
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
			ax=ax,
		)
		fig_path = os.path.join(out_dir, f"{out_name}.{args.format}")
		plt.tight_layout()
		plt.savefig(fig_path, dpi=300, bbox_inches="tight")
		plt.close()
		logger.info(f"Phenotype distribution figure saved to: {fig_path}")
	except Exception as e:
		logger.warning(f"Plotting failed: {e}")

