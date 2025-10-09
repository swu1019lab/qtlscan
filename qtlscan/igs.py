from __future__ import annotations
import os
import json
import math
import random
import importlib
import time
from typing import List, Tuple, Optional
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qtlscan.log import logger


def _read_phenotype(path: str, sample_col: Optional[str]):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        sep = "\t" if os.path.splitext(path)[1].lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)
    if df.shape[1] < 2:
        raise ValueError("Phenotype file must contain at least 2 columns (sample + trait columns)")
    if sample_col is None:
        sample_col = df.columns[0]
    if sample_col not in df.columns:
        raise ValueError(f"Sample column '{sample_col}' not found in phenotype file")
    return df, sample_col


def _vcf_samples(vcf_path: str) -> List[str]:
	try:
		pysam = importlib.import_module('pysam')
	except Exception as e:
		raise RuntimeError("pysam is required to read VCF; please install pysam") from e
	with pysam.VariantFile(vcf_path) as vf:
		return list(vf.header.samples)


def _encode_gt_tuple(gt) -> int:
	if gt is None:
		return -1
	if any(a is None for a in gt):
		return -1
	alle = list(gt)
	if len(alle) == 0:
		return -1
	if all(a == 0 for a in alle):
		return 0
	if len(set(alle)) == 1 and alle[0] != 0:
		return 2
	return 1


def _reservoir_pick_indices(iterable_len_est: int, k: int, seed: int) -> set:
	rnd = random.Random(seed)
	picked = []
	for i in range(iterable_len_est):
		if i < k:
			picked.append(i)
		else:
			j = rnd.randrange(i + 1)
			if j < k:
				picked[j] = i
	return set(picked)


def _maf_and_missing(col: np.ndarray) -> Tuple[float, float]:
	mask = col >= 0
	called = col[mask]
	miss_rate = 1.0 - (len(called) / len(col) if len(col) else 0.0)
	if len(called) == 0:
		return 0.0, miss_rate
	# g in {0,1,2}
	p_alt = called.mean() / 2.0
	maf = min(p_alt, 1.0 - p_alt)
	return float(maf), float(miss_rate)


def _build_genotype_matrix(
	vcf_path: str,
	samples: List[str],
	variant_selection: str,
	random_variants: Optional[int],
	variant_ids_file: Optional[str],
	maf_min: float,
	max_missing: float,
	seed: int,
) -> Tuple[np.ndarray, List[str]]:
	try:
		pysam = importlib.import_module('pysam')
	except Exception as e:
		raise RuntimeError("pysam is required to read VCF; please install pysam") from e
	logger.info("Reading VCF and building genotype matrix...")
	vf = pysam.VariantFile(vcf_path)
	sample_index = {s: i for i, s in enumerate(list(vf.header.samples))}
	sel_idx = [sample_index[s] for s in samples]

	pick_ids: Optional[set] = None
	if variant_selection == "ids":
		if not variant_ids_file:
			raise ValueError("--variant-ids-file is required when --variant-selection=ids")
		with open(variant_ids_file, "r", encoding="utf-8") as f:
			pick_ids = {line.strip() for line in f if line.strip()}

	# First pass: decide which variants to include (indices or IDs)
	picked_local: Optional[set] = None
	if variant_selection == "random":
		if random_variants is None or random_variants <= 0:
			raise ValueError("--random-variants must be >0 when variant-selection=random")
		# estimate by scanning once to count
		total = 0
		for _ in vf:
			total += 1
		picked_local = _reservoir_pick_indices(total, random_variants, seed)
		vf.reset()

	X_cols: List[np.ndarray] = []
	variant_ids: List[str] = []
	keep_idx = 0
	for i, rec in enumerate(vf):
		rid = rec.id or f"{rec.chrom}_{rec.pos}_{rec.ref}_{rec.alts[0]}"
		if variant_selection == "ids":
			if pick_ids is not None and rid not in pick_ids:
				continue
		elif variant_selection == "random":
			if picked_local is not None and i not in picked_local:
				continue
		# build genotype vector for selected samples
		col = np.full(len(samples), -1, dtype=np.int8)
		for j, sidx in enumerate(sel_idx):
			call = rec.samples[sidx]
			gt = call.get("GT", None)
			col[j] = _encode_gt_tuple(gt)
		# QC
		maf, miss = _maf_and_missing(col.astype(np.int16))
		if maf < maf_min or miss > max_missing:
			continue
		X_cols.append(col)
		variant_ids.append(rid)
		keep_idx += 1
	vf.close()

	if not X_cols:
		raise RuntimeError("No variants passed selection/QC; try lower --maf-min or higher --max-missing, or change selection.")
	X = np.stack(X_cols, axis=1).astype(np.float32)
	logger.info(f"Variants kept after selection/QC: {len(variant_ids)}")
	return X, variant_ids


def _sanitize_feature_names(names: List[str]) -> List[str]:
	"""Return LightGBM-safe feature names, replacing disallowed characters and ensuring uniqueness.

	Rules:
	- Keep [A-Za-z0-9_] only; replace others with underscore.
	- If empty after sanitizing, use f_#.
	- Ensure uniqueness by appending _2, _3, ... if collisions occur.
	"""
	safe = []
	seen = set()
	for i, n in enumerate(names):
		s = re.sub(r"[^A-Za-z0-9_]", "_", str(n))
		if not s:
			s = f"f_{i}"
		base = s
		k = 1
		while s in seen:
			k += 1
			s = f"{base}_{k}"
		seen.add(s)
		safe.append(s)
	return safe


def _make_preprocess(impute: str, scale: str, for_svm: bool):
	from sklearn.pipeline import Pipeline
	from sklearn.impute import SimpleImputer
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	steps = []
	strategy = impute
	steps.append(("imputer", SimpleImputer(strategy=strategy)))
	if scale == "standard" or for_svm:
		steps.append(("scaler", StandardScaler(with_mean=True)))
	elif scale == "minmax":
		steps.append(("scaler", MinMaxScaler()))
	return Pipeline(steps)


def _make_model(name: str, target_type: str, params: Optional[dict]):
	from sklearn.svm import SVR, SVC
	from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
	from sklearn.linear_model import Lasso
	from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
	model = None
	if name == "svm":
		model = SVR() if target_type == "reg" else SVC(probability=True)
	elif name == "lgb":
		try:
			lightgbm = importlib.import_module('lightgbm')
		except Exception as e:
			raise RuntimeError("LightGBM not installed; please install lightgbm") from e
		base = dict(random_state=0, verbose=-1, verbosity=-1, force_row_wise=True)
		model = lightgbm.LGBMRegressor(**base) if target_type == "reg" else lightgbm.LGBMClassifier(**base)
	elif name == "xgb":
		try:
			xgboost = importlib.import_module('xgboost')
		except Exception as e:
			raise RuntimeError("XGBoost not installed; please install xgboost") from e
		common = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0)
		model = xgboost.XGBRegressor(**common, max_depth=6, tree_method="hist") if target_type == "reg" else xgboost.XGBClassifier(**common, max_depth=6, tree_method="hist", eval_metric="logloss")
	elif name == "rf":
		model = RandomForestRegressor(random_state=0, n_estimators=300) if target_type == "reg" else RandomForestClassifier(random_state=0, n_estimators=300)
	elif name == "lasso":
		if target_type != "reg":
			raise ValueError("lasso only supports regression")
		model = Lasso(alpha=0.001, random_state=0)
	elif name == "knn":
		model = KNeighborsRegressor(n_neighbors=5) if target_type == "reg" else KNeighborsClassifier(n_neighbors=5)
	else:
		raise ValueError(f"Unsupported model: {name}")
	if params:
		model.set_params(**params)
	return model


def _default_scoring(target_type: str, n_classes: Optional[int]) -> str:
	if target_type == "reg":
		return "r2"
	if n_classes == 2:
		return "roc_auc"
	return "accuracy"


def _tune_model(model, X, y, target_type: str, tune: str, n_iter: int, param_grid: Optional[dict], param_space: Optional[dict], scoring: Optional[str], cv: int, seed: int):
	if tune == "none":
		return model, None, None
	scoring2 = scoring
	if scoring2 is None:
		n_classes = None
		if target_type == "clf":
			try:
				n_classes = int(len(np.unique(y)))
			except Exception:
				n_classes = None
		scoring2 = _default_scoring(target_type, n_classes)

	if tune == "grid":
		from sklearn.model_selection import GridSearchCV
		gs = GridSearchCV(model, param_grid=param_grid or {}, scoring=scoring2, cv=cv, n_jobs=-1, verbose=1)
		gs.fit(X, y)
		return gs.best_estimator_, gs.best_params_, gs.cv_results_
	elif tune == "random":
		from sklearn.model_selection import RandomizedSearchCV
		rs = RandomizedSearchCV(model, param_distributions=param_space or {}, n_iter=n_iter, scoring=scoring2, cv=cv, n_jobs=-1, random_state=seed, verbose=1)
		rs.fit(X, y)
		return rs.best_estimator_, rs.best_params_, rs.cv_results_
	elif tune == "bayes":
		try:
			skopt = importlib.import_module('skopt')
		except Exception as e:
			raise RuntimeError("BayesSearchCV requires scikit-optimize; please install scikit-optimize") from e
		# param_space 需由 CLI 提供或在上层构造；若为空则回退为无调参
		if not param_space:
			logger.warning("BayesSearchCV param-space is empty; skip tuning")
			return model, None, None
		BayesSearchCV = getattr(skopt, 'BayesSearchCV')
		opt = BayesSearchCV(model, search_spaces=param_space, n_iter=n_iter, scoring=scoring2, cv=cv, n_jobs=-1, random_state=seed, verbose=1)
		opt.fit(X, y)
		return opt.best_estimator_, opt.best_params_, opt.cv_results_
	else:
		return model, None, None


def _plot_regression(y_true, y_pred, trait: str, out_dir: str, out_name: str, fmt: str, figsize: Tuple[float, float]):
	os.makedirs(out_dir, exist_ok=True)
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	ax.scatter(y_true, y_pred, s=15, alpha=0.6, edgecolors='none')
	lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
	ax.plot(lims, lims, 'k--', lw=1)
	from sklearn.metrics import r2_score, mean_absolute_error
	r2 = r2_score(y_true, y_pred)
	mae = mean_absolute_error(y_true, y_pred)
	# Pearson r
	if len(y_true) > 1:
		pr = np.corrcoef(y_true, y_pred)[0, 1]
	else:
		pr = np.nan
	ax.set_title(f"{trait}  R2={r2:.3f}  MAE={mae:.3f}  r={pr:.3f}")
	ax.set_xlabel("Observed")
	ax.set_ylabel("Predicted")
	fig.tight_layout()
	path = os.path.join(out_dir, f"{out_name}.pred_vs_obs.{trait}.{fmt}")
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return path


def _plot_residuals(y_true, y_pred, trait: str, out_dir: str, out_name: str, fmt: str, figsize: Tuple[float, float]):
	res = y_pred - y_true
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	ax.hist(res, bins=30, color="#4C72B0", alpha=0.8)
	ax.set_title(f"Residuals: {trait}")
	ax.set_xlabel("Residual")
	ax.set_ylabel("Count")
	fig.tight_layout()
	path = os.path.join(out_dir, f"{out_name}.residuals.{trait}.{fmt}")
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return path


def _compute_importance_shap(model, X, background_n: int, target_type: str):
	try:
		shap = importlib.import_module('shap')
	except Exception as e:
		raise RuntimeError("SHAP is requested but not installed; please install shap") from e
	rng = np.random.default_rng(0)
	if background_n and X.shape[0] > background_n:
		idx = rng.choice(X.shape[0], size=background_n, replace=False)
		background = X[idx]
	else:
		background = X
	explainer = None
	# Tree models
	tree_types = ("LGBM", "XGB", "RandomForest")
	if any(t in type(model).__name__ for t in tree_types):
		explainer = shap.TreeExplainer(model)
	# Linear
	elif type(model).__name__ in ("Lasso",):
		explainer = shap.LinearExplainer(model, X)
	else:
		explainer = shap.KernelExplainer(model.predict if target_type == "reg" else model.predict_proba, background)
	values = explainer.shap_values(X)
	# For multiclass, choose mean absolute across classes
	if isinstance(values, list):
		vals = np.mean([np.abs(v) for v in values], axis=0)
	else:
		vals = np.abs(values)
	imp = np.mean(vals, axis=0)
	return imp, values


def _plot_importance_bar(features: List[str], scores: np.ndarray, trait: str, out_dir: str, out_name: str, fmt: str, topk: int, figsize: Tuple[float, float]):
	order = np.argsort(scores)[::-1][:topk]
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	ax.barh(range(len(order)), scores[order][::-1], color="#DD8452")
	ax.set_yticks(range(len(order)))
	ax.set_yticklabels([features[i] for i in order][::-1], fontsize=8)
	ax.invert_yaxis()
	ax.set_xlabel("Importance")
	ax.set_title(f"Feature Importance: {trait}")
	fig.tight_layout()
	path = os.path.join(out_dir, f"{out_name}.importance.{trait}.{fmt}")
	fig.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return path


def _plot_shap_summary(values, X, trait: str, out_dir: str, out_name: str, fmt: str, feature_names: Optional[List[str]] = None):
	try:
		shap = importlib.import_module('shap')
	except Exception:
		return None
	# Let shap create the figure; then capture and close it to avoid leaks
	kwargs = {"show": False}
	if feature_names is not None:
		kwargs["feature_names"] = feature_names
	shap.summary_plot(values, X, **kwargs)
	fig = plt.gcf()
	path = os.path.join(out_dir, f"{out_name}.shap_summary.{trait}.{fmt}")
	plt.savefig(path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return path


def _plot_shap_dependence(values, X, feature_names: List[str], topk: int, trait: str, out_dir: str, out_name: str, fmt: str):
	try:
		shap = importlib.import_module('shap')
	except Exception:
		return []
	# compute mean absolute importance for ranking
	if isinstance(values, list):
		vals = np.mean([np.abs(v) for v in values], axis=0)
	else:
		vals = np.abs(values)
	imp = np.mean(vals, axis=0)
	order = np.argsort(imp)[::-1][:topk]
	paths = []
	for idx in order:
		fig = plt.figure(figsize=(6, 5))
		shap.dependence_plot(idx, values, X, feature_names=feature_names, show=False)
		path = os.path.join(out_dir, f"{out_name}.shap_dep.{trait}.feat{idx}.{fmt}")
		plt.savefig(path, dpi=300, bbox_inches="tight")
		plt.close(fig)
		paths.append(path)
	return paths


def train(args):
	# Parse traits
	phe_df, sample_col = _read_phenotype(args.phe, args.sample_col)
	traits = [t.strip() for t in args.traits.split(",")] if args.traits else [c for c in phe_df.columns if c != sample_col]
	if not traits:
		raise ValueError("No trait columns found; specify --traits or provide phenotype columns")

	# Align samples with VCF
	vcf_samples = _vcf_samples(args.vcf)
	phe_df = phe_df[phe_df[sample_col].isin(vcf_samples)].copy()
	phe_df = phe_df.set_index(sample_col)
	if phe_df.empty:
		raise RuntimeError("No overlapping samples between VCF and phenotype file")
	samples = [s for s in vcf_samples if s in phe_df.index]
	logger.info(f"Overlapping samples: {len(samples)}")

	# Build genotype matrix with selection & QC
	logger.info(
		f"Variant selection: {args.variant_selection}; "
		f"random_variants={getattr(args, 'random_variants', None)}; "
		f"maf_min={args.maf_min}; max_missing={args.max_missing}; seed={args.seed}"
	)
	X, variant_ids = _build_genotype_matrix(
		args.vcf, samples,
		args.variant_selection, args.random_variants, args.variant_ids_file,
		args.maf_min, args.max_missing, args.seed,
	)
	logger.info(f"Genotype matrix shape: {X.shape} (samples x variants => {X.shape[0]} x {X.shape[1]})")

	# Make LightGBM-safe feature names while keeping original variant IDs for reporting
	feat_names_safe = _sanitize_feature_names(variant_ids)

	# Outputs
	os.makedirs(args.out_dir, exist_ok=True)
	out_base_dir = args.out_dir

	# Save variants used
	with open(os.path.join(out_base_dir, f"{args.out_name}.variants_used.txt"), "w", encoding="utf-8") as f:
		for vid in variant_ids:
			f.write(f"{vid}\n")

	# Predictions & metrics
	preds_records = []
	metrics_records = []
	importance_rows = []

	# Iterate traits
	from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
	from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score

	logger.info(f"Traits to process: {', '.join(traits)}")
	for trait in traits:
		y = phe_df[trait].values
		# drop samples with nan in this trait
		mask = np.isfinite(y)
		X_t = X[mask]
		y_t = y[mask]
		samp_t = np.array(samples)[mask]
		if len(y_t) < 3:
			logger.warning(f"Trait {trait}: too few samples after filtering; skip")
			continue

		# split
		Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(X_t, y_t, samp_t, test_size=args.test_size, random_state=args.seed)
		logger.info(f"Trait {trait}: n_total={len(y_t)}, n_train={len(ytr)}, n_test={len(yte)}, n_features={X.shape[1]}")

		# model & preprocess
		# read params from JSON file if provided
		def _read_json_file(path: Optional[str]):
			if not path:
				return None
			with open(path, 'r', encoding='utf-8') as f:
				return json.load(f)
		# auto-params: build heuristic defaults by (n_samples, n_features)
		n_samples, n_features = Xtr.shape[0], Xtr.shape[1]
		params_auto = None
		if getattr(args, 'auto_params', False):
			params_auto = {}
			if args.model == 'lgb':
				# conservative defaults for small-N high-D
				params_auto.update({
					"max_depth": 6,
					"num_leaves": int(min(31, max(7, n_samples//10))) if args.target_type=="reg" else int(min(31, max(7, n_samples//10))),
					"min_child_samples": max(10, int(round(n_samples*0.05))),
					"learning_rate": 0.05,
					"n_estimators": 500,
					"subsample": 0.8,
					"colsample_bytree": 0.8,
					"reg_alpha": 0.1,
					"reg_lambda": 0.5,
					"min_gain_to_split": 0.0
				})
			elif args.model == 'xgb':
				params_auto.update({
					"max_depth": 5,
					"min_child_weight": 5,
					"gamma": 0.0,
					"learning_rate": 0.05,
					"n_estimators": 400,
					"subsample": 0.8,
					"colsample_bytree": 0.8,
					"reg_alpha": 0.0,
					"reg_lambda": 1.0,
					"tree_method": "hist"
				})
			elif args.model == 'rf':
				params_auto.update({
					"n_estimators": 500,
					"max_depth": 15,
					"max_features": "sqrt",
					"min_samples_leaf": 5
				})
			elif args.model == 'lasso':
				# leave alpha to CV or user; provide a mild default
				params_auto.update({"alpha": 0.001})
			elif args.model == 'svm':
				# prefer linear for high-D
				params_auto.update({"C": 1.0, "kernel": "rbf" if n_features<500 else "linear"})
			elif args.model == 'knn':
				k0 = max(3, int(round(math.sqrt(max(1, n_samples)))))
				params_auto.update({"n_neighbors": k0})

		params_user = _read_json_file(getattr(args, 'params', None)) or {}
		params_merged = {}
		if params_auto:
			params_merged.update(params_auto)
		params_merged.update(params_user)
		model = _make_model(args.model, args.target_type, params_merged if params_merged else None)
		logger.info(f"Trait {trait}: model={args.model} ({args.target_type}); init_params={getattr(model, 'get_params', lambda: {})()}")
		preprocess = _make_preprocess(args.impute, args.scale, args.model == "svm")
		logger.info(f"Trait {trait}: preprocess impute={args.impute}, scale={args.scale}")
		# keep DataFrame with feature names to avoid sklearn warning
		# use sanitized names for model compatibility (e.g., LightGBM JSON restrictions)
		feat_names = feat_names_safe
		Xtr_df = pd.DataFrame(Xtr, columns=feat_names)
		Xte_df = pd.DataFrame(Xte, columns=feat_names)

		# Fit/transform then wrap back to DataFrame to preserve feature names
		Xtr_p_arr = preprocess.fit_transform(Xtr_df)
		Xte_p_arr = preprocess.transform(Xte_df)
		Xtr_p = pd.DataFrame(Xtr_p_arr, columns=feat_names, index=Xtr_df.index)
		Xte_p = pd.DataFrame(Xte_p_arr, columns=feat_names, index=Xte_df.index)

		# tuning (optional) on train
		best_params = None
		cv_results = None
		if args.tune != "none":
			try:
				grid = _read_json_file(getattr(args, 'param_grid', None))
				space_raw = _read_json_file(getattr(args, 'param_space', None))
				space = None
				if args.tune == 'bayes' and space_raw:
					try:
						skopt = importlib.import_module('skopt')
						skspace = getattr(skopt, 'space')
						Integer = getattr(skspace, 'Integer')
						Real = getattr(skspace, 'Real')
						Categorical = getattr(skspace, 'Categorical')
						space = {}
						for k, v in space_raw.items():
							if isinstance(v, dict):
								typ = v.get('type') or v.get('kind')
								if typ in ('int','integer'):
									space[k] = Integer(int(v['low']), int(v['high']))
								elif typ in ('real','float'):
									space[k] = Real(float(v['low']), float(v['high']))
								elif typ in ('cat','categorical'):
									choices = v.get('choices') or v.get('values') or []
									space[k] = Categorical(list(choices))
								else:
									# fallback to categorical for unknown spec
									space[k] = Categorical(list(v.values()) if isinstance(v, dict) else [v])
							elif isinstance(v, list):
								space[k] = Categorical(v)
							else:
								space[k] = Categorical([v])
					except Exception:
						space = None
				elif space_raw:
					# random search: expand to lists
					parsed = {}
					for k, v in space_raw.items():
						if isinstance(v, dict):
							typ = v.get('type') or v.get('kind')
							if typ in ('int','integer'):
								low, high = int(v.get('low')), int(v.get('high'))
								step = v.get('step')
								if step is not None:
									parsed[k] = list(range(low, high + 1, int(step)))
								else:
									parsed[k] = list(range(low, high + 1))
							elif typ in ('real','float'):
								low, high = float(v.get('low')), float(v.get('high'))
								q = int(v.get('q') or 20)
								parsed[k] = list(np.linspace(low, high, q))
							elif typ in ('cat','categorical'):
								parsed[k] = v.get('choices') or v.get('values') or []
							else:
								parsed[k] = v
						elif isinstance(v, list):
							parsed[k] = v
						else:
							parsed[k] = [v]
					space = parsed
			except Exception:
				grid, space = None, None
			tuned, best_params, cv_results = _tune_model(model, Xtr_p, ytr, args.target_type, args.tune, args.n_iter, grid, space, args.scoring, args.cv, args.seed)
			model = tuned

		# fit & predict
		t0 = time.perf_counter()
		model.fit(Xtr_p, ytr)
		fit_time = time.perf_counter() - t0
		logger.info(f"Trait {trait}: fit done in {fit_time:.3f}s")
		t1 = time.perf_counter()
		ytr_pred = model.predict(Xtr_p)
		yte_pred = model.predict(Xte_p)
		pred_time = time.perf_counter() - t1
		logger.info(f"Trait {trait}: predict done in {pred_time:.3f}s")

		# optional: K-fold evaluation on training set (no leakage)
		if getattr(args, 'eval_cv', False):
			try:
				from sklearn.base import clone as _sk_clone  # not strictly needed if we rebuild
			except Exception:
				_sk_clone = None
			n_splits = int(max(2, min(int(args.cv), len(ytr)))) if int(args.cv) > 1 else 2
			cv_name = None
			fold_rows = []  # (trait, fold, metric, value)
			metric_lists = {}
			try:
				if args.target_type == 'clf':
					# Prefer stratified when feasible
					unique, counts = np.unique(ytr, return_counts=True)
					ok = len(unique) >= 2 and np.min(counts) >= n_splits
					if ok:
						cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
						cv_name = 'StratifiedKFold'
					else:
						cv = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
						cv_name = 'KFold'
				else:
					cv = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
					cv_name = 'KFold'
			except Exception:
				cv = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
				cv_name = 'KFold'

			logger.info(f"Trait {trait}: running {cv_name} with {n_splits} folds on training set for evaluation")
			fold_idx = 0
			for tr_idx, va_idx in cv.split(Xtr_p, ytr if args.target_type == 'clf' and cv_name == 'StratifiedKFold' else None):
				fold_idx += 1
				# prepare raw arrays for this fold to avoid leakage
				X_tr_raw = Xtr[tr_idx]
				X_va_raw = Xtr[va_idx]
				y_tr_raw = ytr[tr_idx]
				y_va_raw = ytr[va_idx]
				# DataFrames with original feature naming
				X_tr_df = pd.DataFrame(X_tr_raw, columns=feat_names)
				X_va_df = pd.DataFrame(X_va_raw, columns=feat_names)
				# fresh preprocess & model per fold
				prep_f = _make_preprocess(args.impute, args.scale, args.model == 'svm')
				mdl_f = _make_model(args.model, args.target_type, params_merged if params_merged else None)
				# transform
				X_tr_p = prep_f.fit_transform(X_tr_df)
				X_va_p = prep_f.transform(X_va_df)
				# wrap back to DataFrame with feature names to keep name consistency (avoids sklearn/lightgbm warnings)
				X_tr_p_df = pd.DataFrame(X_tr_p, columns=feat_names, index=X_tr_df.index)
				X_va_p_df = pd.DataFrame(X_va_p, columns=feat_names, index=X_va_df.index)
				# fit and predict
				mdl_f.fit(X_tr_p_df, y_tr_raw)
				if args.target_type == 'reg':
					pred_va = mdl_f.predict(X_va_p_df)
					mvals = {
						"r2": r2_score(y_va_raw, pred_va),
						"mae": mean_absolute_error(y_va_raw, pred_va),
						"rmse": math.sqrt(mean_squared_error(y_va_raw, pred_va)),
						"pearsonr": float(np.corrcoef(y_va_raw, pred_va)[0,1]) if len(y_va_raw)>1 else np.nan,
					}
				else:
					pred_va_cls = mdl_f.predict(X_va_p_df)
					prob_va = None
					if hasattr(mdl_f, 'predict_proba'):
						try:
							prob_va = mdl_f.predict_proba(X_va_p_df)
						except Exception:
							prob_va = None
					auc = np.nan
					if prob_va is not None and isinstance(prob_va, np.ndarray) and prob_va.ndim==2 and prob_va.shape[1]==2:
						try:
							auc = roc_auc_score(y_va_raw, prob_va[:,1])
						except Exception:
							auc = np.nan
					mvals = {
						"accuracy": accuracy_score(y_va_raw, pred_va_cls),
						"f1_macro": f1_score(y_va_raw, pred_va_cls, average='macro'),
						"roc_auc": auc,
					}
				for k, v in mvals.items():
					fold_rows.append((trait, fold_idx, k, float(v) if np.isfinite(v) else np.nan))
					metric_lists.setdefault(k, []).append(float(v) if np.isfinite(v) else np.nan)

			# write per-fold metrics
			if fold_rows:
				cv_path = os.path.join(out_base_dir, f"{args.out_name}.train_cv_metrics.{trait}.tsv")
				pd.DataFrame(fold_rows, columns=["trait","fold","metric","value"]).to_csv(cv_path, sep='\t', index=False)
				logger.info(f"Trait {trait}: train CV metrics saved -> {cv_path}")
			# add mean/std into global metrics_records
			if metric_lists:
				for m, arr in metric_lists.items():
					vals = np.array(arr, dtype=float)
					m_mean = float(np.nanmean(vals)) if vals.size else np.nan
					m_std = float(np.nanstd(vals)) if vals.size else np.nan
					metrics_records.append((trait, "train_cv_mean", m, m_mean))
					metrics_records.append((trait, "train_cv_std", m, m_std))
				try:
					if args.target_type == 'reg':
						logger.info(f"Trait {trait}: CV mean R2={m_mean:.3f} (std={m_std:.3f})")
					else:
						acc_mean = float(np.nanmean(np.array(metric_lists.get('accuracy', [np.nan]), dtype=float)))
						logger.info(f"Trait {trait}: CV mean Acc={acc_mean:.3f}")
				except Exception:
					pass

		# metrics
		if args.target_type == "reg":
			def recs(split, yt, yp):
				return [
					(trait, split, "r2", r2_score(yt, yp)),
					(trait, split, "mae", mean_absolute_error(yt, yp)),
					(trait, split, "rmse", math.sqrt(mean_squared_error(yt, yp))),
					(trait, split, "pearsonr", float(np.corrcoef(yt, yp)[0,1]) if len(yt)>1 else np.nan),
				]
			metrics_records.extend(recs("train", ytr, ytr_pred))
			metrics_records.extend(recs("test", yte, yte_pred))
			try:
				m_tr = {m: v for _, _, m, v in metrics_records[-8:-4]}
				m_te = {m: v for _, _, m, v in metrics_records[-4:]}
				logger.info(f"Trait {trait}: R2 train={m_tr.get('r2'):.3f}, test={m_te.get('r2'):.3f}; RMSE train={m_tr.get('rmse'):.3f}, test={m_te.get('rmse'):.3f}")
			except Exception:
				pass
		else:
			# classification minimal set
			def mcls(split, yt, yp):
				prob = None
				if hasattr(model, "predict_proba"):
					try:
						prob = model.predict_proba(Xte_p if split=="test" else Xtr_p)
					except Exception:
						prob = None
				auc = None
				if prob is not None and prob.shape[1] == 2:
					auc = roc_auc_score(yt, prob[:,1])
				return [
					(trait, split, "accuracy", accuracy_score(yt, yp)),
					(trait, split, "f1_macro", f1_score(yt, yp, average="macro")),
					(trait, split, "roc_auc", auc if auc is not None else np.nan),
				]
			ytr_pred_cls = model.predict(Xtr_p)
			yte_pred_cls = model.predict(Xte_p)
			metrics_records.extend(mcls("train", ytr, ytr_pred_cls))
			metrics_records.extend(mcls("test", yte, yte_pred_cls))
			try:
				m_tr = {m: v for _, _, m, v in metrics_records[-6:-3]}
				m_te = {m: v for _, _, m, v in metrics_records[-3:]}
				logger.info(f"Trait {trait}: Acc train={m_tr.get('accuracy'):.3f}, test={m_te.get('accuracy'):.3f}; F1 train={m_tr.get('f1_macro'):.3f}, test={m_te.get('f1_macro'):.3f}; AUC test={m_te.get('roc_auc', float('nan')):.3f}")
			except Exception:
				pass

		# save predictions
		for s, yt, yp in zip(str_tr, ytr, ytr_pred):
			preds_records.append((s, trait, "train", float(yt), float(yp)))
		for s, yt, yp in zip(str_te, yte, yte_pred):
			preds_records.append((s, trait, "test", float(yt), float(yp)))

		# plots
		fig_size = (args.width, args.height)
		p1 = _plot_regression(yte, yte_pred, trait, out_base_dir, args.out_name, args.format, fig_size) if args.target_type=="reg" else None
		p2 = _plot_residuals(yte, yte_pred if args.target_type=="reg" else yte_pred_cls, trait, out_base_dir, args.out_name, args.format, fig_size) if args.target_type=="reg" else None

		# importance
		if args.importance in ("shap", "auto"):
			try:
				imp, shap_values = _compute_importance_shap(model, Xte_p, args.shap_background, args.target_type)
				# save bar
				imp_path = _plot_importance_bar(variant_ids, imp, trait, out_base_dir, args.out_name, args.format, args.importance_topk, fig_size)
				importance_rows.extend([(trait, variant_ids[i], float(imp[i])) for i in range(len(imp))])
				# shap plots (use original feature names for readability)
				_plot_shap_summary(shap_values, Xte_p, trait, out_base_dir, args.out_name, args.format, feature_names=variant_ids)
				_plot_shap_dependence(shap_values, Xte_p, variant_ids, args.shap_dependence_topk, trait, out_base_dir, args.out_name, args.format)
				try:
					order = np.argsort(imp)[::-1][:min(5, len(imp))]
					summary = ", ".join(f"{variant_ids[i]}={imp[i]:.4f}" for i in order)
					logger.info(f"Trait {trait}: top SHAP features: {summary}")
				except Exception:
					pass
			except Exception as e:
				logger.warning(f"SHAP importance failed ({e}); skipping SHAP for {trait}")

		elif args.importance == "permutation":
			try:
				from sklearn.inspection import permutation_importance
				r = permutation_importance(model, Xte_p, yte, n_repeats=10, random_state=args.seed, n_jobs=-1)
				imp = r.importances_mean
				_plot_importance_bar(variant_ids, imp, trait, out_base_dir, args.out_name, args.format, args.importance_topk, fig_size)
				importance_rows.extend([(trait, variant_ids[i], float(imp[i])) for i in range(len(imp))])
				try:
					order = np.argsort(imp)[::-1][:min(5, len(imp))]
					summary = ", ".join(f"{variant_ids[i]}={imp[i]:.4f}" for i in order)
					logger.info(f"Trait {trait}: top permutation importance: {summary}")
				except Exception:
					pass
			except Exception as e:
				logger.warning(f"Permutation importance failed ({e}); skipping for {trait}")

		# persist trained model bundle for later inference
		try:
			joblib = importlib.import_module('joblib')
		except Exception:
			joblib = None
		if joblib is not None:
			from datetime import datetime
			base = os.path.join(out_base_dir, f"{args.out_name}.{trait}")
			model_path = base + ".model.joblib"
			prep_path = base + ".preprocess.joblib"
			feat_sani_path = base + ".features.sanitized.txt"
			feat_orig_path = base + ".features.original.txt"
			bundle_path = base + ".bundle.json"
			try:
				joblib.dump(model, model_path)
				joblib.dump(preprocess, prep_path)
				with open(feat_sani_path, 'w', encoding='utf-8') as f:
					for nm in feat_names:
						f.write(f"{nm}\n")
				with open(feat_orig_path, 'w', encoding='utf-8') as f:
					for nm in variant_ids:
						f.write(f"{nm}\n")
				bundle = {
					"trait": trait,
					"created_at": datetime.now().isoformat(timespec='seconds'),
					"model_type": args.model,
					"target_type": args.target_type,
					"paths": {
						"model": os.path.abspath(model_path),
						"preprocess": os.path.abspath(prep_path),
						"features_sanitized": os.path.abspath(feat_sani_path),
						"features_original": os.path.abspath(feat_orig_path)
					},
					"preprocess": {"impute": args.impute, "scale": args.scale},
					"feature_n": len(feat_names)
				}
				with open(bundle_path, 'w', encoding='utf-8') as f:
					json.dump(bundle, f, indent=2)
				logger.info(f"Saved trained bundle for {trait}: {bundle_path}")
			except Exception as e:
				logger.warning(f"Failed to save trained bundle for {trait}: {e}")

		# save tuning results
		if best_params is not None:
			with open(os.path.join(out_base_dir, f"{args.out_name}.best_params.{trait}.json"), "w", encoding="utf-8") as f:
				json.dump(best_params, f, indent=2)
			logger.info(f"Saved best_params for {trait} -> {args.out_name}.best_params.{trait}.json")
		if cv_results is not None:
			pd.DataFrame(cv_results).to_csv(os.path.join(out_base_dir, f"{args.out_name}.cv_results.{trait}.tsv"), sep='\t', index=False)
			logger.info(f"Saved cv_results for {trait} -> {args.out_name}.cv_results.{trait}.tsv")

	# write predictions & metrics & importance
	if preds_records:
		pd.DataFrame(preds_records, columns=["sample","trait","split","y_true","y_pred"]).to_csv(os.path.join(out_base_dir, f"{args.out_name}.predictions.csv"), index=False)
	if metrics_records:
		pd.DataFrame(metrics_records, columns=["trait","split","metric","value"]).to_csv(os.path.join(out_base_dir, f"{args.out_name}.metrics.tsv"), sep='\t', index=False)
	if importance_rows:
		pd.DataFrame(importance_rows, columns=["trait","feature","importance"]).to_csv(os.path.join(out_base_dir, f"{args.out_name}.feature_importance.tsv"), sep='\t', index=False)

	# HTML report
	if getattr(args, "report", True):
		report_path = os.path.join(out_base_dir, f"{args.out_name}.report.html")
		# Always render with bundled Jinja2 template
		try:
			jinja2 = importlib.import_module('jinja2')
		except Exception as e:
			logger.warning("Jinja2 not installed; skip HTML report. Please install jinja2 to enable templated report.")
		else:
			from datetime import datetime
			# Locate bundled template (same package directory as this file); fallback to importlib.resources
			tpl_src = None
			tpl_default = os.path.join(os.path.dirname(__file__), 'report.html')
			if os.path.isfile(tpl_default):
				with open(tpl_default, 'r', encoding='utf-8') as f:
					tpl_src = f.read()
				logger.info(f"Using report template file: {tpl_default}")
			else:
				try:
					import importlib.resources as _ires
					pkg = __package__ or 'qtlscan.qtlscan'
					# Python 3.9+: files API
					if hasattr(_ires, 'files'):
						p = _ires.files(pkg).joinpath('report.html')
						tpl_src = p.read_text(encoding='utf-8')
						logger.info(f"Using report template resource (files API): {p}")
					else:
						with _ires.open_text(pkg, 'report.html', encoding='utf-8') as f:
							tpl_src = f.read()
						logger.info(f"Using report template resource (open_text): {pkg}/report.html")
				except Exception:
					tpl_src = None

			# After attempting to locate template, decide how to proceed
			if not tpl_src:
				logger.warning(f"Report template not found; looked for {tpl_default} and package resource. Skipping report generation.")
			else:
				tpl = jinja2.Template(tpl_src)

				# metrics head
				met_rows = []
				met_cols = []
				met_path = os.path.join(out_base_dir, f"{args.out_name}.metrics.tsv")
				if os.path.isfile(met_path):
					try:
						_met_df = pd.read_csv(met_path, sep='\t').head(200)
						# convert to template-friendly structures
						met_cols = list(_met_df.columns)
						met_rows = _met_df.to_dict(orient='records')
					except Exception:
						met_rows = []
						met_cols = []
				# top-K feature importance per trait
				importance_top = []
				imp_path = os.path.join(out_base_dir, f"{args.out_name}.feature_importance.tsv")
				if os.path.isfile(imp_path):
					try:
						df_imp = pd.read_csv(imp_path, sep='\t')
						for tr, sub in df_imp.groupby('trait'):
							ss = sub.sort_values('importance', ascending=False).head(int(args.importance_topk))
							items = [{"feature": r["feature"], "importance": float(r["importance"])} for _, r in ss.iterrows()]
							importance_top.append({"trait": tr, "items": items})
					except Exception:
						importance_top = []
				# outputs index (selected important files)
				outputs = []
				for fn in sorted(os.listdir(out_base_dir)):
					if fn.startswith(args.out_name):
						outputs.append(fn)
				# qc & config summary
				qc = {
					"maf_min": args.maf_min,
					"max_missing": args.max_missing,
					"variant_selection": args.variant_selection,
				}
				# figures (relative paths)
				figs = []
				for fn in sorted(os.listdir(out_base_dir)):
					if fn.endswith(f".{args.format}") and ("pred_vs_obs" in fn or "residuals" in fn or "importance" in fn or "shap_" in fn):
						path = os.path.join(out_base_dir, fn)
						src = os.path.relpath(path, os.path.dirname(report_path))
						figs.append({"src": src, "caption": fn})
				# model params snapshot
				try:
					model_params = getattr(model, 'get_params', lambda: {})()
				except Exception:
					model_params = {}
				model_params_items = list(model_params.items()) if isinstance(model_params, dict) else []
				# persist params for reproducibility
				try:
					with open(os.path.join(out_base_dir, f"{args.out_name}.used_params.json"), "w", encoding="utf-8") as f:
						json.dump(model_params, f, indent=2)
				except Exception:
					pass
				html = tpl.render(
					title=args.report_title,
					generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
					version="1.0.0",
					vcf_path=os.path.abspath(args.vcf),
					phe_path=os.path.abspath(args.phe),
					n_samples=len(samples), n_variants=len(variant_ids),
					model_name=args.model, target_type=args.target_type, tune=args.tune,
					selection_desc=f"{args.variant_selection}; MAF≥{args.maf_min}; Missing≤{args.max_missing}",
					test_size=args.test_size, cv=args.cv, seed=args.seed,
					impute=args.impute, scale=args.scale, importance=args.importance,
					metrics=met_rows, metrics_columns=met_cols,
					figures=figs,
					traits_list=traits,
					importance_top=importance_top,
					outputs=outputs,
					qc=qc,
					model_params=model_params, model_params_items=model_params_items,
					footer_note=""
				)
				with open(report_path, 'w', encoding='utf-8') as f:
					f.write(html)
				logger.info(f"HTML report saved to: {report_path}")

	logger.info("Done.")


def predict(args):
	"""Predict on new samples using a trained model bundle (inference-only).

	Inputs:
	- args.bundle: path to a *.bundle.json saved by train()
	- args.features: CSV/TSV with samples as rows and feature columns (original variant IDs preferred)
	- args.sample_col: optional sample ID column (default: first column)
	Outputs:
	- <out_name>.predictions.csv in out_dir
	"""
	# load bundle
	with open(args.bundle, 'r', encoding='utf-8') as f:
		bundle = json.load(f)
	paths = bundle.get('paths', {})
	try:
		joblib = importlib.import_module('joblib')
	except Exception as e:
		raise RuntimeError("joblib is required to load trained bundle; please install joblib") from e
	model = joblib.load(paths['model'])
	preprocess = joblib.load(paths['preprocess'])
	# read feature name orders
	with open(paths['features_sanitized'], 'r', encoding='utf-8') as f:
		feat_sani = [line.strip() for line in f if line.strip()]
	with open(paths['features_original'], 'r', encoding='utf-8') as f:
		feat_orig = [line.strip() for line in f if line.strip()]
	# read new features
	try:
		df = pd.read_csv(args.features, sep=None, engine='python')
	except Exception:
		sep = "\t" if os.path.splitext(args.features)[1].lower() in {".tsv", ".txt"} else ","
		df = pd.read_csv(args.features, sep=sep)
	if df.shape[1] < 2:
		raise ValueError("Feature file must contain at least 2 columns (sample + features)")
	sample_col = args.sample_col or df.columns[0]
	if sample_col not in df.columns:
		raise ValueError(f"Sample column '{sample_col}' not found in feature file")
	df = df.set_index(sample_col)
	# align incoming columns to training sanitized names using saved mapping
	# primary: original -> sanitized from bundle; secondary: if already sanitized, keep; tertiary: sanitize and match
	orig2sani = {o: s for o, s in zip(feat_orig, feat_sani)}
	rename_map = {}
	for c in df.columns:
		if c in orig2sani:
			rename_map[c] = orig2sani[c]
		elif c in feat_sani:
			# already sanitized, keep name
			pass
		else:
			c_sani = _sanitize_feature_names([c])[0]
			if c_sani in feat_sani:
				rename_map[c] = c_sani
	# apply rename and drop duplicate columns after mapping
	df2 = df.rename(columns=rename_map)
	df2 = df2.loc[:, ~df2.columns.duplicated(keep='first')]
	# ensure all expected features exist; add missing as NaN, reorder
	for col in feat_sani:
		if col not in df2.columns:
			df2[col] = np.nan
	df2 = df2[feat_sani]
	# transform and predict
	X_p = preprocess.transform(df2)
	y_pred = None
	try:
		y_pred = model.predict(X_p)
	except Exception as e:
		raise RuntimeError(f"Model prediction failed: {e}")
	# probabilities for binary classifiers (optional)
	proba = None
	if hasattr(model, 'predict_proba'):
		try:
			proba = model.predict_proba(X_p)
		except Exception:
			proba = None
	# save outputs
	os.makedirs(args.out_dir, exist_ok=True)
	out_csv = os.path.join(args.out_dir, f"{args.out_name}.predictions.csv")
	rows = []
	idx = list(df2.index)
	if proba is not None and isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
		for sid, yp, p in zip(idx, y_pred, proba[:,1]):
			rows.append((sid, bundle.get('trait') or args.trait or 'trait', float(yp), float(p)))
		pd.DataFrame(rows, columns=['sample','trait','y_pred','prob_1']).to_csv(out_csv, index=False)
	else:
		for sid, yp in zip(idx, y_pred):
			rows.append((sid, bundle.get('trait') or args.trait or 'trait', float(yp)))
		pd.DataFrame(rows, columns=['sample','trait','y_pred']).to_csv(out_csv, index=False)
	logger.info(f"Predictions saved to: {out_csv}")

