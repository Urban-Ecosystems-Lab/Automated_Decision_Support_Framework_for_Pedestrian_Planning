"""Automated Pattern Identification Framework.

Automated modeling, SHAP interpretation, clustering, and response-pattern fitting.

Inputs must already contain numeric predictors in their intended names and units.

Run `python pedestrian_framework.py --help` for command-line options.

Input data and configuration
----------------------------
Supply a CSV containing the five required columns: `DeviceID` (intersection identifier), `DateHour` (observation timestamp), `Latitude`, `Longitude`, and `HourlyPedCount` (nonnegative pedestrian count). All remaining columns are treated as numeric predictors. The identifier, timestamp, and coordinates are excluded from model predictors; the identifier and coordinates support intersection-level grouping and geographic clustering. Column order and dataset dimensions are read directly from the CSV; no separate schema file is required.

Predictors must have meaningful names and consistent units before analysis. Complete categorical encoding and source-data integration upstream. Missing numeric values are mean-imputed within each training split for evaluation; the final explanatory model is fitted to all observations.

| Configuration section | Settings |
|---|---|
| `model` | Objective, randomized-search space, trial count, test fraction, validation folds, and baseline comparisons |
| `clustering` | Importance-gap search range, minimum drop, fallback count, stability resamples, cluster range, and geographic weighting |
| `patterns` | Number of ranked features, candidate limits, segment support, breakpoint grid, bootstrap controls, and MAE tolerances |
| `figures` | Figure generation, resolution, and number of displayed features |

Paths and random seeds are configured at the top level. Selected model hyperparameters are saved with each run.

Feature selection and context clustering
---------------------------------------
Features are ranked by aggregate mean absolute SHAP importance. The relative drop between adjacent ranks is:

```text
Drop_k (%) = 100 × (1 − I_(k+1) / I_k)
```

By default, the largest drop of at least 30% among adjacent pairs within the top 10 features defines the cutoff. All features above that cutoff are selected. Ties select the earlier cutoff. If no qualifying positive drop exists, the algorithm selects the top three and records the fallback. Zero-to-zero gaps are undefined and do not qualify. These are configurable selection heuristics.

An intersection bootstrap assesses feature-selection stability by resampling complete intersection contributions to SHAP importance. The default is 100 resamples. Selection frequencies and cutoff counts are reported conditional on the fitted model.

Clustering uses standardized projected coordinates and intersection means of selected predictors. Each block is normalized by the square root of its dimension, with equal total weight assigned to geography and predictors by default. KMeans solutions from 2 through 20 clusters are evaluated, subject to data availability and a minimum of two intersections per cluster. The highest eligible silhouette score determines the selected count; ties favor fewer clusters. Cluster identifiers are specific to each run.

Pattern selection orchestration
-------------------------------
Segmented candidates are fitted on a data-supported breakpoint grid. Defaults permit up to five segments with at least 25 observations per segment. The best breakpoint placement is identified for each supported segment count. Each family is simplified sequentially while the next lower segment count increases MAE by no more than 10%.

At the aggregate level, 100 observation-bootstrap resamples assess fit variability conditional on the fitted model and SHAP values. Supported functional forms and segment counts are compared across resamples before family simplification. Up to three surviving forms with mean bootstrap MAE within 10% of the best are retained and refitted to the complete dependence values. Valid-resample counts and selection decisions are exported. The required valid-resample fraction is configurable and defaults to 100%.

At the cluster level, the same candidate library is fitted directly to cluster-stratified SHAP values. After segment simplification, the simplest form within 10% of the lowest MAE is selected. All cluster analyses use explanations from the same aggregate model.

Interpretation
--------------
SHAP response patterns describe associations represented by a fitted model. For XGBoost with Poisson loss, the exported SHAP values explain the raw model output on the log scale. They are not causal intervention effects.

Curve-bootstrap intervals summarize variability in fitted-curve metrics conditional on the fixed model. They exclude uncertainty from model estimation and are percentile intervals for metric distributions, not confidence intervals for mean error. Parameter tests are conditional on the selected form and breakpoints and do not adjust for model selection, repeated observations, or multiple comparisons. Breakpoint precision is limited by the candidate grid.

Random observation-level cross-validation assesses predictive consistency within the observed sample. Evaluation at unseen intersections requires a spatial or intersection-level validation design.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import re
import sys
import warnings
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from pedestrian_patterns import PatternFitter
from pedestrian_figures import plot_patterns, plot_overview

DEFAULT_CONFIG = Path(__file__).with_name('framework_config.json')


def load_config(path=DEFAULT_CONFIG):
    """Resolve paths and validate algorithm settings and input/output paths."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ('input_path', 'output_root'):
        config[key] = str((path.parent / config[key]).resolve())
    validate_config(config)
    return config


def validate_config(config):
    """Validate parameter ranges and required algorithm settings."""
    p, c, m = config['patterns'], config['clustering'], config['model']
    for value, label in [(p['bootstrap_replicates'], 'bootstrap_replicates'),
                         (c['feature_stability_replicates'], 'feature_stability_replicates'),
                         (p['min_points_per_segment'], 'min_points_per_segment'),
                         (p['top_n'], 'top_n'), (c['gap_search_top_n'], 'gap_search_top_n'),
                         (c['fallback_top_n'], 'fallback_top_n'),
                         (p['breakpoint_candidates'], 'breakpoint_candidates'),
                         (p['max_citywide_forms'], 'max_citywide_forms'),
                         (m['search_iterations'], 'search_iterations')]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{label} must be a positive integer.')
    if any(not isinstance(jobs, int) or jobs == 0 for jobs in [m['jobs'],p['jobs']]):
        raise ValueError('Parallel jobs must be a nonzero integer.')
    if c['min_intersections_per_cluster'] < 1:
        raise ValueError('Minimum intersections per cluster must be positive.')
    if p['max_segments'] < 2 or m['cv_folds'] < 2 or c['k_min'] < 2 or c['k_max'] < c['k_min']:
        raise ValueError('Invalid segment, validation-fold or cluster count.')
    if not 0 < p['bootstrap_min_valid_fraction'] <= 1 or not 0 < p['bootstrap_ci_level'] < 1:
        raise ValueError('Invalid bootstrap validity or interval level.')
    if not 0 <= c['minimum_gap_pct'] <= 100 or not 0 <= c['geography_weight'] <= 1:
        raise ValueError('Gap percentage and geography weight are out of range.')
    if p['retention_tolerance_pct'] < 0 or p['simplification_tolerance_pct'] < 0:
        raise ValueError('MAE tolerances cannot be negative.')
    if not 0 < m['test_fraction'] < 1:
        raise ValueError('test_fraction must be between zero and one.')


def prepare_data(config):
    """Validate numeric predictors, outcome values, and intersection metadata."""
    raw = pd.read_csv(config['input_path'])
    required = ['DeviceID', 'DateHour', 'Latitude', 'Longitude', 'HourlyPedCount']
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f'Missing required input columns: {missing}')
    if raw['DeviceID'].isna().any() or len(raw) < config['model']['cv_folds'] + 2:
        raise ValueError('Missing intersection IDs or too few observations.')
    y = pd.to_numeric(raw['HourlyPedCount'], errors='coerce')
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Pedestrian counts must be finite and nonnegative.')
    for col, bounds in [('Latitude', (-90, 90)), ('Longitude', (-180, 180))]:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
        if not raw[col].between(*bounds).all():
            raise ValueError(f'Invalid coordinates in {col}.')
    positions = raw.groupby('DeviceID')[['Latitude', 'Longitude']].nunique()
    if (positions > 1).any().any():
        raise ValueError('Each intersection must have one consistent coordinate pair.')
    x = raw.drop(columns=required)
    nonnumeric = [col for col in x if not pd.api.types.is_numeric_dtype(x[col])]
    if nonnumeric:
        raise ValueError(f'All columns outside the five required fields must be numeric predictors; preprocess or remove: {nonnumeric}')
    x = x.astype(float)
    if x.columns.duplicated().any():
        raise ValueError('Duplicate predictor names.')
    x = x.replace([np.inf, -np.inf], np.nan)
    unusable = x.columns[x.isna().all()].tolist()
    if unusable:
        warnings.warn(f'Dropping entirely missing or nonnumeric features: {unusable}')
        x = x.drop(columns=unusable)
    if x.empty or x.shape[1] == 0:
        raise ValueError('No usable numeric predictors.')
    return raw, x, y


def metrics(y, predicted):
    """Calculate mean absolute error, root mean squared error, and R-squared."""
    return {'MAE': float(mean_absolute_error(y, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(y, predicted))),
            'R2': float(r2_score(y, predicted))}


def model_pipeline(parameters):
    """Construct a mean-imputation and XGBoost regression pipeline."""
    return Pipeline([('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
                     ('model', XGBRegressor(**parameters))])


def fit_model(x, y, config, output):
    """Tune on the training split, evaluate held-out data and random CV, then refit."""
    m = config['model']
    train, test = train_test_split(np.arange(len(x)), test_size=m['test_fraction'], random_state=config['seed'])
    params = {'objective': m['objective'], 'tree_method': 'hist',
              'random_state': config['seed'], 'n_jobs': m['jobs']}
    log('Selecting XGBoost hyperparameters on the training split')
    search = RandomizedSearchCV(model_pipeline(params),
        {'model__' + key: value for key, value in m['search_space'].items()},
        n_iter=m['search_iterations'], scoring='r2',
        cv=KFold(m['cv_folds'], shuffle=True, random_state=config['seed']),
        random_state=config['seed'], n_jobs=m['jobs'], error_score='raise')
    search.fit(x.iloc[train], y.iloc[train])
    params.update({key.removeprefix('model__'): value for key, value in search.best_params_.items()})
    rows = []
    pipeline = model_pipeline(params).fit(x.iloc[train], y.iloc[train])
    rows.append({'Model': 'XGBoost', 'Evaluation': 'Held-out test split', 'Fold': 0,
                 **metrics(y.iloc[test], pipeline.predict(x.iloc[test]))})
    if m['compare_baselines']:
        baselines = {
            'Random forest': RandomForestRegressor(n_estimators=200, random_state=config['seed'], n_jobs=m['jobs']),
            'MLP': Pipeline([('scale', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=config['seed']))]),
            'SVR': Pipeline([('scale', StandardScaler()), ('model', SVR())]),
        }
        for name, estimator in baselines.items():
            baseline = Pipeline([('imputer', SimpleImputer(keep_empty_features=True)), ('estimator', estimator)])
            baseline.fit(x.iloc[train], y.iloc[train])
            rows.append({'Model': name, 'Evaluation': 'Held-out test split', 'Fold': 0,
                         **metrics(y.iloc[test], baseline.predict(x.iloc[test]))})
    splitter = KFold(m['cv_folds'], shuffle=True, random_state=config['seed'])
    for fold, (tr, te) in enumerate(splitter.split(x), start=1):
        fitted = model_pipeline(params).fit(x.iloc[tr], y.iloc[tr])
        rows.append({'Model': 'XGBoost', 'Evaluation': 'Random observation CV', 'Fold': fold,
                     **metrics(y.iloc[te], fitted.predict(x.iloc[te]))})
    pd.DataFrame(rows).to_csv(output / 'model_evaluation.csv', index=False)
    final = model_pipeline(params).fit(x, y)
    return final, {'selected_parameters': params, 'search_best_cv_r2': float(search.best_score_)}


def select_gap_features(importance, settings):
    """Largest adjacent relative gap within the leading N, with explicit fallback.

    'Within top N' means both members of the pair are in the top N.
    Stable sorting resolves ties in input-column order.
    """
    ranked = importance.sort_values(ascending=False, kind='stable')
    table = pd.DataFrame({'Feature': ranked.index, 'Mean absolute SHAP': ranked.values})
    current = table['Mean absolute SHAP'].to_numpy()
    if np.any(~np.isfinite(current)) or np.any(current < 0) or len(current) == 0:
        raise ValueError('Feature importance must be finite, nonnegative and nonempty.')
    gaps = np.full(len(current), np.nan)
    if len(current) > 1:
        gaps[:-1] = np.divide(100 * (current[:-1] - current[1:]), current[:-1],
                             out=np.full(len(current)-1, np.nan), where=current[:-1] > 0)
    table['Drop to next (%)'] = gaps
    table['Rank'] = np.arange(1, len(current) + 1)
    table['Searched gap'] = table['Rank'] < min(settings['gap_search_top_n'], len(current))
    eligible = table['Searched gap'] & (table['Drop to next (%)'] >= settings['minimum_gap_pct']) & (table['Drop to next (%)'] > 0)
    if eligible.any():
        row = table.loc[eligible, 'Drop to next (%)'].idxmax()
        count, reason = int(row + 1), 'largest qualifying relative drop'
    else:
        count, reason = min(settings['fallback_top_n'], len(current)), 'no clear separation; fallback'
    table['Selected'] = table['Rank'] <= count
    return table.loc[table['Selected'], 'Feature'].tolist(), table, reason


def feature_stability(shap_values, devices, columns, settings, seed, output=None):
    """Intersection bootstrap of fixed SHAP values; does not refit the model."""
    codes, unique = pd.factorize(devices, sort=False)
    totals = np.zeros((len(unique), shap_values.shape[1]))
    np.add.at(totals, codes, np.abs(shap_values))
    counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    rows, frequency = [], dict.fromkeys(columns, 0)
    for replicate in range(settings['feature_stability_replicates']):
        sampled = rng.integers(0, len(unique), size=len(unique))
        importance = pd.Series(totals[sampled].sum(axis=0) / counts[sampled].sum(), index=columns)
        selected, _, reason = select_gap_features(importance, settings)
        for col in selected:
            frequency[col] += 1
        rows.append({'Replicate': replicate + 1, 'Selected count': len(selected),
                     'Features': json.dumps(selected), 'Reason': reason})
    frequencies = pd.DataFrame([
        {'Feature': col, 'Selection frequency': n/settings['feature_stability_replicates']}
        for col, n in frequency.items()])
    counts = pd.Series([row['Selected count'] for row in rows]).value_counts().sort_index()
    summary = {'resamples': settings['feature_stability_replicates'],
               'selected_count_frequency': {str(k): int(v) for k,v in counts.items()},
               'fallback_frequency': sum('fallback' in row['Reason'] for row in rows)/len(rows)}
    return frequencies, summary


def cluster_intersections(raw, x, importance, config, output):
    """Select influential predictors and partition intersections by context."""
    c = config['clustering']
    selected, gap_table, reason = select_gap_features(importance, c)
    missing = [col for col in selected if col not in x]
    if missing:
        raise ValueError(f'Clustering features unavailable: {missing}')
    work = raw[['DeviceID', 'Latitude', 'Longitude']].copy()
    for col in selected:
        work[col] = x[col].to_numpy()
    intersections = work.groupby('DeviceID', sort=False, as_index=False).agg({
        'Latitude': 'first', 'Longitude': 'first', **{col: 'mean' for col in selected}})
    import geopandas as gpd
    points = gpd.GeoDataFrame(intersections, geometry=gpd.points_from_xy(intersections['Longitude'], intersections['Latitude']), crs='EPSG:4326').to_crs(3857)
    geo = StandardScaler().fit_transform(np.column_stack([points.geometry.x, points.geometry.y])) / np.sqrt(2)
    signal = StandardScaler().fit_transform(intersections[selected]) / np.sqrt(len(selected))
    weight = c['geography_weight']
    matrix = np.hstack([geo * np.sqrt(weight), signal * np.sqrt(1-weight)])
    trials, labels_by_k = [], {}
    for k in range(c['k_min'], min(c['k_max'], len(intersections)-1) + 1):
        labels = KMeans(n_clusters=k, random_state=config['seed'], n_init=30).fit_predict(matrix)
        sizes = np.bincount(labels)
        valid = len(np.unique(labels)) == k and sizes.min() >= c['min_intersections_per_cluster']
        score = silhouette_score(matrix, labels) if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels) else np.nan
        trials.append({'K': k, 'Silhouette': score, 'Smallest cluster': sizes.min(), 'Eligible': valid})
        labels_by_k[k] = labels
    scores = pd.DataFrame(trials)
    if scores.empty or not scores['Eligible'].any():
        raise ValueError('Too few distinct intersections for an eligible cluster solution.')
    k = int(scores.loc[scores.Eligible].sort_values(['Silhouette','K'], ascending=[False,True]).iloc[0]['K'])
    scores['Selected'] = scores.K.eq(k)
    intersections['Cluster'] = labels_by_k[k] + 1
    labels = raw['DeviceID'].map(intersections.set_index('DeviceID')['Cluster']).to_numpy()
    return labels, intersections, scores


def predict_curve(candidate, x):
    """Evaluate a fitted response function at the supplied feature values."""
    x = np.asarray(x, dtype=float)
    b, c, method = candidate['breakpoints'], candidate['coefficients'], candidate['method']
    if method in {'Constant','Linear','Quadratic','Cubic','Quartic'}:
        if 'scaled_coefficients' in candidate:
            return np.polynomial.polynomial.polyval((x-candidate['polynomial_center'])/candidate['polynomial_scale'], candidate['scaled_coefficients'])
        return np.polynomial.polynomial.polyval(x, c)
    if method == 'Logarithmic':
        return c[0] + c[1] * np.log1p(x)
    if method == 'Sigmoid':
        return c[0] + c[1] * expit(c[2]*(x-c[3]))
    if method.startswith('Piecewise-constant'):
        return c[np.digitize(x, b, right=True)]
    hinges = [np.maximum(0, x-point) for point in b]
    if method.startswith('Continuous plateau'):
        design = np.column_stack([np.ones(x.size), x-hinges[-1], *[h-hinges[-1] for h in hinges[:-1]]])
    else:
        design = np.column_stack([np.ones(x.size), x, *hinges])
    return design @ c


def describe_pattern(candidate, x):
    """Describe the fitted curve over the observed range, in original feature units."""
    lo, hi = float(np.min(x)), float(np.max(x))
    grid = np.linspace(lo, hi, 501)
    fitted = predict_curve(candidate, grid)
    eps = max(float(np.ptp(fitted))*1e-6, 1e-10)
    differences = np.diff(fitted)
    up, down = np.any(differences > eps), np.any(differences < -eps)
    direction = 'mixed' if up and down else 'increasing' if up else 'decreasing' if down else 'constant'
    c, method = candidate['coefficients'], candidate['method']
    slopes = []
    if method.startswith('Continuous'):
        slopes = [float(c[1] + sum(c[2:segment+2])) for segment in range(len(candidate['breakpoints'])+1)]
        if method.startswith('Continuous plateau'):
            slopes[-1] = 0.0
    turning = []
    if method in {'Quadratic','Cubic','Quartic'}:
        # Derivative roots are numerically stable in the scaled coordinate system.
        poly = np.polynomial.Polynomial(candidate['scaled_coefficients']).deriv()
        for root in poly.roots():
            if abs(root.imag) < 1e-8:
                value = float(root.real*candidate['polynomial_scale']+candidate['polynomial_center'])
                if lo < value < hi:
                    turning.append(value)
    midpoint = float(c[3]) if method == 'Sigmoid' and lo <= c[3] <= hi else None
    return {'Direction': direction, 'Observed min': lo, 'Observed max': hi,
            'Segment slopes': json.dumps(slopes), 'Turning points': json.dumps(sorted(turning)),
            'Sigmoid midpoint': midpoint, 'Ending plateau': method.startswith('Continuous plateau'),
            'Breakpoints': json.dumps(candidate['breakpoints'])}


def curve_equation(candidate, x):
    """Explicit piecewise equations, including coefficients and interval limits."""
    method, coefficients = candidate['method'], candidate['coefficients']
    if not candidate['breakpoints']:
        return candidate['equation']
    edges = [float(np.min(x)), *candidate['breakpoints'], float(np.max(x))]
    pieces = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        condition = (f'x <= {hi:.8g}' if index == 0 else
                     f'x > {lo:.8g}' if index == len(edges)-2 else
                     f'{lo:.8g} < x <= {hi:.8g}')
        if method.startswith('Piecewise-constant'):
            expression = f'{coefficients[index]:.8g}'
        else:
            slope = float(coefficients[1] + sum(coefficients[2:index+2]))
            if method.startswith('Continuous plateau') and index == len(edges)-2:
                slope = 0.0
            midpoint = (lo+hi)/2
            intercept = float(predict_curve(candidate, [midpoint])[0] - slope*midpoint)
            expression = f'{intercept:.8g} {slope:+.8g}*x'
        pieces.append(f'{expression} when {condition}')
    return 'SHAP(x) = ' + '; '.join(pieces)


def bootstrap_one(seed, x, shap_values, settings):
    """Fit supported candidate forms to one resampled dependence dataset."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=len(x))
    try:
        candidates = PatternFitter(settings).fit_pattern_library(x[indices], shap_values[indices], simplify=False)
        return {'candidates': [{key: candidate[key] for key in ('method', 'mae', 'rmse', 'r2')} for candidate in candidates], 'error': None}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return {'candidates': [], 'error': str(error)}


def citywide_patterns(feature, x, shap_values, config, output=None):
    """Compare fixed family/segment counts across resamples, then simplify families.

    Evaluating every supported segment count before simplification avoids comparing
    only resamples in which a particular segment count happened to be selected.
    """
    p = config['patterns']
    fitter = PatternFitter(p)
    full = {c['method']: c for c in fitter.fit_pattern_library(x, shap_values, simplify=False)}
    feature_seed = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:4], 'little')
    seeds = np.random.SeedSequence([p['bootstrap_seed'], feature_seed]).spawn(p['bootstrap_replicates'])
    results = Parallel(n_jobs=p['jobs'])(delayed(bootstrap_one)(int(seed.generate_state(1)[0]), x, shap_values, p) for seed in seeds)
    rows = []
    for index, result in enumerate(results, start=1):
        for c in result['candidates']:
            rows.append({'Feature': feature, 'Replicate': index, 'Pattern': c['method'],
                         'MAE': c['mae'], 'RMSE': c['rmse'], 'R2': c['r2']})
        if result['error']:
            rows.append({'Feature': feature, 'Replicate': index, 'Pattern': 'FAILED', 'Error': result['error']})
    records = pd.DataFrame(rows)
    summaries = []
    alpha = (1-p['bootstrap_ci_level'])/2
    for method, group in records.loc[records.Pattern.ne('FAILED')].groupby('Pattern', sort=False):
        summary = {'Feature': feature, 'Pattern': method, 'Valid replicates': len(group),
                   'Eligible': len(group)/p['bootstrap_replicates'] >= p['bootstrap_min_valid_fraction'] and method in full}
        for metric in ('MAE','RMSE','R2'):
            summary.update({f'Mean {metric}': group[metric].mean(),
                            f'{metric} lower': group[metric].quantile(alpha),
                            f'{metric} upper': group[metric].quantile(1-alpha)})
        summaries.append(summary)
    table = pd.DataFrame(summaries)
    eligible = []
    for row in summaries:
        if row['Eligible']:
            candidate = dict(full[row['Pattern']])
            candidate['mae'] = row['Mean MAE']
            eligible.append(candidate)
    if not eligible:
        raise ValueError(f'No bootstrap-eligible forms for {feature}; check sample support and bootstrap-validity settings.')
    simplified = fitter.reduce_segmented_candidates(eligible)
    minimum = min(c['mae'] for c in simplified)
    chosen = sorted([c for c in simplified if c['mae'] <= minimum*(1+p['retention_tolerance_pct']/100)+1e-12],
                    key=lambda c: (c['mae'], c['n_parameters'], c['method']))[:p['max_citywide_forms']]
    methods = [c['method'] for c in chosen]
    table['Survives simplification'] = table.Pattern.isin([c['method'] for c in simplified])
    table['Retained'] = table.Pattern.isin(methods)
    return [full[method] for method in methods], table


def candidate_row(feature, scope, candidate, x, retained):
    """Assemble fit statistics and response descriptors for one candidate."""
    return {'Feature': feature, 'Scope': scope, 'Pattern': candidate['method'], 'Retained': retained,
            'N': len(x), 'Equation': curve_equation(candidate, x),
            'Coefficients': json.dumps(candidate['coefficients'].tolist()),
            'MAE': candidate['mae'], 'RMSE': candidate['rmse'], 'R2': candidate['r2'],
            'F statistic': candidate['f_stat'], 'F p-value': candidate['f_pvalue'],
            'Effective parameters': candidate['n_parameters'], **describe_pattern(candidate, x)}


def run_patterns(x, shap_values, labels, config, output, figures, intersection_ids=None):
    """Fit aggregate and cluster response patterns and export their diagnostics."""
    p, fitter = config['patterns'], PatternFitter(config['patterns'])
    ranked = pd.Series(np.mean(np.abs(shap_values), axis=0), index=x.columns).sort_values(ascending=False, kind='stable')
    features = ranked.head(p['top_n']).index.tolist()
    missing = [feature for feature in features if feature not in x]
    if missing or not features:
        raise ValueError(f'Invalid pattern feature selection: {missing}')
    all_rows, tests, summary_tables = [], [], []
    results = {}
    for feature in features:
        log(f'Fitting patterns: {feature}')
        values, shap_y = x[feature].to_numpy(), shap_values[:, x.columns.get_loc(feature)]
        retained, bootstrap = citywide_patterns(feature, values, shap_y, config, output)
        summary_tables.append(bootstrap)
        for candidate in retained:
            all_rows.append(candidate_row(feature, 'Citywide', candidate, values, True))
        groups = {}
        for cluster in np.unique(labels):
            mask = labels == cluster
            if mask.sum() < 2:
                warnings.warn(f'Skipping {feature}, cluster {cluster}: too few observations')
                continue
            candidates = fitter.fit_pattern_library(values[mask], shap_y[mask], simplify=False)
            # Same family simplification, then simplest eligible form within tolerance.
            selected = fitter.retained_candidates(fitter.reduce_segmented_candidates(candidates))[0]
            groups[int(cluster)] = selected
            all_rows.append(candidate_row(feature, f'Cluster {cluster}', selected, values[mask], True))
        for scope, candidates in [('Citywide', retained)] + [(f'Cluster {k}',[v]) for k,v in groups.items()]:
            for candidate in candidates:
                for test in fitter.conditional_parameter_tests(candidate):
                    tests.append({'Feature': feature, 'Scope': scope, 'Pattern': candidate['method'],
                                  'Inference': 'conditional on fixed fitted model, selected form and breakpoints', **test})
        interval_rows = bootstrap[bootstrap.Retained]
        intervals = {row['Pattern']: {
            metric.lower(): {'mean': row[f'Mean {metric}'],
                             'lower': row[f'{metric} lower'], 'upper': row[f'{metric} upper']}
            for metric in ['MAE', 'RMSE', 'R2']
        } for _, row in interval_rows.iterrows()}
        ids = np.asarray(intersection_ids) if intersection_ids is not None else np.arange(len(labels))
        results[feature] = {
            'citywide': retained, 'clusters': groups, 'bootstrap_metrics': intervals,
            'cluster_sizes': {int(group): int(pd.Series(ids[labels == group]).nunique()) for group in groups},
        }
        if config['figures']['enabled']:
            plot_patterns(feature, values, shap_y, labels, results[feature], figures, config)
        selected_rows = pd.DataFrame(all_rows).drop(columns='Retained')
        bootstrap_rows = pd.concat(summary_tables, ignore_index=True)
        bootstrap_rows = bootstrap_rows[bootstrap_rows.Retained].copy()
        metric_columns = ['Valid replicates'] + [f'{prefix}{metric}{suffix}'
            for metric in ['MAE', 'RMSE', 'R2']
            for prefix, suffix in [('Mean ', ''), ('', ' lower'), ('', ' upper')]]
        bootstrap_rows = bootstrap_rows[['Feature', 'Pattern'] + metric_columns]
        bootstrap_rows['Scope'] = 'Citywide'
        bootstrap_rows = bootstrap_rows.rename(columns={
            col: ('Bootstrap valid replicates' if col == 'Valid replicates' else
                  'Bootstrap mean ' + col.removeprefix('Mean ') if col.startswith('Mean ') else
                  'Bootstrap ' + col) for col in metric_columns})
        selected_rows = selected_rows.merge(bootstrap_rows, on=['Feature','Scope','Pattern'], how='left', validate='one_to_one')
        selected_rows = selected_rows.rename(columns={'MAE':'Full-data MAE', 'RMSE':'Full-data RMSE', 'R2':'Full-data R2'})
        selected_rows.to_csv(output / 'patterns.csv', index=False)
        pd.DataFrame(tests, columns=['Feature','Scope','Pattern','Inference','Parameter / contrast','Estimate','Standard error','t-value','p-value']).to_csv(output / 'parameter_tests.csv', index=False)
    return results


def slug(text):
    """Convert a display label to a lowercase filename component."""
    return re.sub(r'[^a-z0-9]+','_',text.lower()).strip('_')


def write_json(path, value):
    """Serialize metadata with support for NumPy values and filesystem paths."""
    def convert(obj):
        if isinstance(obj,np.generic): return obj.item()
        if isinstance(obj,np.ndarray): return obj.tolist()
        if isinstance(obj,Path): return str(obj)
        raise TypeError(f'Not serializable: {type(obj)}')
    Path(path).write_text(json.dumps(value,indent=2,default=convert)+'\n')


def log(message):
    """Write a progress message immediately."""
    print(message,flush=True)


def _run_analysis(config, output_dir):
    """Compute one complete analysis in a new staging directory."""
    config=copy.deepcopy(config);validate_config(config)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    figures=output/'figures';figures.mkdir()
    manifest={'configuration':config,'framework':'automated_pattern_identification','status':'running','started_utc':stamp,'warnings':[],
              'notes':['SHAP explains the fixed citywide model output scale, not a causal effect.',
                       'Curve bootstrap resamples observations conditional on the fitted model.',
                       'Feature-selection bootstrap resamples intersections conditional on the fitted model.',
                       'Random CV evaluates observation-level generalization; it is not spatial validation.']}
    packages=['numpy','pandas','scipy','scikit-learn','xgboost','shap','geopandas','joblib','matplotlib']
    manifest['python']=sys.version
    manifest['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ['pedestrian_framework.py', 'pedestrian_patterns.py', 'pedestrian_figures.py']}
    manifest['packages']={}
    for name in packages:
        try:manifest['packages'][name]=version(name)
        except PackageNotFoundError:manifest['packages'][name]='unavailable'
    write_json(output/'run_summary.json',manifest)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            log('Preparing data')
            raw,x,y=prepare_data(config)
            manifest.update({'input_sha256':hashlib.sha256(Path(config['input_path']).read_bytes()).hexdigest(),
                             'rows':len(x),'features':x.shape[1],'intersections':raw['DeviceID'].nunique()})
            model, model_summary=fit_model(x,y,config,output)
            manifest['model']=model_summary
            filled=pd.DataFrame(model.named_steps['imputer'].transform(x),columns=x.columns,index=x.index)
            import shap
            log('Computing citywide SHAP values')
            explainer=shap.TreeExplainer(model.named_steps['model'])
            shap_values=np.asarray(explainer.shap_values(filled))
            margin=model.named_steps['model'].predict(filled, output_margin=True)
            if not np.allclose(shap_values.sum(axis=1) + explainer.expected_value, margin, rtol=1e-4, atol=1e-4):
                raise ValueError('SHAP additivity check failed on the model output scale.')
            importance=pd.Series(np.abs(shap_values).mean(axis=0),index=x.columns).sort_values(ascending=False,kind='stable')
            stability, stability_summary = feature_stability(shap_values,raw['DeviceID'],x.columns,config['clustering'],config['seed'])
            manifest['feature_selection_stability'] = stability_summary
            log('Selecting clustering features and cluster count')
            labels,intersections,scores=cluster_intersections(raw,filled,importance,config,output)
            cluster_imp=pd.DataFrame({f'Cluster {k}':np.abs(shap_values[labels==k]).mean(axis=0) for k in np.unique(labels)},index=x.columns)
            cluster_imp['Equal-weight cluster average']=cluster_imp.mean(axis=1)
            _, gap_table, selection_reason = select_gap_features(importance,config['clustering'])
            feature_summary = gap_table.rename(columns={
                'Mean absolute SHAP':'Citywide mean absolute SHAP',
                'Selected':'Selected for clustering'}).merge(
                stability,on='Feature',how='left',validate='one_to_one').merge(
                cluster_imp.rename_axis('Feature').reset_index(),on='Feature',how='left',validate='one_to_one')
            feature_summary.to_csv(output/'feature_summary.csv',index=False)
            intersections.to_csv(output/'clusters.csv',index=False)
            selected_score=scores.loc[scores.Selected].iloc[0]
            manifest['clustering'] = {
                'k':int(selected_score.K), 'silhouette':float(selected_score.Silhouette),
                'selected_feature_count':int(gap_table.Selected.sum()),
                'feature_selection_reason':selection_reason,
                'cluster_count_comparison':scores.to_dict(orient='records'),
                'cluster_labels_are_run_specific':True}

            if config['figures']['enabled']:
                importance.attrs['columns']=x.columns.tolist()
                plot_overview(importance,shap_values,labels,intersections,scores,gap_table,config,figures)
            run_patterns(filled,shap_values,labels,config,output,figures,
                         intersection_ids=raw['DeviceID'])
            manifest['warnings']=sorted({str(w.message) for w in captured})
            manifest['status']='completed'
    except Exception as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(output/'run_summary.json',manifest)
    return output


def run_framework(config, output_dir=None):
    """Replace one results directory after the complete analysis succeeds.

    Failed runs leave the previous results intact. Only directories created by
    this framework may be replaced.
    """
    validate_config(config)
    destination = Path(output_dir or config['output_root']).resolve()
    if destination.exists():
        manifest_path = destination / 'run_summary.json'
        if not manifest_path.exists():
            manifest_path = destination / 'manifest.json'
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f'Refusing to replace a directory without a framework manifest: {destination}') from error
        if previous_manifest.get('framework') != 'automated_pattern_identification':
            raise ValueError(f'Refusing to replace a directory not owned by this framework: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.framework-', dir=destination.parent) as workspace:
        workspace = Path(workspace)
        staged = _run_analysis(config, workspace / 'results')
        previous = workspace / 'previous'
        if destination.exists():
            os.replace(destination, previous)
        try:
            os.replace(staged, destination)
        except OSError:
            if previous.exists():
                os.replace(previous, destination)
            raise
    log(f'Completed: {destination}')
    return destination


def main():
    """Parse command-line arguments and execute the configured analysis."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir',type=Path,help='Results directory; a successful run replaces prior framework results.')
    parser.add_argument('--quick',action='store_true',help='Reduced run: two features, five resamples, two folds, three segments, one tuning trial and no baselines.')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.quick:
        config['model'].update(search_iterations=1,compare_baselines=False,cv_folds=2)
        config['clustering']['feature_stability_replicates']=5
        config['patterns'].update(top_n=2,bootstrap_replicates=5,max_segments=3,breakpoint_candidates=6)
    run_framework(config,args.output_dir)


if __name__=='__main__':
    main()
