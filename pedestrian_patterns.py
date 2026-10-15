"""Candidate response functions and diagnostics for SHAP dependence values.

Tests are conditional diagnostics, not selection-adjusted inference.

Candidate forms and selection
-----------------------------
The candidate library includes constant, linear, quadratic, cubic, quartic, logarithmic, sigmoid, continuous piecewise-linear, ending-plateau, and step functions. These forms represent responses such as a steady increase, diminishing returns, discrete transitions, and changes in direction.

Segmented candidates are fitted on a data-supported breakpoint grid. Defaults permit up to five segments with at least 25 observations per segment. The best breakpoint placement is identified for each supported segment count. Each family is simplified sequentially while the next lower segment count increases MAE by no more than 10%.

At the aggregate level, 100 observation-bootstrap resamples assess fit variability conditional on the fitted model and SHAP values. Supported functional forms and segment counts are compared across resamples before family simplification. Up to three surviving forms with mean bootstrap MAE within 10% of the best are retained and refitted to the complete dependence values. Valid-resample counts and selection decisions are exported. The required valid-resample fraction is configurable and defaults to 100%.

At the cluster level, the same candidate library is fitted directly to cluster-stratified SHAP values. After segment simplification, the simplest form within 10% of the lowest MAE is selected. All cluster analyses use explanations from the same aggregate model.

Conditional inference
---------------------
Curve-bootstrap intervals summarize variability in fitted-curve metrics conditional on the fixed model. They exclude uncertainty from model estimation and are percentile intervals for metric distributions, not confidence intervals for mean error. Parameter tests are conditional on the selected form and breakpoints and do not adjust for model selection, repeated observations, or multiple comparisons. Breakpoint precision is limited by the candidate grid.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.stats import f as f_distribution, t as t_distribution
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PatternFitter:
    def __init__(self, settings):
        self.settings = settings

    def _format_number(self, value, digits=3):
        if not np.isfinite(value):
            return 'n/a'
        return f'{value:.{digits}f}'


    def _ols_fit(self, design, y):
        column_scale = np.maximum(np.linalg.norm(design, axis=0), 1.0)
        coefficients = np.linalg.lstsq(design / column_scale, y, rcond=None)[0] / column_scale
        return coefficients, design @ coefficients


    def _candidate_breakpoints(self, x, max_candidates=None):
        """Return a compact, data-supported grid of possible breakpoints."""
        max_candidates = self.settings["breakpoint_candidates"] if max_candidates is None else max_candidates
        unique_x = np.unique(x)
        if unique_x.size < 3:
            return np.array([])
        valid = np.array([
            value for value in unique_x[:-1]
            if np.sum(x <= value) >= self.settings["min_points_per_segment"]
            and np.sum(x > value) >= self.settings["min_points_per_segment"]
        ])
        if valid.size <= max_candidates:
            return valid
        positions = np.linspace(0, valid.size - 1, max_candidates).round().astype(int)
        return np.unique(valid[positions])


    def _segments_are_supported(self, x, breakpoints):
        edges = np.r_[-np.inf, breakpoints, np.inf]
        counts = [np.sum((x > edges[i]) & (x <= edges[i + 1])) for i in range(len(edges) - 1)]
        return min(counts, default=0) >= self.settings["min_points_per_segment"]


    def _model_diagnostics(self, y, fitted, design=None, coefficients=None, effective_parameters=None):
        residuals = y - fitted
        n = y.size
        rss = float(np.sum(residuals ** 2))
        null_rss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(r2_score(y, fitted))
        mae = float(mean_absolute_error(y, fitted))
        rmse = float(np.sqrt(mean_squared_error(y, fitted)))
        parameter_count = int(effective_parameters if effective_parameters is not None else design.shape[1])

        # Conditional tests: useful descriptive diagnostics, but not selection criteria.
        f_stat = np.nan
        f_pvalue = np.nan
        df_model = max(parameter_count - 1, 0)
        df_residual = n - parameter_count
        if df_model > 0 and df_residual > 0 and rss > 0:
            f_stat = ((null_rss - rss) / df_model) / (rss / df_residual)
            f_pvalue = float(f_distribution.sf(max(f_stat, 0), df_model, df_residual))

        max_t_stat = np.nan
        min_t_pvalue = np.nan
        coefficient_covariance = None
        coefficient_df_residual = np.nan
        if (design is not None and coefficients is not None and design.shape[1] < n
                and np.linalg.matrix_rank(design) == design.shape[1]):
            coefficient_df_residual = n - design.shape[1]
            try:
                coefficient_covariance = (rss / coefficient_df_residual) * (np.linalg.pinv(design) @ np.linalg.pinv(design).T)
                standard_errors = np.sqrt(np.clip(np.diag(coefficient_covariance), 0, None))
                usable = standard_errors > 0
                t_values = np.abs(np.asarray(coefficients)[usable] / standard_errors[usable])
                if t_values.size:
                    max_t_stat = float(np.max(t_values))
                    min_t_pvalue = float(2 * t_distribution.sf(max_t_stat, coefficient_df_residual))
            except np.linalg.LinAlgError:
                coefficient_covariance = None

        return {
            'mae': mae, 'rmse': rmse, 'r2': r2, 'rss': rss,
            't_stat_max': max_t_stat, 't_pvalue_min': min_t_pvalue,
            'f_stat': f_stat, 'f_pvalue': f_pvalue,
            'n_parameters': parameter_count,
            'coefficient_covariance': coefficient_covariance,
            'coefficient_df_residual': coefficient_df_residual,
        }


    def _make_candidate(self, name, x, y, fitted, equation, design=None, coefficients=None,
                        breakpoints=(), effective_parameters=None):
        diagnostics = self._model_diagnostics(
            y, fitted, design=design, coefficients=coefficients,
            effective_parameters=effective_parameters,
        )
        return {
            'method': name,
            'fitted': fitted,
            'equation': equation,
            'breakpoints': tuple(float(value) for value in breakpoints),
            'coefficients': None if coefficients is None else np.asarray(coefficients, dtype=float),
            **diagnostics,
        }


    def _fit_polynomial(self, x, y, degree):
        center = float(np.mean(x))
        scale = max(float(np.std(x)), 1.0)
        z = (x - center) / scale
        scaled_design = np.polynomial.polynomial.polyvander(z, degree)
        if np.linalg.matrix_rank(scaled_design) < degree + 1:
            return None
        scaled_coefficients, fitted = self._ols_fit(scaled_design, y)
        transform = np.zeros((degree + 1, degree + 1))
        base = np.polynomial.Polynomial([-center / scale, 1 / scale])
        for power in range(degree + 1):
            converted = (base ** power).coef
            transform[:len(converted), power] = converted
        coefficients = transform @ scaled_coefficients
        design = scaled_design
        terms = []
        for power, coefficient in enumerate(coefficients):
            if power == 0:
                terms.append(f'{coefficient:.3f}')
            elif power == 1:
                terms.append(f'{coefficient:+.3f}x')
            else:
                terms.append(f'{coefficient:+.3f}x^{{{power}}}')
        names = {0: 'Constant', 1: 'Linear', 2: 'Quadratic', 3: 'Cubic', 4: 'Quartic'}
        equation = r'$y = ' + ' '.join(terms) + r'$'
        result = self._make_candidate(names[degree], x, y, fitted, equation, design, scaled_coefficients)
        covariance = result['coefficient_covariance']
        if covariance is not None:
            result['coefficient_covariance'] = transform @ covariance @ transform.T
        result['coefficients'] = coefficients
        result['polynomial_center'] = center
        result['polynomial_scale'] = scale
        result['scaled_coefficients'] = scaled_coefficients
        return result


    def _fit_logarithmic(self, x, y):
        if np.any(x <= -1):
            return None
        if np.unique(x).size < 2:
            return None
        transformed = np.log1p(x)
        design = np.column_stack([np.ones(x.size), transformed])
        coefficients, fitted = self._ols_fit(design, y)
        equation = rf'$y = {coefficients[0]:.3f} {coefficients[1]:+.3f}\ln(1+x)$'
        return self._make_candidate('Logarithmic', x, y, fitted, equation, design, coefficients)


    def _sigmoid_function(self, x, baseline, amplitude, steepness, midpoint):
        return baseline + amplitude * expit(steepness * (x - midpoint))


    def _fit_sigmoid(self, x, y):
        if np.unique(x).size < 4 or np.ptp(y) <= 1e-12:
            return None
        x_range = max(float(np.ptp(x)), 1.0)
        y_range = max(float(np.ptp(y)), 1e-6)
        initial = [float(np.quantile(y, 0.1)), float(np.quantile(y, 0.9) - np.quantile(y, 0.1)),
                   4 / x_range, float(np.median(x))]
        lower = [float(np.min(y) - 4 * y_range), -8 * y_range, -50 / x_range, float(np.min(x) - x_range)]
        upper = [float(np.max(y) + 4 * y_range), 8 * y_range, 50 / x_range, float(np.max(x) + x_range)]
        try:
            coefficients, _ = curve_fit(
                self._sigmoid_function, x, y, p0=initial, bounds=(lower, upper), maxfev=30000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        fitted = self._sigmoid_function(x, *coefficients)
        # Numerical Jacobian used only for conditional exploratory t diagnostics.
        epsilon = 1e-5
        jacobian = np.column_stack([
            (self._sigmoid_function(x, *(coefficients + np.eye(4)[j] * epsilon)) - fitted) / epsilon
            for j in range(4)
        ])
        equation = (rf'$y = {coefficients[0]:.3f}{coefficients[1]:+.3f}\,\sigma'
                    rf'\!\left({coefficients[2]:.3f}(x-{coefficients[3]:.3f})\right)$')
        return self._make_candidate('Sigmoid', x, y, fitted, equation, jacobian, coefficients)


    def _fit_continuous_piecewise(self, x, y, breakpoints, plateau=False):
        hinges = [np.maximum(0, x - breakpoint) for breakpoint in breakpoints]
        if plateau:
            last_hinge = hinges[-1]
            design = np.column_stack([
                np.ones(x.size), x - last_hinge,
                *[hinge - last_hinge for hinge in hinges[:-1]],
            ])
        else:
            design = np.column_stack([np.ones(x.size), x, *hinges])
        coefficients, fitted = self._ols_fit(design, y)
        method = f"{'Continuous plateau' if plateau else 'Continuous piecewise'} ({len(breakpoints) + 1} segments)"
        breakpoint_text = ', '.join(self._format_number(value, 2) for value in breakpoints)
        if plateau:
            equation = rf'$\mathrm{{Connected\ plateau}};\quad \tau = ({breakpoint_text});\quad \beta_{{\mathrm{{final}}}}=0$'
        else:
            equation = rf'$y = {coefficients[0]:.3f}{coefficients[1]:+.3f}x + \sum_j \Delta_j(x-\tau_j)_+;\quad \tau = ({breakpoint_text})$'
        # Each unknown breakpoint counts as an additional fitted quantity in the F diagnostic.
        return self._make_candidate(
            method, x, y, fitted, equation, design, coefficients, breakpoints,
            effective_parameters=design.shape[1] + len(breakpoints),
        )


    def _fit_step_model(self, x, y, breakpoints):
        edges = np.r_[-np.inf, breakpoints, np.inf]
        segment_ids = np.digitize(x, breakpoints, right=True)
        design = np.column_stack([segment_ids == segment for segment in range(len(edges) - 1)]).astype(float)
        coefficients, fitted = self._ols_fit(design, y)
        parts = []
        for segment, level in enumerate(coefficients):
            if segment == 0:
                region = f'x≤{breakpoints[0]:.2f}'
            elif segment == len(coefficients) - 1:
                region = f'x>{breakpoints[-1]:.2f}'
            else:
                region = f'{breakpoints[segment - 1]:.2f}<x≤{breakpoints[segment]:.2f}'
            parts.append(f'{level:.3f} [{region}]')
        return self._make_candidate(
            f'Piecewise-constant step ({len(coefficients)} ranges)', x, y, fitted,
            rf'$y = c_j\quad \mathrm{{within\ level}}\ j;\quad \tau = ({", ".join(self._format_number(value, 2) for value in breakpoints)})$', design, coefficients, breakpoints,
            effective_parameters=design.shape[1] + len(breakpoints),
        )


    def _fit_segmented_family(self, x, y, plateau=False, step=False):
        breakpoint_grid = self._candidate_breakpoints(x)
        best_by_segments = {}
        for segments in range(2, self.settings["max_segments"] + 1):
            for breakpoints in combinations(breakpoint_grid, segments - 1):
                if not self._segments_are_supported(x, breakpoints):
                    continue
                if step:
                    candidate = self._fit_step_model(x, y, breakpoints)
                else:
                    candidate = self._fit_continuous_piecewise(x, y, breakpoints, plateau=plateau)
                # Keep only the best placement at each segment count rather than
                # retaining fitted arrays for every breakpoint combination.
                if segments not in best_by_segments or candidate['mae'] < best_by_segments[segments]['mae']:
                    best_by_segments[segments] = candidate
        return list(best_by_segments.values())


    def _segmented_family_name(self, candidate):
        method = candidate['method']
        if method.startswith('Continuous piecewise'):
            return 'Continuous piecewise'
        if method.startswith('Continuous plateau'):
            return 'Continuous plateau'
        if method.startswith('Piecewise-constant step'):
            return 'Piecewise-constant step'
        return None


    def _segment_count(self, candidate):
        return int(candidate['method'].split('(')[1].split()[0])


    def reduce_segmented_candidates(self, candidates):
        """Sequentially simplify each segmented family without a material MAE loss.

        Starting from the best-fitting segment count, compare only the next simpler
        count. If its relative MAE increase is within the configured simplification tolerance,
        retain that simpler model and repeat. Stop at the first material loss.
        """
        reduced = []
        families = {}
        for candidate in candidates:
            family = self._segmented_family_name(candidate)
            if family is None:
                reduced.append(candidate)
            else:
                families.setdefault(family, []).append(candidate)

        for family_candidates in families.values():
            by_count = {self._segment_count(candidate): candidate for candidate in family_candidates}
            best_mae = min(candidate['mae'] for candidate in by_count.values())
            zero_tolerance = max(1e-12, max(candidate.get('shap_range', 1.0) for candidate in by_count.values()) * 1e-12)
            current = min((candidate for candidate in by_count.values() if candidate['mae'] <= best_mae + zero_tolerance),
                          key=self._segment_count)

            while self._segment_count(current) > min(by_count):
                simpler_count = self._segment_count(current) - 1
                simpler = by_count.get(simpler_count)
                if simpler is None:
                    break
                relative_mae_loss = 100 * (simpler['mae'] - current['mae']) / max(current['mae'], 1e-12)
                if relative_mae_loss <= self.settings["simplification_tolerance_pct"]:
                    current = simpler
                else:
                    break

            reduced.append(current)

        return reduced

    def fit_pattern_library(self, x, y, simplify=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size < 2:
            raise ValueError("At least two finite dependence observations are required.")

        candidates = [candidate for degree in range(5)
                      if (candidate := self._fit_polynomial(x, y, degree)) is not None]
        logarithmic = self._fit_logarithmic(x, y)
        if logarithmic is not None:
            candidates.append(logarithmic)
        sigmoid = self._fit_sigmoid(x, y)
        if sigmoid is not None:
            candidates.append(sigmoid)
        candidates.extend(self._fit_segmented_family(x, y, plateau=False))
        candidates.extend(self._fit_segmented_family(x, y, plateau=True))
        candidates.extend(self._fit_segmented_family(x, y, step=True))
        shap_range = float(np.max(y) - np.min(y))
        if shap_range <= 0:
            shap_range = 1.0
        for candidate in candidates:
            candidate['nmae_pct'] = 100 * candidate['mae'] / shap_range
            candidate['shap_range'] = shap_range
        return sorted(self.reduce_segmented_candidates(candidates) if simplify else candidates,
                      key=lambda candidate: candidate['nmae_pct'])


    def _within_relative_mae_allowance(self, candidates):
        """Return candidates within the configured relative MAE allowance."""
        best_mae = min(candidate['mae'] for candidate in candidates)
        limit = best_mae * (1 + self.settings["retention_tolerance_pct"] / 100)
        numerical_tolerance = max(1e-12, max(candidate.get('shap_range', 1.0) for candidate in candidates) * 1e-12)
        return [candidate for candidate in candidates if candidate['mae'] <= limit + numerical_tolerance]


    def retained_candidates(self, candidates):
        """Select the simplest candidate whose MAE is within the relative allowance."""
        eligible = self._within_relative_mae_allowance(candidates)
        eligible.sort(key=lambda candidate: (candidate['n_parameters'], candidate['mae']))
        return eligible[:self.settings["max_citywide_forms"]]


    def display_candidates(self, candidates):
        """Return the configured number of best-fitting eligible candidates."""
        eligible = self._within_relative_mae_allowance(candidates)
        eligible.sort(key=lambda candidate: candidate['mae'])
        return eligible[:self.settings["max_citywide_forms"]]



    def _conditional_contrast_test(self, candidate, label, contrast):
        """Conditional t-test for a coefficient or linear coefficient contrast."""
        covariance = candidate.get('coefficient_covariance')
        coefficients = candidate.get('coefficients')
        df_residual = candidate.get('coefficient_df_residual', np.nan)
        if covariance is None or coefficients is None or not np.isfinite(df_residual) or df_residual <= 0:
            return None
        contrast = np.asarray(contrast, dtype=float)
        estimate = float(contrast @ coefficients)
        variance = float(contrast @ covariance @ contrast)
        if variance <= 0 or not np.isfinite(variance):
            return None
        standard_error = float(np.sqrt(variance))
        t_value = estimate / standard_error
        p_value = float(2 * t_distribution.sf(abs(t_value), df_residual))
        return {'Parameter / contrast': label, 'Estimate': estimate,
                'Standard error': standard_error, 't-value': t_value, 'p-value': p_value}


    def conditional_parameter_tests(self, candidate):
        """Relevant conditional tests; selected forms/breakpoints are treated as fixed."""
        method = candidate['method']
        coefficients = candidate.get('coefficients')
        if coefficients is None:
            return []
        tests = []

        def add(label, contrast):
            result = self._conditional_contrast_test(candidate, label, contrast)
            if result is not None:
                tests.append(result)

        if method == 'Linear':
            add('Slope (x)', [0, 1])
        elif method in {'Quadratic', 'Cubic', 'Quartic'}:
            for power in range(1, len(coefficients)):
                contrast = np.zeros(len(coefficients)); contrast[power] = 1
                add(f'Coefficient of x^{power}', contrast)
        elif method == 'Logarithmic':
            add('Log coefficient', [0, 1])
        elif method == 'Sigmoid':
            # Amplitude and steepness describe the presence of a sigmoid effect.
            # The midpoint test (x0 = 0) is scale-dependent and is not reported.
            for index, label in enumerate(['Amplitude', 'Steepness'], start=1):
                contrast = np.zeros(len(coefficients)); contrast[index] = 1
                add(label, contrast)
        elif method.startswith('Continuous piecewise') or method.startswith('Continuous plateau'):
            breakpoints = tuple(candidate['breakpoints'])
            plateau = method.startswith('Continuous plateau')
            for segment in range(len(breakpoints) + 1):
                contrast = np.zeros(len(coefficients))
                if not (plateau and segment == len(breakpoints)):
                    contrast[1] = 1
                    contrast[2:segment + 1] = 1
                if np.any(contrast):
                    add(f'Segment {segment + 1} slope', contrast)
        elif method.startswith('Piecewise-constant step'):
            for index, breakpoint in enumerate(candidate['breakpoints']):
                contrast = np.zeros(len(coefficients))
                contrast[index:index + 2] = [-1, 1]
                add(f'Level jump at x = {breakpoint:.2f}', contrast)
        return tests



