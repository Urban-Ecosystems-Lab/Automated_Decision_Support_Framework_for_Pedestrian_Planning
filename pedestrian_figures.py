"""Visual representation of pattern panels, SHAP rankings, cluster maps, and importance plots."""
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
import numpy as np
from scipy.special import expit

from pedestrian_patterns import PatternFitter

CITY_COLORS = ['#173F5F', '#C73E1D', '#2E7D32']
CLUSTER_COLORS = ['#F0A202', '#7E57C2', '#00A6A6', '#D81B60', '#6D4C41']
conditional_parameter_tests = PatternFitter({}).conditional_parameter_tests
_sigmoid_function = PatternFitter({})._sigmoid_function


def apply_style():
    """Apply the typography and line styling used in the study notebook."""
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 18,
        'axes.titlesize': 22, 'axes.labelsize': 20,
        'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 15,
    })


def cluster_palette(count):
    """Preserve notebook cluster colors and extend them for additional groups."""
    extra = list(plt.get_cmap('tab20').colors)
    colors = CLUSTER_COLORS + extra
    return [colors[index % len(colors)] for index in range(count)]

def _range_label(index, breakpoints):
    if index == 0:
        return rf'$x \leq {breakpoints[0]:.2f}$'
    if index == len(breakpoints):
        return rf'$x > {breakpoints[-1]:.2f}$'
    return rf'${breakpoints[index - 1]:.2f} < x \leq {breakpoints[index]:.2f}$'

def _continuous_piecewise_value(x_value, coefficients, breakpoints, plateau):
    hinges = np.maximum(0, x_value - np.asarray(breakpoints, dtype=float))
    if plateau:
        last_hinge = hinges[-1]
        row = np.array([1.0, x_value - last_hinge,
                        *[hinge - last_hinge for hinge in hinges[:-1]]])
    else:
        row = np.array([1.0, x_value, *hinges])
    return float(row @ coefficients)

def _short_method_name(method):
    """Publication-style names for fitted pattern labels."""
    if method.startswith('Continuous piecewise'):
        count = int(method.split('(')[1].split()[0])
        return f'{count}-segment piecewise-linear'
    if method.startswith('Continuous plateau'):
        count = int(method.split('(')[1].split()[0])
        return f'{count}-segment plateau model'
    if method.startswith('Piecewise-constant step'):
        count = int(method.split('(')[1].split()[0])
        return f'{count}-level step function'
    return method

def _visual_test_pvalues(candidate):
    tests = conditional_parameter_tests(candidate)
    allowed = None
    if candidate['method'] == 'Sigmoid':
        allowed = {'Amplitude', 'Steepness'}
    elif candidate['method'].startswith('Continuous piecewise') or candidate['method'].startswith('Continuous plateau'):
        allowed = {test['Parameter / contrast'] for test in tests
                   if test['Parameter / contrast'].startswith('Segment ')}
    return {test['Parameter / contrast']: test['p-value'] for test in tests
            if allowed is None or test['Parameter / contrast'] in allowed}

def _visual_latex_number(value, signed=False, p_value=np.nan):
    text = f'{value:+.3f}' if signed else f'{value:.3f}'
    if p_value < 0.05:
        return rf'\mathbf{{{text}}}'
    return text

def _visual_display_equation(candidate):
    coefficients = candidate.get('coefficients')
    if coefficients is None:
        return candidate['equation']
    test_pvalues = _visual_test_pvalues(candidate)
    method = candidate['method']
    if method == 'Linear':
        slope = _visual_latex_number(coefficients[1], signed=True, p_value=test_pvalues.get('Slope (x)', np.nan))
        return rf'$y = {coefficients[0]:.3f}{slope}x$'
    if method == 'Logarithmic':
        coefficient = _visual_latex_number(coefficients[1], signed=True, p_value=test_pvalues.get('Log coefficient', np.nan))
        return rf'$y = {coefficients[0]:.3f} {coefficient}\ln(1+x)$'
    if method in {'Quadratic', 'Cubic', 'Quartic'}:
        terms = [f'{coefficients[0]:.3f}']
        for power, coefficient in enumerate(coefficients[1:], start=1):
            label = f'Coefficient of x^{power}'
            term = _visual_latex_number(coefficient, signed=True, p_value=test_pvalues.get(label, np.nan))
            terms.append(rf'{term}x' if power == 1 else rf'{term}x^{{{power}}}')
        return r'$y = ' + ' '.join(terms) + r'$'
    if method == 'Sigmoid':
        amplitude_sign = '+' if coefficients[1] >= 0 else '-'
        amplitude = _visual_latex_number(abs(coefficients[1]),
                                         p_value=test_pvalues.get('Amplitude', np.nan))
        exponent = _visual_latex_number(-coefficients[2], signed=True,
                                       p_value=test_pvalues.get('Steepness', np.nan))
        midpoint_term = f'x{-coefficients[3]:+.3f}'
        return (rf'$y = {coefficients[0]:.3f} {amplitude_sign} '
                rf'\frac{{{amplitude}}}{{1 + \exp\left[{exponent}({midpoint_term})\right]}}$')
    return candidate['equation']

def _visual_sigmoid_equation_parts(candidate):
    """Return separately sized prefix and logistic fraction for display."""
    coefficients = candidate['coefficients']
    test_pvalues = _visual_test_pvalues(candidate)
    amplitude_sign = '+' if coefficients[1] >= 0 else '-'
    amplitude = _visual_latex_number(abs(coefficients[1]),
                                     p_value=test_pvalues.get('Amplitude', np.nan))
    exponent = _visual_latex_number(-coefficients[2], signed=True,
                                   p_value=test_pvalues.get('Steepness', np.nan))
    midpoint_term = f'x{-coefficients[3]:+.3f}'
    prefix = rf'$y = {coefficients[0]:.3f} {amplitude_sign}$'
    fraction = rf'$\frac{{{amplitude}}}{{1 + \exp\left[{exponent}({midpoint_term})\right]}}$'
    return prefix, fraction

def _visual_piecewise_rows(candidate, x_min, x_max):
    breakpoints = tuple(candidate['breakpoints'])
    if not breakpoints:
        return None
    coefficients = candidate['coefficients']
    if candidate['method'].startswith('Piecewise-constant step'):
        return [(f'{level:.3f}', _range_label(index, breakpoints).removeprefix('$').removesuffix('$'))
                for index, level in enumerate(coefficients)]
    plateau = candidate['method'].startswith('Continuous plateau')
    test_pvalues = _visual_test_pvalues(candidate)
    edges = (x_min, *breakpoints, x_max)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right <= left:
            continue
        y_left = _continuous_piecewise_value(left, coefficients, breakpoints, plateau)
        y_right = _continuous_piecewise_value(right, coefficients, breakpoints, plateau)
        slope = (y_right - y_left) / (right - left)
        intercept = y_left - slope * left
        slope_text = _visual_latex_number(slope, signed=True,
                                           p_value=test_pvalues.get(f'Segment {index + 1} slope', np.nan))
        formula = f'{intercept:.3f}' if abs(slope) < 5e-8 else f'{intercept:.3f}{slope_text}x'
        condition = _range_label(index, breakpoints).removeprefix('$').removesuffix('$')
        rows.append((formula, condition))
    return rows

def _visual_step_jump_lines(candidate):
    jump_pvalues = {result['Parameter / contrast']: result['p-value']
                    for result in conditional_parameter_tests(candidate)
                    if result['Parameter / contrast'].startswith('Level jump')}
    lines = []
    for index, breakpoint in enumerate(candidate['breakpoints']):
        jump = candidate['coefficients'][index + 1] - candidate['coefficients'][index]
        label = f'Level jump at x = {breakpoint:.2f}'
        jump_text = _visual_latex_number(jump, signed=True, p_value=jump_pvalues.get(label, np.nan))
        lines.append(rf'$\Delta_{{{breakpoint:.2f}}} = {jump_text}$')
    return lines

def _curve_values(candidate, x):
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

def build_pattern_figure(feature_name, values_x, values_shap, labels, result):
    """Build the notebook's citywide, statistics, cluster-scatter and cluster-fit panels."""
    apply_style()
    candidate_colors = [CITY_COLORS[index % len(CITY_COLORS)] for index in range(len(result['citywide']))]
    shown_city_candidates = result['citywide']
    metric_intervals = result.get('bootstrap_metrics', {})
    groups = list(result['clusters'])
    context_colors = cluster_palette(len(groups))
    group_results = []
    for group in groups:
        mask = labels == group
        group_results.append({
            'group': group, 'x': values_x[mask], 'shap': values_shap[mask],
            'selected': result['clusters'][group],
            'n_intersections': result['cluster_sizes'][group],
        })
    all_values = [values_shap] + [c['fitted'] for c in shown_city_candidates]
    for group_result in group_results:
        all_values.append(group_result['selected']['fitted'])
    lower = min(float(np.min(v)) for v in all_values)
    upper = max(float(np.max(v)) for v in all_values)
    pad = max((upper - lower) * 0.07, 0.01)
    y_limits = (lower - pad, upper + pad)
    fig, axes = plt.subplots(2, 2, figsize=(30, 20), sharey='row',
                             gridspec_kw={'width_ratios': [0.95, 1.35]},
                             constrained_layout=False)
    FIGURE_FONT_SIZE = 28
    city_axis, statistics_axis = axes[0]
    group_points_axis, group_fit_axis = axes[1]
    order = np.argsort(values_x)
    city_axis.scatter(values_x, values_shap, s=70, alpha=0.28, color='0.40', edgecolors='none')
    for color, candidate in zip(candidate_colors, shown_city_candidates):
        if candidate['method'] == 'Sigmoid':
            curve_x = np.linspace(np.min(values_x), np.max(values_x), 400)
            curve_y = _sigmoid_function(curve_x, *candidate['coefficients'])
            city_axis.plot(curve_x, curve_y, lw=4.5, color=color)
        else:
            city_axis.plot(values_x[order], candidate['fitted'][order], lw=4.5, color=color)
    city_axis.set_title('Citywide SHAP values and retained patterns', fontsize=FIGURE_FONT_SIZE,
                        pad=18, fontweight='bold')

    statistics_axis.axis('off')
    text_y = 0.95
    for color, candidate in zip(candidate_colors, shown_city_candidates):
        rows = _visual_piecewise_rows(candidate, np.min(values_x), np.max(values_x))
        statistics_axis.text(0.02, text_y, _short_method_name(candidate['method']),
                             transform=statistics_axis.transAxes, va='top', ha='left',
                             fontsize=FIGURE_FONT_SIZE, color=color)
        line_y = text_y - 0.082
        if rows:
            row_step = 0.062
            brace_center = line_y - row_step * (len(rows) - 1) / 2
            brace_size = FIGURE_FONT_SIZE * (1.45 + 0.32 * len(rows))
            statistics_axis.text(0.03, brace_center, r'$y =$ ', transform=statistics_axis.transAxes,
                                 va='center', ha='left', fontsize=FIGURE_FONT_SIZE, color=color)
            statistics_axis.text(0.135, brace_center, r'$\{$', transform=statistics_axis.transAxes,
                                 va='center', ha='left', fontsize=brace_size, color=color)
            for row_index, (formula, condition) in enumerate(rows, start=1):
                statistics_axis.text(0.20, line_y,
                                     rf'${formula},\quad {condition.removeprefix("$").removesuffix("$")}$',
                                     transform=statistics_axis.transAxes, va='top', ha='left',
                                     fontsize=FIGURE_FONT_SIZE, color=color)
                line_y -= row_step
            if candidate['method'].startswith('Piecewise-constant step'):
                for jump_line in _visual_step_jump_lines(candidate):
                    statistics_axis.text(0.20, line_y, jump_line,
                                         transform=statistics_axis.transAxes, va='top', ha='left',
                                         fontsize=FIGURE_FONT_SIZE, color=color)
                    line_y -= row_step
        else:
            if candidate['method'] == 'Sigmoid':
                sigmoid_prefix, sigmoid_fraction = _visual_sigmoid_equation_parts(candidate)
                equation_box = HPacker(
                    children=[
                        TextArea(sigmoid_prefix, textprops={'fontsize': FIGURE_FONT_SIZE, 'color': color}),
                        TextArea(sigmoid_fraction, textprops={'fontsize': FIGURE_FONT_SIZE * 1.35, 'color': color}),
                    ],
                    align='center', pad=0, sep=7,
                )
                statistics_axis.add_artist(AnchoredOffsetbox(
                    loc='upper left', child=equation_box, frameon=False, pad=0, borderpad=0,
                    bbox_to_anchor=(0.05, line_y), bbox_transform=statistics_axis.transAxes,
                ))
                line_y -= 0.125
            else:
                statistics_axis.text(0.05, line_y, _visual_display_equation(candidate),
                                     transform=statistics_axis.transAxes, va='top', ha='left',
                                     fontsize=FIGURE_FONT_SIZE, color=color)
                line_y -= 0.062
        intervals = metric_intervals.get(candidate['method'])
        if intervals is None:
            metric_text = (f'Citywide MAE = {candidate["mae"]:.3f};  '
                           f'RMSE = {candidate["rmse"]:.3f};  R² = {candidate["r2"]:.3f}')
            interval_text = 'Bootstrap interval unavailable'
        else:
            metric_text = (f'Mean MAE = {intervals["mae"]["mean"]:.3f} '
                           f'[{intervals["mae"]["lower"]:.3f}, {intervals["mae"]["upper"]:.3f}]')
            interval_text = (f'Mean RMSE = {intervals["rmse"]["mean"]:.3f} '
                             f'[{intervals["rmse"]["lower"]:.3f}, {intervals["rmse"]["upper"]:.3f}];  '
                             f'Mean R² = {intervals["r2"]["mean"]:.3f} '
                             f'[{intervals["r2"]["lower"]:.3f}, {intervals["r2"]["upper"]:.3f}]')
        statistics_axis.text(0.05, line_y, metric_text,
                             transform=statistics_axis.transAxes, va='top', ha='left',
                             fontsize=FIGURE_FONT_SIZE, color=color)
        statistics_axis.text(0.05, line_y - 0.062, interval_text,
                             transform=statistics_axis.transAxes, va='top', ha='left',
                             fontsize=FIGURE_FONT_SIZE, color=color)
        text_y = line_y - 0.16

    for result, color in zip(group_results, context_colors):
        group_points_axis.scatter(result['x'], result['shap'], s=75, alpha=0.36,
                                  color=color, edgecolors='none')
        group_order = np.argsort(result['x'])
        selected = result['selected']
        if selected['method'] == 'Sigmoid':
            curve_x = np.linspace(np.min(result['x']), np.max(result['x']), 400)
            curve_y = _sigmoid_function(curve_x, *selected['coefficients'])
            group_fit_axis.plot(curve_x, curve_y, lw=4.5, color=color,
                                label=f"Cluster {result['group']} (n={result['n_intersections']} intersections): {_short_method_name(selected['method'])}")
        else:
            group_fit_axis.plot(result['x'][group_order], selected['fitted'][group_order], lw=4.5,
                                color=color,
                                label=f"Cluster {result['group']} (n={result['n_intersections']} intersections): {_short_method_name(selected['method'])}")
    for axis in (city_axis, group_points_axis, group_fit_axis):
        axis.axhline(0, color='black', lw=1.6, ls='--', alpha=0.75)
        axis.set_ylim(y_limits)
        axis.set_xlabel(feature_name, fontsize=FIGURE_FONT_SIZE)
        axis.set_ylabel('SHAP value', fontsize=FIGURE_FONT_SIZE)
        axis.tick_params(labelsize=FIGURE_FONT_SIZE)
        axis.grid(alpha=0.18)
        axis.spines[['top', 'right']].set_visible(False)
    group_points_axis.set_title('SHAP values stratified by cluster', fontsize=FIGURE_FONT_SIZE,
                                pad=18, fontweight='bold')
    group_fit_axis.set_title('Best fitted pattern in each cluster', fontsize=FIGURE_FONT_SIZE,
                             pad=18, fontweight='bold')
    legend_location = 'best'
    group_fit_axis.legend(loc=legend_location, frameon=True, fontsize=max(12, FIGURE_FONT_SIZE * min(1, 4 / len(groups))),
                          title='Selected pattern', title_fontsize=FIGURE_FONT_SIZE)
    # Reserve a narrow lower margin for the significance note.
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.94, hspace=0.30, wspace=0.10)
    fig.text(0.025, 0.018,
             r'Bold parameters are statistically significant at $p < 0.05$.',
             ha='left', va='bottom', fontsize=FIGURE_FONT_SIZE * 0.86, color='0.25')
    return fig



def plot_patterns(feature, x, shap_y, labels, result, figures, config):
    """Save the four-panel comparison for one automatically ranked feature."""
    fig = build_pattern_figure(feature, x, shap_y, labels, result)
    destination = Path(figures) / 'patterns'
    destination.mkdir(exist_ok=True)
    import re
    filename = re.sub(r'[^a-z0-9]+', '_', feature.lower()).strip('_') + '.png'
    fig.savefig(destination / filename, dpi=config['figures']['dpi'])
    plt.close(fig)


def plot_overview(importance, shap_values, labels, intersections, scores, gap_table, config, figures):
    """Save SHAP rankings, the importance-drop decision and the context map."""
    apply_style()
    figures = Path(figures)
    dpi = config['figures']['dpi']
    top = importance.head(config['patterns']['top_n'])
    positions = np.arange(len(top))
    fig, axis = plt.subplots(figsize=(15, 12))
    bars = axis.barh(positions, top.values, height=0.66, alpha=0.9, color='#55A868')
    axis.set_yticks(positions)
    axis.set_yticklabels([fill(label, width=40) for label in top.index], fontsize=16)
    axis.set_xlabel('Mean |SHAP|', fontsize=22)
    axis.set_title('Top SHAP Features', fontsize=26, pad=14)
    axis.invert_yaxis()
    axis.set_xlim(0, max(float(top.max()), 1e-12) * 1.12)
    for bar, value in zip(bars, top.values):
        axis.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                  f'{value:.3f}', va='center', ha='left', fontsize=15, clip_on=False)
    axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / 'shap_ranking.png', dpi=dpi)
    plt.close(fig)

    average = np.mean([np.abs(shap_values[labels == group]).mean(axis=0)
                       for group in np.unique(labels)], axis=0)
    lookup = dict(zip(importance.attrs['columns'], average))
    display = top.sort_values()
    positions = np.arange(len(display))
    fig, axis = plt.subplots(figsize=(15, 13))
    cluster_bars = axis.barh(positions - 0.18, [lookup[c] for c in display.index],
                             height=0.36, color='#4C78A8', alpha=0.92,
                             label='Cluster-level mean |SHAP|')
    city_bars = axis.barh(positions + 0.18, display.values,
                          height=0.36, color='#59A14F', alpha=0.92,
                          label='Citywide mean |SHAP|')
    axis.set_yticks(positions)
    axis.set_yticklabels(display.index, fontsize=15)
    axis.set_xlabel('Mean |SHAP|', fontsize=18)
    axis.tick_params(axis='x', labelsize=15)
    axis.legend(loc='lower right', frameon=True, fontsize=15)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='x', alpha=0.16)
    axis.set_axisbelow(True)
    maximum = max(float(display.max()), max(lookup[c] for c in display.index), 1e-12)
    axis.set_xlim(0, maximum * 1.14)
    for bars in [cluster_bars, city_bars]:
        for bar in bars:
            axis.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                      f'{bar.get_width():.3f}', va='center', ha='left', fontsize=13)
    fig.tight_layout()
    fig.savefig(figures / 'shap_ranking_by_cluster.png', dpi=dpi)
    plt.close(fig)

    visible = gap_table.head(min(len(gap_table), config['clustering']['gap_search_top_n'] + 1))
    count = int(gap_table.Selected.sum())
    decision = gap_table.iloc[count - 1]
    gap = decision['Drop to next (%)']
    qualifies = bool(decision['Searched gap'] and np.isfinite(gap)
                     and gap >= config['clustering']['minimum_gap_pct'] and gap > 0)
    title = f'Selected cutoff: {count} features'
    detail = f'{gap:.1f}% drop' if qualifies else 'No qualifying drop; configured fallback'
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(visible.Rank, visible['Mean absolute SHAP'], marker='o', lw=2.8, ms=9, color='#55A868')
    axis.axvline(count + 0.5, lw=2.5, ls='--', color='#C73E1D', label=title + '\n' + detail)
    axis.set_xlabel('Feature rank', fontsize=18)
    axis.set_ylabel('Mean |SHAP|', fontsize=18)
    axis.set_title('Clustering Feature Selection', fontsize=20, pad=12)
    axis.set_xticks(visible.Rank)
    axis.tick_params(labelsize=15)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=0.22)
    axis.legend(fontsize=14, frameon=True)
    fig.tight_layout()
    fig.savefig(figures / 'feature_importance_drop.png', dpi=dpi)
    plt.close(fig)

    import geopandas as gpd
    points = gpd.GeoDataFrame(intersections, geometry=gpd.points_from_xy(
        intersections['Longitude'], intersections['Latitude']), crs=4326).to_crs(3857)
    minx, miny, maxx, maxy = points.total_bounds
    pad_x, pad_y = (maxx-minx)*0.05 or 1, (maxy-miny)*0.05 or 1
    xmin, xmax, ymin, ymax = minx-pad_x, maxx+pad_x, miny-pad_y, maxy+pad_y
    ymax -= config['figures'].get('map_crop_top_fraction', 0.30) * (ymax-ymin)
    fig, axis = plt.subplots(figsize=(9, 14))
    groups = sorted(points.Cluster.unique())
    handles = []
    for group, color in zip(groups, cluster_palette(len(groups))):
        subset = points[(points.Cluster == group) & (points.geometry.y <= ymax)]
        axis.scatter(subset.geometry.x, subset.geometry.y, s=48, c=[color], alpha=0.92,
                     edgecolors='white', linewidths=1.35, zorder=3)
        handles.append(Line2D([0], [0], marker='o', linestyle='', markerfacecolor=color,
                             markeredgecolor='black', markeredgewidth=0.4, markersize=np.sqrt(42),
                             label=f'Cluster {group} (n={(points.Cluster == group).sum()} intersections)'))
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    if config['figures'].get('basemap', True):
        import contextily as ctx
        try:
            ctx.add_basemap(axis, source=ctx.providers.Esri.WorldGrayCanvas,
                            crs=points.crs.to_string(), attribution_size=5)
        except Exception as error:
            import warnings
            warnings.warn(f'Basemap unavailable: {error}')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect('equal', adjustable='box')
    axis.legend(handles=handles, title='Clusters', frameon=True, fontsize=16, title_fontsize=18, loc='upper right')
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / 'clusters.png', dpi=dpi)
    plt.close(fig)
