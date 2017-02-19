import os
import re
from collections import defaultdict
from pkg_resources import resource_string

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as graph_objs
from plotly.tools import FigureFactory as FF


def js_get_plotly():
    """ Minimized plotly.js library.

    :return:
        res : string
    """
    path = os.path.join('offline', 'plotly.min.js')
    plotlyjs = resource_string('plotly', path).decode('utf-8')

    res = ''.join([
        '<script type="text/javascript">',
        plotlyjs,
        '</script>'
    ])

    return res


def js_inject_graph_variables(div_ids):
    """ Code to define plot variables.

    :param div_ids: a list of unique div identifiers

    :return:
        res : string
    """
    if not div_ids:
        return ''

    res = '\n'.join([
        '<script type="text/javascript">',
        *['var plot{counter} = document.getElementById(\'{div_id}\');'.format(
            counter=i, div_id=div_id) for i, div_id in enumerate(div_ids)],
        'var all_plots = [{}];'.format(', '.join('plot' + str(i) for i in
                                                 range(len(div_ids)))),
        '</script>'
    ])

    return res


def js_inject_link_selection(div_ids):
    """ Code to define on-click functions for all plots in order to link
    selection.

    :param div_ids: a list of unique div identifiers

    :return:
        res : string
    """
    if not div_ids:
        return ''

    res = '\n'.join([
        '<script type="text/javascript">',
        'var redrawing = false;',
        *[('plot{counter}.on'.format(counter=i) +
           '(\'plotly_click\', function (data){'
           'if (!redrawing){'
           'var pt_number = data.points[0].pointNumber;'
           'var trace_name = data.points[0].data.name;'
           'redrawing = true;'
           'update_and_restyle(pt_number, trace_name);'
           '} else {redrawing=false;}'
           '});') for i, div_id in enumerate(div_ids)],
        'function update_and_restyle(a, name){'
        'for (var i = 0; i < all_plots.length; i++) {'
        'var colors = [];'
        'var update_needed = false;'
        'for (var j = 0; j < all_plots[i].data.length; j++){'
        'var trace = all_plots[i].data[j];'
        'var trace_color_array = trace.marker.color;'
        'if (trace.name == name){'
        'update_needed = true;'
        'trace_color_array[a] = invert_color(trace_color_array[a]);}'
        'colors.push(trace_color_array);}'
        'if (update_needed)'
        'Plotly.restyle(all_plots[i], \'marker.color\', colors);}}',
        'function invert_color(color){'
        'if (color == \'red\') return \'green\';'
        'return \'red\';}',
        '</script>'
    ])

    return res


def create_custom_html_document(figures, link_plots=False):
    """ Given a list of plot.ly graph objects, create a HTML document
    where each plot (figure) is drawn in a div element. Base plot.ly
    library and additional scripts are included.

    :param figures: a list of figures
    :param link_plots: if True, scripts are injected to enable selection
    of points across all created plots. In this instance, traces must be
    uniquely named and each marker color must be forwarded as a list of
    colors (strings). If two traces hold the same name, they are presumed
    to display the same data.
    """
    divs = [offline.plot(fig, output_type='div', show_link=False,
                         include_plotlyjs=False) for fig in figures]

    # regex matching deemed safer than calls to
    # protected functions (offline._plot_html)
    div_ids = [re.search(r'div id=[\"\']([\w\-]+)[\"\']', div).group(1)
               for div in divs]

    extra_scripts = []

    if link_plots:
        extra_scripts.extend([js_inject_graph_variables(div_ids),
                              js_inject_link_selection(div_ids)])

    # # # TODO: resize scripts not needed?
    # # # TODO: pdf image downloads

    with open('output.html', 'w') as f:
        f.write(''.join([
            '<html>',
            '<head><meta charset="utf-8" /></head>',
            '<body>',
            js_get_plotly(),
            *divs,
            *extra_scripts,
            '</body>',
            '</html>']))


def custom_assertions(visualization_method):
    def inner(approximation_sets, names=None, *args):
        assert (approximation_sets and len(approximation_sets) == len(names)
                if names else True)
        assert len(set(a.shape[1] for a in approximation_sets)) == 1
        return visualization_method(approximation_sets, names, *args)

    return inner


@custom_assertions
def hyper_space_diagonal_counting(approximation_sets, names=None, num_bins=5):

    # TODO: documentation, axis naming

    def index_pairing(binned_vec):
        def pair():  # Cantor's pairing function
            return int((x+y) * (x+y+1) / 2 + y)

        def another_pair():  # http://szudzik.com/ElegantPairing.pdf
            return int((x**2 + 3*x + 2*x*y + y + y**2) / 2)

        def elegant_pair():  # http://szudzik.com/ElegantPairing.pdf
            return int(y**2 + x if x < y else x**2 + x + y)

        num_dim = binned_vec.size
        if num_dim == 1:
            return binned_vec.tolist()
        if num_dim == 2:
            x, y = binned_vec
            return [another_pair()]

        return index_pairing(index_pairing(binned_vec[:int(num_dim/2)]) +
                             index_pairing(binned_vec[int(num_dim/2):])
                             if binned_vec.size % 2 == 1 else
                             index_pairing(binned_vec[:2]) +
                             index_pairing(binned_vec[2:]))

    # determine bins for each of the objective functions
    combined = np.row_stack(approximation_sets)
    col_min, col_max = np.amin(combined, axis=0), np.amax(combined, axis=0)
    col_bins = [np.linspace(min_, max_, num_bins + 1)
                for min_, max_ in zip(col_min, col_max)]

    # reformulate according to the previous step
    approximation_sets = [np.column_stack(np.digitize(col, col_bins[i])
                                          for i, col in enumerate(a.T)) - 1
                          for a in approximation_sets]

    # if necessary, represent multiple functions on each single axis
    approximation_sets = (approximation_sets if combined.shape[1] == 2 else
                          [np.row_stack(index_pairing(vec[:int(vec.size/2)]) +
                                        index_pairing(vec[int(vec.size/2):])
                                        for vec in a)
                           for a in approximation_sets])

    # there may only be two dimensions remaining
    assert all(a.shape[1] == 2 for a in approximation_sets)

    # create figure
    data = []
    combined = np.row_stack(approximation_sets)
    max_x, max_y = np.amax(combined, axis=0)
    for i, a in enumerate(approximation_sets):
        temp = np.zeros((max_x+1, max_y+1))
        for vec in a:
            temp[vec[0], vec[1]] += 1

        x, y, z = [], [], []
        for x_, smh in enumerate(temp):
            for y_, cnt in enumerate(smh):
                x.extend([x_, x_, None])
                y.extend([y_, y_, None])
                z.extend([0, cnt, None])

        data.append(graph_objs.Scatter3d(x=x, y=y, z=z,
                                         name=names[i] if names else '',
                                         mode='lines'))

    layout = dict(title='Hyper Space Diagonal Counting',
                  xaxis=dict(title=''),
                  yaxis=dict(title=''))

    fig = dict(data=data, layout=layout)
    return [fig, ]


def distance_and_distribution_chart(approximation_sets, names=None,
                                    sort_by_col=0):
    """
    Visualization method.

    Plot the non-dominated solutions against their distances to the
    approximate Pareto front (distance chart) and their distances
    between each other (distribution chart). The approximation of the
    Pareto front is determined by considering all non-dominated vectors
    from the forwarded approximation sets.

    Parameters
    ----------
    approximation_sets : list of ndarray
        Visualize these approximation sets.
    names : list of str
        Optional list of approximation sets' names.
    sort_by_col : int
        Sort solutions by this objective (to calculate distributions).

    Returns
    -------
    figures : list of Figure
        Two figures are returned, a distance chart & a distribution
        chart.
    distances : list of ndarray
        For each approximation set, distances between the non-dominated
        solutions and the approximate Pareto front.
    distributions : list of ndarray
        For each approximation set, distances between the non-dominated
        solutions within the set, taking into consideration the distance
        between the boundary solutions and the approximate Pareto front.

    Notes
    -----
    Returned distance and distribution ndarrays for a given
    approximation set are ordered in the same manner.

    References
    ----------
    .. [1] "Visualization Technique for Analyzing Non-Dominated Set
           Comparison", Kiam Heong Ang, Gregory Chong, and Yun Li,
           University of Glasgow
    .. [2] "Recent Advances in Simulated Evolution and Learning",
           Kay Chen Tan, Meng Hiot Lim, Xin Yao, Lipo Wang,
           World Scientific, 2004
    """
    assert (approximation_sets and len(approximation_sets) == len(names)
            if names else True)

    assert (len(set(a.shape[1] for a in approximation_sets)) == 1
            and sort_by_col < approximation_sets[0].shape[1])

    # order the approximation sets by the chosen objective function
    approximation_sets = [a[a[:, sort_by_col].argsort()]
                          for a in approximation_sets]

    # determine all non-dominated vectors
    combined = np.row_stack(approximation_sets)
    non_dominated_indices = np.ones(combined.shape[0], dtype=bool)
    for i, vec in enumerate(combined):
        if non_dominated_indices[i]:
            non_dominated_indices[non_dominated_indices] = np.any(
                combined[non_dominated_indices] <= vec, axis=1)

    # Pareto front approximation ordered by the chosen objective function
    p_approx = combined[non_dominated_indices]
    p_approx = p_approx[p_approx[:, sort_by_col].argsort()]
    p_span = np.amax(p_approx, axis=0) - np.amin(p_approx, axis=0)

    # calculate split indices to later partition the distance vector
    cnt = 0
    splits = []
    for a in approximation_sets:
        splits.append((cnt, cnt + a.shape[0]))
        cnt += a.shape[0]

    # distance metric
    distances = np.array([0 if non_dominated_indices[i] else
                          min(np.linalg.norm((vec - n_d) / p_span)
                              for n_d in p_approx)
                          for i, vec in enumerate(combined)], dtype=float)

    distances = distances / distances.max()
    distances = [distances[s[0]:s[1]] for s in splits]

    # distribution metric
    distributions = []
    for a in approximation_sets:
        temp = np.row_stack((p_approx[0], a, p_approx[-1]))
        distributions.append([np.linalg.norm((temp[i] - temp[i+1]) / p_span)
                              for i in range(temp.shape[0] - 1)])

    d_max = np.hstack(distributions).max()
    distributions = [np.array(d) / d_max for d in distributions]

    # create figures
    data_distance_metric = []
    data_distribution_metric = []
    for i, a in enumerate(approximation_sets):
        data_distance_metric.append(graph_objs.Scatter(
            x=np.arange(len(distances[i])),
            y=distances[i],
            mode='lines',
            name=names[i] if names else ''))

        data_distribution_metric.append(graph_objs.Scatter(
            x=np.arange(len(distributions[i])),
            y=distributions[i],
            mode='lines',
            name=names[i] if names else ''))

    layout_distance_metric = dict(title='Distance chart',
                                  xaxis=dict(title=''),
                                  yaxis=dict(title='distance'))

    layout_distribution_metric = dict(title='Distribution chart',
                                      xaxis=dict(title=''),
                                      yaxis=dict(title='distribution'))

    fig_distance_metric = dict(data=data_distance_metric,
                               layout=layout_distance_metric)

    fig_distribution_metric = dict(data=data_distribution_metric,
                                   layout=layout_distribution_metric)

    return ([fig_distance_metric, fig_distribution_metric], distances,
            distributions)

if __name__ == "__main__":
    pass
