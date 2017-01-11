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


def js_inject_graph_on_click(div_ids):
    """ Code to define on-click functions for all plots and link
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


def create_html_document(figures, link_plots=False):
    """ Given a list of plotly graph objects, return a hmtl document
    where each plot (figure) is drawn in a div element. Base plotly
    library and additional script are included.

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
                              js_inject_graph_on_click(div_ids)])

    # # # TODO: resize scripts not needed?
    # # # TODO: image downloads

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

if __name__ == "__main__":
    # print all possible graph objects
    print(graph_objs.graph_objs.__all__)

    """ Scatter-plot matrix """
    # load 3D approximation sets
    lin_approx_set = np.loadtxt("../MOViE-mnozice/3d.linear.txt")
    sph_approx_set = np.loadtxt("../MOViE-mnozice/3d.spherical.txt")

    # create a pandas dataframe
    df = pd.DataFrame(np.row_stack((lin_approx_set, sph_approx_set)),
                      columns=['x', 'y', 'z'])

    # denote the two data sets
    df['Sets'] = pd.Series(['Linear'] * lin_approx_set.shape[0] +
                           ['Spherical'] * sph_approx_set.shape[0])

    # create plotly figure
    fig = FF.create_scatterplotmatrix(df, index='Sets', diag='box',
                                      size=10, height=1000, width=1000)

    # plot
    offline.plot(fig, show_link=False, filename="scatter_plot_matrix.html",
                 auto_open=False)

    """ 3d Scatter-plot """
    knee_approx_set = np.loadtxt("../MOViE-mnozice/koleno3d.txt")
    x, y, z = [knee_approx_set[:, i] for i in range(knee_approx_set.shape[1])]

    marker1 = dict(line=dict(color='rgba(130, 130, 130, 0.14)', width=0.5),
                   opacity=1, size=12)

    trace1 = graph_objs.Scatter3d(x=x, y=y, z=z, mode='markers',
                                  marker=marker1, hoverinfo='none')

    data = [trace1]

    layout = graph_objs.Layout(margin=dict(l=0, r=0, b=0, t=0),
                               hovermode='closest', title='3d Scatter Plot')

    fig = graph_objs.Figure(data=data, layout=layout)

    offline.plot(fig, show_link=False, filename="3dscatter_with_knee.html",
                 auto_open=False)

    """ 3d Bubble Chart """
    lin_approx_set_4d = np.loadtxt("../MOViE-mnozice/4d.linear.300.txt")
    x, y, z, w = [lin_approx_set_4d[:, i] for i in range(lin_approx_set_4d.shape[1])]

    marker1 = dict(sizemode='diameter', sizeref=0.03, size=w, color=w,
                   colorscale='Viridis', colorbar=dict(title='w'),
                   line=dict(color='rgb(140, 140, 170)'))

    trace1 = graph_objs.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker1)

    data = [trace1]

    layout = dict(width=800, height=800, title='3d Bubble Chart',
                  scene=dict(xaxis=dict(title='x', titlefont=dict(color='Orange')),
                             yaxis=dict(title='y', titlefont=dict(color='rgb(220, 220, 220)')),
                             zaxis=dict(title='z', titlefont=dict(color='rgb(220, 220, 220)')),
                             bgcolor='rgb(20, 24, 54)'))

    fig = dict(data=data, layout=layout)
    offline.plot(fig, show_link=False, filename='3d_bubble_chart.html',
                 auto_open=False)

    """ Scatter plot with PCA dim. reduction """
    lin_approx_set_4d = np.loadtxt("../MOViE-mnozice/4d.linear.300.txt")

    pca = PCA(n_components=2)
    x = lin_approx_set_4d
    x = (x - np.mean(x, 0)) / np.std(x, 0)
    x_r_1 = pca.fit(x).transform(x)

    sph_approx_set_4d = np.loadtxt("../MOViE-mnozice/4d.spherical.300.txt")
    x = sph_approx_set_4d
    x = (x - np.mean(x, 0)) / np.std(x, 0)
    x_r_2 = pca.fit(x).transform(x)

    trace1 = graph_objs.Scatter(x=x_r_1[:, 0], y=x_r_1[:, 1], mode='markers')
    trace2 = graph_objs.Scatter(x=x_r_2[:, 0], y=x_r_2[:, 1], mode='markers')

    data = [trace1, trace2]
    layout = graph_objs.Layout(autosize=False, width=900, height=700,
                               hovermode='closest', title='PCA Scatter Plot',
                               margin=dict(l=0, r=0, b=0, t=0))

    fig = dict(data=data, layout=layout)

    offline.plot(fig, show_link=False, filename='PCA_scatter_plot.html',
                 auto_open=False)
