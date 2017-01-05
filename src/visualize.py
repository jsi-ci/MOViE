import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as graph_objs
from plotly.tools import FigureFactory as FF

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
