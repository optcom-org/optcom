# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

""".. moduleauthor:: Sacha Medaer"""

import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d  # unused import (for '3d')
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util

'''
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuples = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10)), (0, (1, 1)),
              (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (5, 1)),
              (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
              (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),
              (0, (3, 1, 1, 1, 1, 1))]

'''
# For color, see https://matplotlib.org/examples/color/named_colors.html
linecolors = ['violet', 'orange', 'red', 'greenyellow', 'silver', 'brown',
              'pink', 'gray', 'black', 'marroon', 'blue', 'navy', 'gold',
              'cyan', 'palegreen', 'deepskyblue', 'lime',]


plot3d_types = {"plot_surface": ("mesh", "color"),
                "plot_wireframe": ("mesh", "nocolor"),
                "contour3D": ("mesh", "nocolor"),
                "contourf3D": ("mesh", "nocolor"),
                "plot_trisurf": ("ravel", "color"),
                "plot3D": ("ravel", "color"), "scatter3D": ("ravel", "color")}


axis_labels = { "t" : "Time, $t \, (ps)$", \
               "nu" : r"Frequency, $\nu \, (THz)$", \
               "Lambda" : r"Wavelength, $\lambda \, (nm)$", \
               "P_t" : "Power, $|A(z, t)|^2 \, (W)$", \
               "P_nu" : r"Power, $|\tilde{A}(z, \nu)|^2 \, (a.u.)$", \
               "P_lambda" : r"Power, $|\tilde{A}(z, \lambda|^2 \, (a.u.)$$", \
               "z" : "Fibre length, $z \, (km)$", \
               "phi" : "Phase, $\phi(t) \, (rad)$", \
               "chirp" : "Frequency chirp, $\delta \omega \, (rad / ps)$", \
               "t_normal" : r"Normalised time, $\frac{t}{T_0}$", \
               "xi" : r"Normalised distance, $\xi = \frac{z}{L_D}$", \
               "xi_prime" : r"Normalised distance, $\xi' = \frac{z}{L_D'}$",\
               "sigma_a" : r"Cross sections, $\sigma_a \, (nm^2)$",\
               "sigma_e" : r"Cross sections, $\sigma_e \, (nm^2)$",\
               "beta2" : r"$\beta_2 \, (ps^2 \cdot km^{-1})$",\
               "dispersion": r"Dispersion $(ps \cdot nm^{-1} \cdot km^{-1})$",\
               "dispersion_slope": r"Dispersion slope "
                                    "$(ps \cdot nm^{-2} \cdot km^{-1})$",
               "population": r"Population density $(m^{-3})$",
               "n_2": r"Non-linear index, $n_2 \, (m^2\cdot W^{-1})$",
               "gamma": r"Non-linear coefficient, $\gamma \,"
                         "(rad\cdot W^{-1}\cdot km^{-1})$",
               "h_R": r"Raman response, $h_R \, (ps^{-1})$" }


def check_axis_labels(labels_to_check: Optional[List[str]],
                      labels: Dict[str, str]):
    """Check if can assimilate given label to recorded ones."""
    if (labels_to_check is not None):
        for i in range(len(labels_to_check)):
            if (labels_to_check[i] in labels.keys()):
                labels_to_check[i] = labels.get(labels_to_check[i])

    return labels_to_check


def add_subplot_para(plt_to_add, x_label=None, y_label=None, z_label=None,
                     x_range=None, y_range=None, z_range=None,
                     plot_title=None) -> None:

    if (x_label != None):
        plt_to_add.set_xlabel(x_label)
    if (y_label != None):
        plt_to_add.set_ylabel(y_label)
    if (z_label != None):
        plt_to_add.set_zlabel(z_label)
    if (x_range != None):
        plt_to_add.set_xlim(x_range)
    if (y_range != None):
        plt_to_add.set_ylim(y_range)
    if (z_range != None):
        plt_to_add.set_zlim(z_range)
    if (plot_title != None):
        plt_to_add.set_title(plot_title)


def add_2D_subplot(plt_to_add, x_data, y_data, x_label, y_label, x_range,
                   y_range, plot_title, plot_label, plot_linestyle,
                   plot_color, opacity):
    x_data_temp = np.asarray(x_data)
    x_data = np.array([])
    y_data_temp = np.asarray(y_data)
    y_data = np.array([])
    x_data, y_data = util.auto_pad(x_data_temp, y_data_temp)
    multi_channel = len(y_data) > 1
    labels_on_plot = plot_label is not None
    colors_on_plot = plot_color is not None
    if (multi_channel):
        plot_label = util.make_list(plot_label, len(y_data))
    print('in plot', x_data.shape, y_data.shape)
    print(x_data)
    for i in range(len(y_data)):
        if (multi_channel):
            if (labels_on_plot):
                plot_label_temp = plot_label[i] + " (ch.{})".format(i)
            else:
                plot_label_temp = "channel {}".format(i)
        else:
            plot_label_temp = plot_label
        if (not colors_on_plot):
            plot_color = linecolors[add_2D_subplot.counter]
            add_2D_subplot.counter += 1
        if (labels_on_plot or multi_channel):
            plt_to_add.plot(x_data , y_data[i], ls=plot_linestyle,
                            c=plot_color, label=plot_label_temp)
        else:
            plt_to_add.plot(x_data , y_data[i], ls=plot_linestyle,
                            c=plot_color)
        plt_to_add.fill_between(x_data , y_data[i], alpha=opacity,
                                facecolor=plot_color)
        add_subplot_para(plt_to_add, x_label=x_label, y_label=y_label,
                         x_range=x_range, y_range=y_range,
                         plot_title=plot_title)
        if (labels_on_plot or multi_channel):
            plt_to_add.legend(loc = "best")


def add_3D_subplot(plt_to_add, x_data, y_data, z_data, x_label, y_label,
                   z_label, x_range, y_range, z_range, plot_title, plot_color,
                   opacity, plot_type):
    x_data_temp = np.asarray(x_data)
    print('in ploooooooooooooot', x_data_temp.shape)
    x_data_temp = np.asarray(x_data)
    if (x_data_temp.ndim > 1):  # Else single time array, nothing to pad
        if (x_data_temp.ndim == 2):
            temp = np.ones(z_data.shape)
            for i in range(len(temp)):
                temp[i] = (np.ones((z_data.shape[1], z_data.shape[2]))
                           * x_data_temp)
            x_data_temp = temp
        x_data = np.array([])
        z_data_temp = np.asarray(z_data)
        z_data = np.array([])
        x_data, z_data = util.auto_pad(x_data_temp, z_data_temp)
        colors_on_plot = plot_color is not None
    for i in range(len(z_data)):
        mesh_x, mesh_y = np.meshgrid(x_data, y_data[0])
        #if (not colors_on_plot):
        #    plot_color = linecolors[add_3D_subplot.counter]
        #    add_3D_subplot.counter += 1
        if (plot3d_types[plot_type][0] == 'mesh'):
            if (plot3d_types[plot_type][1] == 'color'):
                getattr(plt_to_add, plot_type)(mesh_x, mesh_y, z_data[i],
                                               color=plot_color,
                                               rcount=100, ccount=100,
                                               alpha=opacity)
            else:
                getattr(plt_to_add, plot_type)(mesh_x, mesh_y, z_data[i],
                                               rcount=100, ccount=100,
                                               alpha=opacity)
        else:
            ravel_x = np.ravel(mesh_x)
            ravel_y = np.ravel(mesh_y)
            ravel_z = np.ravel(z_data[i])
            if (plot3d_types[plot_type][1] == 'color'):
                getattr(plt_to_add, plot_type)(ravel_x, ravel_y, ravel_z,
                                              color=plot_color,
                                              alpha=opacity)
            else:
                getattr(plt_to_add, plot_type)(ravel_x, ravel_y, ravel_z,
                                               alpha=opacity)

        add_subplot_para(plt_to_add, x_label=x_label, y_label=y_label,
                         z_label=z_label, x_range=x_range, y_range=y_range,
                         z_range=z_range, plot_title=plot_title)


def get_graph_layout(plot_groups, split, length):
    nbr_graphs = 0
    graphs = []
    if (split is None):
        if (plot_groups is None):
            nbr_graphs = 1
            graphs = [[i for i in range(length)]]
        else:
            nbr_graphs = max(plot_groups) + 1
            graphs = [[] for i in range(nbr_graphs)]
            for i in range(len(plot_groups)):
                graphs[plot_groups[i]].append(i)
    else:
        if (split):
            nbr_graphs = length
            graphs = [[i] for i in range(length)]
        else:
            nbr_graphs = 1
            graphs = [[i for i in range(length)]]

    return graphs, nbr_graphs


def get_graph_structure(nbr_graphs):
    nbr_row = 0
    nbr_col  = 0
    if (nbr_graphs < 3):
        nbr_row = nbr_graphs
    elif (nbr_graphs == 3 or nbr_graphs == 4):
        nbr_row = 2
    else:
        nbr_row = 3
    nbr_col = math.ceil(nbr_graphs / nbr_row)

    return nbr_row, nbr_col


def plot_graph(fig, resolution, fig_title, filename):
    if (fig_title is not None):
        fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout()  # Avoiding overlapping texts (legend)
    fig.set_size_inches(resolution[0]/fig.dpi, resolution[1]/fig.dpi)
    if (filename != ""):
        fig.savefig(filename, bbox_inches='tight')
        util.print_terminal("Graph has been saved on filename '{}'"
                            .format(filename))
    else:
        plt.show()


def plot2d(x_datas: List[Array[float]], y_datas: List[Array[float]],
           x_labels: Optional[List[str]] = None,
           y_labels: Optional[List[str]] = None,
           x_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
           y_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
           plot_linestyles: List[str] = ['-'],
           plot_labels: Optional[List[Optional[str]]] = None,
           plot_titles: Optional[List[str]] = None,
           plot_colors: Optional[List[str]] = None,
           plot_groups: Optional[List[int]] = None,
           split: Optional[bool] = None, opacity: List[float] = [0.2],
           fig_title: Optional[str] = None, filename: str = "",
           resolution: Tuple[float, float] = (1920., 1080.),
           triangle_layout: bool = False) -> None:

    # N.B. if y_datas comes from field, np.ndarray is multidim
    # Initializing -----------------------------------------------------
    fig = plt.gcf()
    # Managing x and y labels ------------------------------------------
    x_labels = check_axis_labels(util.make_list(x_labels), axis_labels)
    y_labels = check_axis_labels(util.make_list(y_labels), axis_labels)
    # Padding ----------------------------------------------------------
    y_datas = util.make_list(y_datas)
    x_datas = util.make_list(x_datas, len(y_datas))
    if (len(y_datas) < len(x_datas)):
        util.warning_terminal("The number of y data must be equal or greater "
            "than the number of x data, graph creation aborted.")

        return None
    plot_labels = util.make_list(plot_labels, len(x_datas))
    plot_colors = util.make_list(plot_colors, len(x_datas))
    plot_linestyles = util.make_list(plot_linestyles, len(x_datas))
    opacity = util.make_list(opacity, len(x_datas))
    if (plot_groups is not None):
        plot_groups= util.make_list(plot_groups, len(x_datas))
    # Preparing graph parameters
    graphs, nbr_graphs = get_graph_layout(plot_groups, split, len(x_datas))
    # Padding ----------------------------------------------------------
    x_labels = util.make_list(x_labels, nbr_graphs)
    y_labels = util.make_list(y_labels, nbr_graphs)
    x_ranges = util.make_list(x_ranges, nbr_graphs, None)
    y_ranges = util.make_list(y_ranges, nbr_graphs, None)
    plot_titles = util.make_list(plot_titles, nbr_graphs, '')
    # Nonexistent field  management (no field recorded in component) ---
    for i in range(len(x_datas)):
        if ((y_datas[i] is None)):
            util.warning_terminal("Try to plot a nonexistent field!")
            x_datas[i] = np.zeros(0)
            y_datas[i] = np.zeros(0)
    # Plot graph -------------------------------------------------------
    # Plot graph -------------------------------------------------------
    nbr_row, nbr_col = get_graph_structure(nbr_graphs)
    triangle = 0 if (triangle_layout and nbr_graphs == 3) else 1
    offset = 0  # Offset for index of plot in layout
    for i, graph in enumerate(graphs) :
        index = i + 1 + offset
        if (triangle | i):
            nbr_col_subplot = nbr_col
        else:
            nbr_col_subplot = nbr_col - 1
            offset += 1
        add_2D_subplot.counter = 0 # For own colors if not specified
        plt_to_add = fig.add_subplot(nbr_row, nbr_col_subplot, index)
        for plot in graph:
            add_2D_subplot(plt_to_add, x_datas[plot], y_datas[plot],
                           x_labels[i], y_labels[i], x_ranges[i],
                           y_ranges[i], plot_titles[i], plot_labels[plot],
                           plot_linestyles[plot], plot_colors[plot],
                           opacity[plot])
    # Plotting / saving ------------------------------------------------
    plot_graph(fig, resolution, fig_title, filename)


def plot3d(x_datas: List[Array[float]], y_datas: List[Array[float]],
           z_datas: Optional[List[Array[float]]] = None,
           x_labels: Optional[List[str]] = None,
           y_labels: Optional[List[str]] = None,
           z_labels: Optional[List[str]] = None,
           x_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
           y_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
           z_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
           plot_titles: Optional[List[str]] = None,
           plot_colors: Optional[List[str]] = None,
           plot_groups: Optional[List[int]] = None,
           split: Optional[bool] = None, opacity: List[float] = [0.8],
           fig_title: Optional[str] = None, filename: str = "",
           resolution: Tuple[float, float] = (1920., 1080.),
           plot_types: List[str] = [cst.DEF_3D_PLOT],
           triangle_layout: bool = False) -> None:

    # N.B. if y_datas comes from field, np.ndarray is multidim
    # Initializing -----------------------------------------------------
    fig = plt.gcf()
    # Managing x and y and z labels ------------------------------------
    x_labels = check_axis_labels(util.make_list(x_labels), axis_labels)
    y_labels = check_axis_labels(util.make_list(y_labels), axis_labels)
    z_labels = check_axis_labels(util.make_list(z_labels), axis_labels)
    # Padding ----------------------------------------------------------
    if (z_datas is not None):
        z_datas = util.make_list(z_datas)
        y_datas = util.make_list(y_datas, len(z_datas))
    else:
        y_datas = util.make_list(y_datas)
    x_datas = util.make_list(x_datas, len(y_datas))

    if (z_datas is not None and len(z_datas) < len(y_datas)):
        util.warning_terminal("The number of z data must be equal or greater "
            "than the number of y data, graph creation aborted.")
        return None

    if (len(y_datas) < len(x_datas)):
        util.warning_terminal("The number of y data must be equal or greater "
            "than the number of x data, graph creation aborted.")
        return None
    plot_colors = util.make_list(plot_colors, len(x_datas))
    opacity = util.make_list(opacity, len(x_datas))
    plot_types = util.make_list(plot_types, len(x_datas))
    for i in range(len(plot_types)):
        if (plot3d_types.get(plot_types[i]) is None):
            util.warning_terminal("3D plot type '{}' does not exist, replace by "
                "'{}'.".format(plot_types[i], cst.DEF_3D_PLOT))
            plot_types[i] = cst.DEF_3D_PLOT
    if (plot_groups is not None):
        plot_groups= util.make_list(plot_groups, len(x_datas))
    # Preparing graph parameters
    graphs, nbr_graphs = get_graph_layout(plot_groups, split, len(x_datas))
    # Padding ----------------------------------------------------------
    x_labels = util.make_list(x_labels, nbr_graphs)
    y_labels = util.make_list(y_labels, nbr_graphs)
    z_labels = util.make_list(z_labels, nbr_graphs)
    x_ranges = util.make_list(x_ranges, nbr_graphs, None)
    y_ranges = util.make_list(y_ranges, nbr_graphs, None)
    z_ranges = util.make_list(z_ranges, nbr_graphs, None)
    plot_titles = util.make_list(plot_titles, nbr_graphs, '')
    # Nonexistent field  management (no field recorded in component) ---
    for i in range(len(x_datas)):
        if ((y_datas[i] is None) or (z_datas[i] is None)):
            util.warning_terminal("Try to plot a nonexistent field! (graph at "
                "position {} will be ignored)".format(i))
            x_datas[i] = np.zeros(0)
            y_datas[i] = np.zeros(0)
            z_datas[i] = np.zeros(0)
    # Plot graph -------------------------------------------------------
    nbr_row, nbr_col = get_graph_structure(nbr_graphs)
    triangle = 0 if (triangle_layout and nbr_graphs == 3) else 1
    offset = 0  # Offset for index of plot in layout
    for i, graph in enumerate(graphs) :
        index = i + 1 + offset
        if (triangle | i):
            nbr_col_subplot = nbr_col
        else:
            nbr_col_subplot = nbr_col - 1
            offset += 1
        add_3D_subplot.counter = 0 # For own colors if not specified
        plt_to_add = fig.add_subplot(nbr_row, nbr_col_subplot, index,
                                     projection='3d')
        for plot in graph:
            add_3D_subplot(plt_to_add, x_datas[plot], y_datas[plot],
                           z_datas[plot], x_labels[i], y_labels[i],
                           z_labels[i], x_ranges[i], y_ranges[i],
                           z_ranges[i], plot_titles[i], plot_colors[plot],
                           opacity[plot], plot_types[plot])
    # Plotting / saving ------------------------------------------------
    plot_graph(fig, resolution, fig_title, filename)
