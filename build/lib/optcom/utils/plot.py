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
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
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


linecolors = ['violet', 'orange', 'green', 'red', 'brown', 'pink', 'gray',
              'olive', 'blue', 'gold', 'black', 'marroon', 'navy']
'''

xy_labels = { "t" : "Time, $t \, (ps)$", \
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


def check_xy_labels(labels_to_check: Optional[List[str]],
                    labels: Dict[str, str]):
    """Check if can assimilate given label to recorded ones."""
    if (labels_to_check is not None):
        for i in range(len(labels_to_check)):
            if (labels_to_check[i] in labels.keys()):
                labels_to_check[i] = labels.get(labels_to_check[i])

    return labels_to_check


def add_subplot_para(plt_to_add, x_label, y_label, x_range, y_range,
                     plot_title):

    if (x_label != None):
        plt_to_add.set_xlabel(x_label)
    if (y_label != None):
        plt_to_add.set_ylabel(y_label)
    if (x_range != None):
        plt_to_add.set_xlim(x_range)
    if (y_range != None):
        plt_to_add.set_ylim(y_range)
    if (plot_title != None):
        plt_to_add.set_title(plot_title)


def add_single_plot(plt_to_add, x_data, y_data,x_label, y_label, x_range,
                    y_range, plot_title, plot_label, plot_linestyle,
                    plot_color, opacity):
    if (y_data.ndim == 1):  # not multidimentsional
        y_data = y_data.reshape((1,-1))
    if (cst.AUTO_PAD_PLOT):
        x_data = np.asarray(x_data)
        x_data, y_data = util.auto_pad(x_data, y_data)
    multi_channel = len(y_data) > 1
    labels_on_plot = plot_label is not None
    colors_on_plot = plot_color is not None
    if (multi_channel):
        plot_label = util.make_list(plot_label, len(y_data))
    for i in range(len(y_data)):
        if (multi_channel):
            if (labels_on_plot):
                plot_label_temp = plot_label[i] + " (ch.{})".format(i)
            else:
                plot_label_temp = "channel {}".format(i)
        else:
            plot_label_temp = plot_label
        if (not colors_on_plot):
            if (labels_on_plot or multi_channel):
                plt_to_add.plot(x_data, y_data[i], ls=plot_linestyle,
                                label=plot_label_temp)
            else:
                plt_to_add.plot(x_data, y_data[i], ls=plot_linestyle)
            plt_to_add.fill_between(x_data , y_data[i], alpha=opacity)
        else:
            if (labels_on_plot or multi_channel):
                plt_to_add.plot(x_data , y_data[i], ls=plot_linestyle,
                                c=plot_color, label=plot_label_temp)
            else:
                plt_to_add.plot(x_data , y_data[i], ls=plot_linestyle,
                                c=plot_color)
            plt_to_add.fill_between(x_data , y_data[i], alpha=opacity,
                                    facecolor=plot_color)
        add_subplot_para(plt_to_add, x_label, y_label, x_range, y_range,
                         plot_title)
        if (labels_on_plot or multi_channel):
            plt_to_add.legend(loc = "best")


def plot(x_datas: List[Array[float]], y_datas: List[Array[float]],
         x_labels: Optional[List[str]] = None,
         y_labels: Optional[List[str]] = None,
         x_ranges: Optional[List[float]] = None,
         y_ranges: Optional[List[float]] = None,
         plot_linestyles: List[str] = ['-'],
         plot_labels: List[Optional[Union[str,List[str]]]] = [None],
         plot_titles: Optional[List[str]] = None,
         plot_colors: Optional[List[str]] = None,
         plot_groups: Optional[List[int]] = None,
         split: Optional[bool] = None, opacity: float = 0.2,
         fig_title: Optional[str] = None, filename: str = ""):

    # N.B. if y_datas comes from field, np.ndarray is multidim
    # initializing -----------------------------------------------------
    plt.clf()
    # Managing x and y labels ------------------------------------------
    x_labels = check_xy_labels(x_labels, xy_labels)
    y_labels = check_xy_labels(y_labels, xy_labels)
    # Padding ----------------------------------------------------------
    y_datas = util.make_list(y_datas)
    x_datas = util.make_list(x_datas, len(y_datas))
    if (len(y_datas) < len(x_datas)):
        util.warning_terminal("The number of y data must be equal or greater "
            "than the number of x data, graph creation aborted.")
        return None
    #x_datas, y_datas = util.pad_list_with_last_elem(x_datas, y_datas, True)
    x_labels = util.make_list(x_labels, len(x_datas))
    y_labels = util.make_list(y_labels, len(y_datas))
    x_ranges = util.make_list(x_ranges, len(x_datas))
    y_ranges = util.make_list(y_ranges, len(y_datas))
    plot_labels = util.make_list(plot_labels, len(x_datas))
    plot_colors = util.make_list(plot_colors, len(x_datas))
    plot_linestyles = util.make_list(plot_linestyles, len(x_datas))
    plot_titles = util.make_list(plot_titles, len(x_datas), '')
    if (plot_groups is not None):
        plot_groups= util.make_list(plot_groups, len(x_datas))
    # Preparing graph parameters
    if (split is None):
        if (plot_groups is None):
            nbr_graphs = 1
            graphs = [[i for i in range(len(x_datas))]]
        else:
            nbr_graphs = max(plot_groups) + 1
            graphs = [[] for i in range(nbr_graphs)]
            for i in range(len(plot_groups)):
                    graphs[plot_groups[i]].append(i)
    else:
        if (split):
            nbr_graphs = len(x_datas)
            graphs = [[i] for i in range(len(x_datas))]
        else:
            nbr_graphs = 1
            graphs = [[i for i in range(len(x_datas))]]
    plot_titles, graphs = util.pad_list_with_last_elem(plot_titles, graphs)
    # Nonexistent field  management (no field recorded in component)
    for i in range(len(y_datas)):
        if (y_datas[i] is None):
                util.warning_terminal("Try to plot a nonexistent field!")
                y_datas[i] = np.zeros(0)
                x_datas[i] = np.zeros(0)
    # Plot graph -------------------------------------------------------
    if (nbr_graphs < 4):
        nbr_row = nbr_graphs
    elif (nbr_graphs == 4):
        nbr_row = 2
    else:
        nbr_row = 3
    nbr_col = math.ceil(nbr_graphs / nbr_row)
    for i, graph in enumerate(graphs) :
        plt_to_add = plt.subplot(nbr_row, nbr_col, i+1)
        for plot in graph:
            add_single_plot(plt_to_add, x_datas[plot], y_datas[plot],
                            x_labels[plot], y_labels[plot], x_ranges[plot],
                            y_ranges[plot], plot_titles[i],
                            plot_labels[plot], plot_linestyles[plot],
                            plot_colors[plot], opacity)
    # Finalizing -------------------------------------------------------
    if (fig_title is not None):
        plt.suptitle(fig_title, fontsize=16)
    plt.tight_layout()  # Avoiding overlapping texts (legend)
    if (filename != ""):
        plt.savefig(filename)
        util.print_terminal("Graph has been saved on filename '{}'"
                            .format(filename))
    else:
        plt.show()
