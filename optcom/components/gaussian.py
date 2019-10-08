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
import copy

import numpy as np
from typing import List, Optional, Sequence, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_start_comp import AbstractStartComp
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Gaussian'


class Gaussian(AbstractStartComp):
    r"""A gaussian pulse Generator.

    Attributes
    ----------
    name : str
        The name of the component.
    ports_type : list of int
        Type of each port of the component, give also the number of
        ports in the component. For types, see
        :mod:`optcom/utils/constant_values/port_types`.
    save : bool
        If True, the last wave to enter/exit a port will be saved.
    channels : int
        The number of channels in the field.
    center_lambda : list of float
        The center wavelength of the channels. :math:`[nm]`
    position : list of float
        Relative position of the pulses in the time window.
        :math:`\in [0,1]`
    width : list of float
        Half width of the pulse. :math:`[ps]`
    peak_power : list of float
        Peak power of the pulses. :math:`[W]`
    bit_rate : list of float
        Bit rate (repetition rate) of the pulse in the time window.
        :math:`[THz]`
    offset_nu : list of float
        The offset frequency. :math:`[THz]`
    order : list of int
        The order of the super gaussian pulse.
    chirp : list of float
        The chirp parameter for chirped pulses.
    init_phi : list of float
        The nitial phase of the pulses.

    Notes
    -----

    .. math:: A(0,t) = \sqrt{P_0}
              \exp\bigg[-\frac{1+iC}{2}
              \bigg(\frac{t-t_0}{T_0}\bigg)^{2m}
              + i\big(\phi_0 - 2\pi (\nu_c  + \nu_{offset})t\big)\bigg]

    Component diagram::

        __________________ [0]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, channels: int = 1,
                 center_lambda: List[float] = [cst.DEF_LAMBDA],
                 position: List[float] = [0.5], width: List[float] = [10.0],
                 peak_power: List[float] = [1e-3],
                 bit_rate: List[float] = [0.0], offset_nu: List[float] = [0.0],
                 order: List[int] = [1], chirp: List[float] = [0.0],
                 init_phi: List[float] = [0.0], save: bool = False) -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        channels :
            The number of channels in the field.
        center_lambda :
            The center wavelength of the channels. :math:`[nm]`
        position :
            Relative position of the pulses in the time window.
            :math:`\in [0,1]`
        width :
            Half width of the pulse. :math:`[ps]`
        peak_power :
            Peak power of the pulses. :math:`[W]`
        bit_rate :
            Bit rate (repetition rate) of the pulse in the time window.
            :math:`[THz]`
        offset_nu :
            The offset frequency. :math:`[THz]`
        order :
            The order of the super gaussian pulse.
        chirp :
            The chirp parameter for chirped pulses.
        init_phi :
            The initial phase of the pulses.
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.OPTI_OUT]
        super().__init__(name, default_name, ports_type, save)
        # Attr types check ---------------------------------------------
        util.check_attr_type(channels, 'channels', int)
        util.check_attr_type(center_lambda, 'center_lambda', float, list)
        util.check_attr_type(position, 'position', float, list)
        util.check_attr_type(width, 'width', float, list)
        util.check_attr_type(peak_power, 'peak_power', float, list)
        util.check_attr_type(bit_rate, 'bit_rate', float, list)
        util.check_attr_type(offset_nu, 'offset_nu', float, list)
        util.check_attr_type(order, 'order', int, list)
        util.check_attr_type(chirp, 'chirp', float, list)
        util.check_attr_type(init_phi, 'init_phi', float, list)
        # Attr ---------------------------------------------------------
        self.channels: int = channels
        self.center_lambda: List[float] = util.make_list(center_lambda,
                                                         channels)
        self.position: List[float] = util.make_list(position, channels)
        self.width: List[float] = util.make_list(width, channels)
        self.peak_power: List[float] = util.make_list(peak_power, channels)
        self.bit_rate: List[float] = util.make_list(bit_rate, channels)
        self.offset_nu: List[float] = util.make_list(offset_nu, channels)
        self.order: List[int] = util.make_list(order, channels)
        self.chirp: List[float] = util.make_list(chirp, channels)
        self.init_phi: List[float] = util.make_list(init_phi, channels)
    # ==================================================================
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        field = Field(domain, cst.OPTI)
        # Bit rate initialization --------------------------------------
        nbr_pulses = []
        for i in range(self.channels):
            if (self.bit_rate[i]):
                nbr_temp = math.floor(domain.time_window * self.bit_rate[i])
                if (nbr_temp):
                    nbr_pulses.append(nbr_temp)
                else:
                    util.warning_terminal("In component {}: the time window "
                        "is too thin for the bit rate specified, bit rate "
                        "will be ignored".format(self.name))
                    nbr_pulses.append(1)
            else:
                nbr_pulses.append(1)

        rel_pos = []
        for i in range(self.channels):
            pos_step = 1/nbr_pulses[i]
            if (nbr_pulses[i]%2):  # Odd
                dist_from_center = nbr_pulses[i]//2 * pos_step
            else:
                dist_from_center = (nbr_pulses[i]//2 - 1)*pos_step + pos_step/2
            rel_pos.append(np.linspace(self.position[i] - dist_from_center,
                                       self.position[i] + dist_from_center,
                                       num=nbr_pulses[i]))
        # Check offset -------------------------------------------------
        for i in range(len(self.offset_nu)):
            if (abs(self.offset_nu[i]) > domain.nu_window):
                self.offset_nu[i] = 0.0
                util.warning_terminal("The offset of channel {} in component "
                    "{} is bigger than half the frequency window, offset will "
                    "be ignored.".format(str(i), self.name))
        # Field initialization -----------------------------------------
        for i in range(self.channels):   # Nbr of channels
            res = np.zeros(domain.time.shape, dtype=cst.NPFT)
            for j in range(len(rel_pos[i])):
                norm_time = domain.get_shift_time(rel_pos[i][j])/self.width[i]
                var_time = np.power(norm_time, 2 * self.order[i])
                phi = (self.init_phi[i]
                       - Domain.nu_to_omega(self.offset_nu[i])*domain.time
                       - 0.5*self.chirp[i]*var_time)
                res += (math.sqrt(self.peak_power[i])
                        * np.exp((-0.5*var_time) + 1j*phi))
            field.append(res, Domain.lambda_to_omega(self.center_lambda[i]))

        output_fields.append(field)
        output_ports.append(0)

        return output_ports, output_fields


if __name__ == "__main__":

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power, spectral_power

    import matplotlib.pyplot as plt

    lt = layout.Layout(Domain(samples_per_bit=4096))

    channels = 3
    center_lambda = [1552.0, 1549.0, 1596.0]
    position = [0.5, 0.3, 0.5]
    width = [2.0, 5.3, 6]
    peak_power = [1e-3, 2e-3, 6e-3]
    bit_rate =  [0.0, 0.03, 0.04]
    offset_nu = [0.0, 1.56, -1.6]
    order = [1, 2, 1]
    chirp = [0.0, 0.5, 0.1]
    init_phi = [0.0, 1.0, 0.0]

    gssn = Gaussian(channels=channels, center_lambda=center_lambda,
                    position=position, width=width, peak_power=peak_power,
                    bit_rate=bit_rate, offset_nu=offset_nu, order=order,
                    chirp=chirp, init_phi=init_phi, save=True)

    lt.run(gssn)

    plot.plot([gssn.fields[0].time, gssn.fields[0].nu],
              [temporal_power(gssn.fields[0].channels),
              spectral_power(gssn.fields[0].channels)], ["t","nu"],
              ["P_t", "P_nu"], plot_titles=["Gaussian pulse"],  split=True)
