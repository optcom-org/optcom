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


default_name = 'CW'


class CW(AbstractStartComp):
    r"""A CW pulse Generator.

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
    peak_power : list of float
        Peak power of the pulses. :math:`[W]`
    offset_nu : list of float
        The offset frequency. :math:`[THz]`
    init_phi : list of float
        The nitial phase of the pulses.

    Notes
    -----

    .. math:: A(0,t) = \sqrt{P_0} \exp\Big[i\big(\phi_0 - 2\pi (\nu_c
                                            +\nu_{offset})t\big)\Big]

    Component diagram::

        __________________ [0]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, channels: int = 1,
                 center_lambda: List[float] = [cst.DEF_LAMBDA],
                 peak_power: List[float] = [1e-3],
                 offset_nu: List[float] = [0.0], init_phi: List[float] = [0.0],
                 save: bool = False) -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        channels :
            The number of channels in the field.
        center_lambda :
            The center wavelength of the channels. :math:`[nm]`
        peak_power :
            Peak power of the pulses. :math:`[W]`
        offset_nu :
            The offset frequency. :math:`[THz]`
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
        util.check_attr_type(peak_power, 'peak_power', float, list)
        util.check_attr_type(offset_nu, 'offset_nu', float, list)
        util.check_attr_type(init_phi, 'init_phi', float, list)
        # Attr ---------------------------------------------------------
        self.channels: int = channels
        self.center_lambda: List[float] = util.make_list(center_lambda,
                                                         channels)
        self.peak_power: List[float] = util.make_list(peak_power, channels)
        self.offset_nu: List[float] = util.make_list(offset_nu, channels)
        self.init_phi: List[float] = util.make_list(init_phi, channels)
    # ==================================================================
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        field = Field(domain, cst.OPTI)
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
            phi = (self.init_phi[i]
                   - Domain.nu_to_omega(self.offset_nu[i])*domain.time)
            res += math.sqrt(self.peak_power[i]) * np.exp(1j*phi)
            field.append(res, Domain.lambda_to_omega(self.center_lambda[i]))

        output_fields.append(field)
        output_ports.append(0)

        return output_ports, output_fields


if __name__ == "__main__":

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power, spectral_power,\
                                            phase

    lt = layout.Layout(Domain(samples_per_bit=4096))

    channels = 3
    center_lambda = [1552.0, 1549.0, 1596.0]
    peak_power = [1e-3, 2e-3, 6e-3]
    offset_nu = [0.0, 1.56, -1.6]
    init_phi = [1.0, 1.0, 0.0]

    cw = CW(channels=channels, center_lambda=center_lambda,
            peak_power=peak_power, offset_nu=offset_nu, init_phi=init_phi,
            save=True)

    lt.run(cw)

    plot_titles = ["CW pulse temporal power", "CW pulse spectral power",
                   "CW pulse phase"]

    plot.plot([cw.fields[0].time, cw.fields[0].nu, cw.fields[0].time],
              [temporal_power(cw.fields[0].channels),
               spectral_power(cw.fields[0].channels),
               phase(cw.fields[0].channels)], ["t","nu","t"],
               ["P_t", "P_nu", "phi"], plot_titles=plot_titles, split=True)
