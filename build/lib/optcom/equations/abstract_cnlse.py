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

from typing import Callable, List, Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.asymmetry import Asymmetry
from optcom.effects.coupling import Coupling
from optcom.effects.kerr import Kerr
from optcom.equations.abstract_coupled_equation import AbstractCoupledEquation
from optcom.field import Field


class AbstractCNLSE(AbstractCoupledEquation):
    """Coupled non linear Schrodinger equations.

    Represent the different effects in the NLSE as well as the
    interaction of NLSEs propagating along each others. Note that
    automatic calculation of the coupling coefficients rely on a formula
    that is only correct for symmetric coupler.

    Attributes
    ----------
    nbr_eqs : int
        Number of NLSEs in the CNLSE.

    """

    def __init__(self, nbr_fibers: int,
                 beta: Optional[Union[List[List[float]], Callable]],
                 kappa: Optional[List[List[List[float]]]],
                 sigma_cross: List[List[float]],
                 c2c_spacing: List[List[float]],
                 core_radius: List[float], V: List[float], n_0: List[float],
                 ASYM: bool, COUP: bool, XPM: bool, medium: str) -> None:
        r"""
        Parameters
        ----------
        nbr_fibers :
            The number of fibers in the coupler.
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]`
        kappa :
            The coupling coefficients. :math:`[km^{-1}]`
        sigma_cross :
            Positive term multiplying the XPM term of the NLSE inbetween
            the fibers.
        c2c_spacing :
            The center to center distance between two cores.
            :math:`[\mu m]`
        core_radius :
            The core radius. :math:`[\mu m]`
        V :
            The fiber parameter.
        n_0 :
            The refractive index outside of the waveguides.
        ASYM :
            If True, trigger the asymmetry effects between cores.
        COUP :
            If True, trigger the coupling effects between cores.
        XPM :
            If True, trigger the cross-phase modulation.
        medium :
            The main medium of the fiber.

        """

        super().__init__(nbr_fibers)
        if (beta is not None):
            beta_: Union[List[List[float]], List[List[Callable]]] =\
                util.make_matrix(beta, nbr_fibers, nbr_fibers)
        sigma_cross = util.make_matrix(sigma_cross, nbr_fibers, nbr_fibers,
                                       sym=True)
        if (kappa is not None):
            kappa = util.make_tensor(kappa, nbr_fibers, nbr_fibers, 0)
        V = util.make_list(V, nbr_fibers)
        n_0 = util.make_list(n_0, nbr_fibers)
        c2c_spacing = util.make_matrix(c2c_spacing, nbr_fibers, nbr_fibers,
                                       sym=True)
        core_radius = util.make_list(core_radius, nbr_fibers)
        for i in range(nbr_fibers):
            for j in range(nbr_fibers):
                if (i != j):
                    if (ASYM):
                        if (beta is not None):
                            self._effects_lin[i][j].append(
                                Asymmetry(beta_01=beta_[i][0],
                                          beta_02=beta_[j][0]))
                        else:
                            self._effects_lin[i][j].append(
                                Asymmetry(medium=medium))
                    if (COUP):
                        if (kappa is not None):
                            self._effects_all[i][j].append(
                                Coupling(kappa[i][j]))
                        else:
                            same_V = sum(V)/len(V) == V[0]
                            same_a = (sum(core_radius) / len(core_radius)
                                      == core_radius[0])
                            same_n_0 = sum(n_0)/len(n_0) == n_0[0]
                            if (not (same_V and same_a and same_n_0)):
                                util.warning_terminal("Automatic calculation "
                                    "of coupling coefficient assumes same "
                                    "V, core_radius and n_0 for now, "
                                    "different ones provided, might lead to "
                                    "unrealistic results.")
                            self._effects_all[i][j].append(
                                Coupling(V=V[i], a=core_radius[i],
                                         d=c2c_spacing[i][j], n_0=n_0[i]))
                    if (XPM):
                        self._effects_non_lin[i][j].append(Kerr(SPM=False,
                            XPM=True, FWM=False, sigma=sigma_cross[i][j]))
    # ==================================================================
    def op_non_lin(self, waves: Array[cst.NPFT], id: int,
                   corr_wave: Optional[Array[cst.NPFT]] = None
                   ) -> Array[cst.NPFT]:
        """Non linear operator of the equation."""
        eq_id = self._eq_id(id)
        rel_wave_id = self._rel_wave_id(id)
        gamma = self._eqs[eq_id][0].gamma[rel_wave_id]

        return (gamma * self._call_main("op", "non_lin", waves, id, corr_wave)
                + self._call_sub("op", "non_lin", waves, id, corr_wave))
