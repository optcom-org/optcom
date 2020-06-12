# Copyright 2019 The Optcom Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""".. moduleauthor:: Sacha Medaer"""

import math
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


class NLPhaseShift(AbstractParameter):
    r"""Compute the non-linear phase shift. [14]_

    References
    ----------
    .. [14] Wise, F.W., Chong, A. and Renninger, W.H., 2008. High‐energy
            femtosecond fiber lasers based on pulse propagation at
            normal dispersion. Laser & Photonics Reviews, 2(1‐2),
            pp.58-73.

    """

    def __init__(self, nl_index: Union[float, np.ndarray, Callable],
                 eff_area: Union[float, np.ndarray, Callable],
                 power: Union[float, np.ndarray, Callable], dz: float) -> None:
        r"""
        Parameters
        ----------
        nl_index :
            The non linear index. :math:`[m^2\cdot W^{-1}]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        eff_area:
            The effective area. :math:`[\mu m^2]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        power :
            The powers. :math:`[W]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        dz :
            The distance of propagation. :math:`[km]`

        """
        self._nl_index: Union[float, np.ndarray, Callable] = nl_index
        self._eff_area: Union[float, np.ndarray, Callable] = eff_area
        self._power: Union[float, np.ndarray, Callable] = power
        self._dz: float = dz
    # ==================================================================
    @property
    def nl_index(self) -> Union[float, np.ndarray, Callable]:

        return self._nl_index
    # ------------------------------------------------------------------
    @nl_index.setter
    def nl_index(self, nl_index: Union[float, np.ndarray, Callable]) -> None:

        self._nl_index = nl_index
    # ==================================================================
    @property
    def eff_area(self) -> Union[float, np.ndarray, Callable]:

        return self._eff_area
    # ------------------------------------------------------------------
    @eff_area.setter
    def eff_area(self, eff_area: Union[float, np.ndarray, Callable]) -> None:

        self._eff_area = eff_area
    # ==================================================================
    @property
    def power(self) -> Union[float, np.ndarray, Callable]:

        return self._power
    # ------------------------------------------------------------------
    @power.setter
    def power(self, power: Union[float, np.ndarray, Callable]) -> None:

        self._power = power
    # ==================================================================
    @property
    def dz(self) -> float:

        return self._dz
    # ------------------------------------------------------------------
    @dz.setter
    def dz(self, dz: float) -> None:

        self._dz = dz
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the non-linear phase shift.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the non-linear phase shift.

        """
        fct = CallableContainer(NLPhaseShift.calc_nl_phase_shift,
                                [omega, self._nl_index, self._eff_area,
                                 self._power, self._dz])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_nl_phase_shift(omega: float, nl_index: float, eff_area: float,
                            power: float, dz: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_nl_phase_shift(omega: np.ndarray, nl_index: np.ndarray,
                            eff_area: np.ndarray, power: np.ndarray,
                            dz: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_nl_phase_shift(omega, nl_index, eff_area, power, dz):
        r"""Calculate the non-linear phase shift.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        nl_index :
            The non linear index. :math:`[m^2\cdot W^{-1}]`
        eff_area:
            The effective area. :math:`[\mu m^2]`
        power :
            The powers. :math:`[W]`
        dz :
            The distance of propagation. :math:`[km]`

        Returns
        -------
        :
            Value of the non-linear phase shift.

        Notes
        -----

        .. math:: \Phi^{NL} = \frac{\omega}{c} n_2 \frac{P}{A_{eff}} dz

        """
        nl_index *=  1e-6  # m^2 W^{-1} -> km^2 W^{-1}
        eff_area *= 1e-18    # um^2 -> km^2
        c = cst.C * 1e-12    # ps/nm -> ps/km

        return (omega/c) * nl_index * (power/eff_area) * dz


if __name__ == "__main__":
    """Plot the non linear phase shift as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    medium: str = "SiO2"
    power: float = 0.01 # W
    dz: float = 1e-5    # km
    # With float
    omega: float = oc.lambda_to_omega(1552.0)
    core_radius: float = 5.0
    n_clad: float = 1.44
    sellmeier: oc.Sellmeier = oc.Sellmeier(medium)
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    eff_area_inst: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    nl_ind_inst: oc.NLIndex = oc.NLIndex(medium)
    nl_phase: oc.NLPhaseShift = oc.NLPhaseShift(nl_ind_inst, eff_area_inst,
                                                power, dz)
    print(nl_phase(omega))
    nl_ind: float = nl_ind_inst(omega)
    eff_area: float = eff_area_inst(omega)
    print(oc.NLPhaseShift.calc_nl_phase_shift(omega, nl_ind, eff_area, power,
                                              dz))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900., 1550., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    res: np.ndarray = nl_phase(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['Non-linear phase shift']
    plot_titles: List[str] = ["Non linear phase shift as a function of the "
                              "wavelength \n for Silica core with constant "
                              "cladding refractive index."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0])
