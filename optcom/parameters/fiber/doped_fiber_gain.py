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


class DopedFiberGain(AbstractParameter):

    def __init__(self, sigma: Union[float, np.ndarray, Callable],
                 overlap: Union[float, np.ndarray, Callable],
                 N: Union[float, np.ndarray, Callable]) -> None:
        r"""
        Parameters
        ----------
        sigma :
            The cross sections. :math:`[nm^2]`  If a callable
            is provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
        overlap :
            The overlap factors. If a callable is provided, the variable
            must be angular frequency. :math:`[ps^{-1}]`
        N :
            The state density population. :math:`[nm^{-3}]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`

        """
        self._sigma: Union[float, np.ndarray, Callable] = sigma
        self._overlap: Union[float, np.ndarray, Callable] = overlap
        self._N: Union[float, np.ndarray, Callable] = N
    # ==================================================================
    @property
    def N(self) -> Union[float, np.ndarray, Callable]:

        return self._N
    # ------------------------------------------------------------------
    @N.setter
    def N(self, N: Union[float, np.ndarray, Callable]) -> None:
        self._N = N
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: float, order: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray, order: int) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, *args):
        r"""Compute the gain coefficient.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        order :
            The derivative order of the Taylor series expansion.

        Returns
        -------
        :
            The value of the gain coefficient. :math:`[km^{-1}]`

        """
        omega = args[0]
        fct = CallableContainer(DopedFiberGain.calc_doped_fiber_gain,
                                [self._sigma, self._overlap, self._N])

        if (len(args) == 1):    # No order
            res = np.zeros_like(omega)
            res = fct(omega)
        else:
            order = args[1]
            res = np.zeros((order+1, len(omega)))
            for i in range(order+1):
                res[i] = util.deriv(fct, omega, i)

        return res
    # ==================================================================
    @overload
    @staticmethod
    def calc_doped_fiber_gain(sigma: float, overlap: float, N: float
                              ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_doped_fiber_gain(sigma: np.ndarray, overlap: np.ndarray,
                              N: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_doped_fiber_gain(sigma, overlap, N):
        r"""Calculate the gain of the doped fiber.

        Parameters
        ----------
        sigma :
            The cross sections. :math:`[nm^2]`
        overlap :
            The overlap factors.
        N :
            The state density population. :math:`[nm^{-3}]`

        Returns
        -------
        :
            Value of the gain. :math:`[km^{-1}]`

        Notes
        -----

        .. math:: g(\omega) = \Gamma(\omega) \sigma(\omega) N(\omega)

        """
        g = overlap * sigma * N
        g *= 1e12   # nm^{-1} -> km^{-1}

        return g


if __name__ == "__main__":
    """Plot the fiber gain as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    medium: str = "sio2"
    dopant: str = "yb"
    A_doped: float = oc.PI*25.0
    # With float
    omega: float = oc.lambda_to_omega(1000)
    core_radius: float = 5.0
    sellmeier: oc.Sellmeier = oc.Sellmeier(medium)
    n_clad: float = 1.44
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    A_eff_inst: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    sigma_a_inst: oc.AbsorptionSection = oc.AbsorptionSection(dopant, medium)
    of_inst: oc.OverlapFactor = oc.OverlapFactor(A_eff_inst, A_doped)
    N: float = 0.01
    gain: oc.DopedFiberGain = oc.DopedFiberGain(sigma_a_inst, of_inst, N)
    print(gain(omega))
    sigma_a: float = sigma_a_inst(omega)
    of: float = of_inst(omega)
    print(oc.DopedFiberGain.calc_doped_fiber_gain(sigma_a, of, N))
    # With np.ndarray
    N_0 = 0.01
    N_1 = 0.011
    absorp_inst: oc.DopedFiberGain = oc.DopedFiberGain(sigma_a_inst, of_inst,
                                                       N_0)
    T: float = 293.15
    sigma_e_inst = oc.EmissionSection(dopant, medium, T, sigma_a_inst)
    emission_inst: oc.DopedFiberGain = oc.DopedFiberGain(sigma_e_inst, of_inst,
                                                         N_1)
    lambdas: np.ndarray = np.linspace(900., 1050., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    absorps: np.ndarray = absorp_inst(omegas)
    emissions: np.ndarray = emission_inst(omegas)
    gains: np.ndarray = emissions - absorps
    plot_titles: List[str] = [r"Fiber gain as a function of wavelength with "
                              r"ratio $\frac{N_1}{N_0} = \frac{11}{10}$."]

    oc.plot2d([lambdas], [gains], x_labels=['Lambda'],
              y_labels=[r'Gain, $\, g\,(km^{-1})$'], line_colors=['red'],
              plot_titles=plot_titles, line_opacities=[0.])
