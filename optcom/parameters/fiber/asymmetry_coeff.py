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


class AsymmetryCoeff(AbstractParameter):

    def __init__(self, beta_1: Union[float, Callable],
                 beta_2: Union[float, Callable]) -> None:
        r"""
        Parameters
        ----------
        beta_1 :
            The zeroth order Taylor series dispersion coefficient of
            first waveguide. :math:`[km^{-1}]`   If callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        beta_2 :
            The zeroth order Taylor series dispersion coefficient of
            second waveguide. :math:`[km^{-1}]`  If callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`

        """
        self._beta_1: Union[float, Callable] = beta_1
        self._beta_2: Union[float, Callable] = beta_2
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the asymmetry coefficient.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the asymmetry coefficient. :math:`[km^{-1}]`

        """
        fct = CallableContainer(AsymmetryCoeff.calc_delta,
                                [self._beta_1, self._beta_2])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_delta(beta_1: float, beta_2: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_delta(beta_1: np.ndarray, beta_2: np.ndarray
                   ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_delta(beta_1, beta_2):
        r"""Calculate the measure of asymmetry for the parameters given
        for two waveguides. [2]_

        Parameters
        ----------
        beta_1 :
            The first term of the propagation constant of the first
            waveguide. :math:`[km^{-1}]`
        beta_2 :
            The first term of the propagation constant of the second
            waveguide. :math:`[km^{-1}]`

        Returns
        -------
        :
            Value of the asymmetry measure. :math:`[km^{-1}]`

        Notes
        -----

        .. math:: \delta_{a12} = \frac{1}{2} (\beta_{01} - \beta_{02})

        References
        ----------
        .. [2] Govind Agrawal, Chapter 2: Fibers Couplers,
           Applications of Nonlinear Fiber Optics (Second Edition),
           Academic Press, 2008, Page 57.

        """
        if (isinstance(beta_1, list)):
            beta_1 = np.asarray([beta_1[0]])
        if (isinstance(beta_2, list)):
            beta_2 = np.asarray([beta_2[0]])

        return 0.5 * (beta_1 - beta_2)


if __name__ == "__main__":
    """Plot the asymmetry coefficient as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc
    import optcom.utils.constants as cst

    # With float
    omega: float = oc.lambda_to_omega(1550.0)
    sellmeier: oc.Sellmeier = oc.Sellmeier("sio2", cst.FIBER_CORE_DOPANT,
                                           cst.CORE_DOPANT_CONCENT)
    n_core: float = sellmeier(omega)
    sellmeier = oc.Sellmeier("sio2", cst.FIBER_CLAD_DOPANT,
                             cst.CLAD_DOPANT_CONCENT)
    n_clad: float = sellmeier(omega)
    disp_1: oc.ChromaticDisp = oc.ChromaticDisp(ref_index=n_core)
    disp_2: oc.ChromaticDisp = oc.ChromaticDisp(ref_index=n_clad)
    asym: oc.AsymmetryCoeff = oc.AsymmetryCoeff(disp_1, disp_2)
    beta_1: float = disp_1(omega)[0]
    beta_2: float = disp_2(omega)[0]
    print('betas: ', beta_1, ' and ', beta_2)
    print(asym(omega))
    print(oc.AsymmetryCoeff.calc_delta(beta_1, beta_2))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900., 1550., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    res: np.ndarray = asym(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = [r'Asymmetry coefficient ($km^{-1}$)']
    plot_titles: List[str] = ["Asymmetry coefficient as a function of the "
                              "wavelength \n for Silica cores with dopants "
                              "{} ({}%) and {} ({}%)."
                              .format(cst.FIBER_CORE_DOPANT,
                                      cst.CORE_DOPANT_CONCENT,
                                      cst.FIBER_CLAD_DOPANT,
                                      cst.CLAD_DOPANT_CONCENT)]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0],
              y_ranges=[(20e6, 45e6)])
