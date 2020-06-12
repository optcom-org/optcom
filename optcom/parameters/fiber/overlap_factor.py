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


class OverlapFactor(AbstractParameter):

    def __init__(self, eff_area: Union[float, Callable], doped_area: float
                 ) -> None:
        r"""
        Parameters
        ----------
        eff_area :
            The effective area. :math:`[\mu m^2]`  If a callable is
            provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
        doped_area :
            The doped area. :math:`[\mu m^2]`

        """
        self._doped_area: float = doped_area
        self._eff_area: Union[float, Callable] = eff_area
    # ==================================================================
    @property
    def doped_area(self) -> float:

        return self._doped_area
    # ------------------------------------------------------------------
    @doped_area.setter
    def doped_area(self, doped_area: float) -> None:

        self._doped_area = doped_area
    # ==================================================================
    @property
    def eff_area(self) -> Union[float, Callable]:

        return self._eff_area
    # ------------------------------------------------------------------
    @eff_area.setter
    def eff_area(self, eff_area: Union[float, Callable]) -> None:

        self._eff_area = eff_area
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Calculate the overlap factor.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the overlap factor.

        """
        fct = CallableContainer(OverlapFactor.calc_overlap_factor,
                                [self._eff_area, self._doped_area])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_overlap_factor(eff_area: float, doped_area: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_overlap_factor(eff_area: np.ndarray, doped_area: float,
                            ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_overlap_factor(eff_area, doped_area):
        r"""Calculate the overlap factor.

        Parameters
        ----------
        eff_area :
            The effective area. :math:`[\mu m^2]`
        doped_area :
            The doped area. :math:`[\mu m^2]`

        Returns
        -------
        :
            Value of the overlap factor.

        Notes
        -----
        Considering:

        .. math:: \Gamma  = 1 - e^{-2\frac{A_{doped}}{A_{eff}}}

        """
        if (isinstance(eff_area, float)):
            res = 1 - math.exp(-2*doped_area/eff_area)
        else:
            res = 1 - np.exp(-2*doped_area/eff_area)

        return res


if __name__ == "__main__":
    """Plot the overlap factor as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    # With float
    A_doped: float = oc.PI*25.0
    omega: float = oc.lambda_to_omega(1552.0)
    core_radius: float = 5.0
    sellmeier: oc.Sellmeier = oc.Sellmeier("sio2")
    n_clad: float = 1.44
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    A_eff_inst: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    of_inst: oc.OverlapFactor = oc.OverlapFactor(A_eff_inst, A_doped)
    print(of_inst(omega))

    A_eff: float = A_eff_inst(omega)
    of_inst = oc.OverlapFactor(A_eff, A_doped)
    print(of_inst(omega))
    print(oc.OverlapFactor.calc_overlap_factor(A_eff, A_doped))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900., 1550., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    of_inst = oc.OverlapFactor(A_eff_inst, A_doped)
    res: np.ndarray = of_inst(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['Overlap factor']
    plot_titles: List[str] = ["Overlap factor as a function of the wavelength "
                              "\n for Silica core with constant cladding "
                              "refractive index."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0],
              y_ranges=[(0.75, 1.)])
