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
from typing import Callable, List, Optional, overload, Union, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


# Ref: Govind Agrawal, Chapter 2: Pulse Propagation in Fibers,
# Nonlinear Fiber Optics (Fifth Edition), Academic Press, 2013, Page 34.
# coefficients = [(coeff, exponant)]
w_coefficients: List[Tuple[float, float]]
w_coefficients = [(0.616, 0.0), (1.66, -1.5), (0.987, -6.0)]


class EffectiveArea(AbstractParameter):

    def __init__(self, v_nbr: Union[float, Callable], core_radius: float
                 ) -> None:
        r"""Effective area, currently only for single mode.

        Parameters
        ----------
        v_nbr :
            The V number. If a callable is provided, the
            variable must be angular frequency. :math:`[ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`

        """
        self._core_radius: float = core_radius
        self._v_nbr: Union[float, Callable] = v_nbr
    # ==================================================================
    @property
    def core_radius(self) -> float:

        return self._core_radius
    # ------------------------------------------------------------------
    @core_radius.setter
    def core_radius(self, core_radius: float) -> None:

        self._core_radius = core_radius
    # ==================================================================
    @property
    def v_nbr(self) -> Union[float, Callable]:

        return self._v_nbr
    # ------------------------------------------------------------------
    @v_nbr.setter
    def v_nbr(self, v_nbr: Union[float, Callable]) -> None:

        self._v_nbr = v_nbr
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the effective area.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the effective area. :math:`[\mu m^2]`

        """
        fct = CallableContainer(EffectiveArea.calc_effective_area,
                                [self._v_nbr, self._core_radius])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_effective_area(v_nbr: float, core_radius: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_effective_area(v_nbr: np.ndarray, core_radius: float
                            ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_effective_area(v_nbr, core_radius):
        r"""Calculate the effective area. [4]_

        Parameters
        ----------
        v_nbr :
            The v_nbr number.
        core_radius :
            The radius of the core. :math:`[\mu m]`

        Returns
        -------
        :
            Value of the effective area. :math:`[\mu m^2]`

        Notes
        -----

        .. math:: A_{eff} = \pi w^2 = \pi \big(a(0.616
                  + 1.66V^{-\frac{2}{3}} + 0.987V^{-6})\big)^2

        References
        ----------
        .. [4] Govind Agrawal, Chapter 2: Pulse Propagation in Fibers,
           Nonlinear Fiber Optics (Fifth Edition), Academic Press, 2013,
           Page 34.

        """
        if (isinstance(v_nbr, float)):
            res = 0.0
            for coeff in w_coefficients:
                res += coeff[0] * v_nbr**(coeff[1])
            res = cst.PI * (core_radius*res)**2
        else:
            res = np.zeros(v_nbr.shape)
            for coeff in w_coefficients:
                res += coeff[0] * np.power(v_nbr, coeff[1])
            res = cst.PI * np.square(core_radius*res)

        return res


if __name__ == "__main__":
    """Plot the effective area as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    # With float
    omega: float = oc.lambda_to_omega(1552.0)
    core_radius: float = 5.0
    n_clad: float = 1.44
    sellmeier: oc.Sellmeier = oc.Sellmeier("sio2")
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    eff_area: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    print(eff_area(omega))
    v_nbr: float = v_nbr_inst(omega)
    eff_area = oc.EffectiveArea(v_nbr, core_radius)
    print(eff_area(omega))
    # With numpy ndarray
    lambdas: np.ndarray = np.linspace(900, 1550, 100)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    eff_area = oc.EffectiveArea(v_nbr_inst, core_radius)
    res: np.ndarray = eff_area(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = [r'Effective Area, $\,\mu m^2$']
    plot_titles: List[str] = ["Effective Area as a function of the wavelength "
                              "\n for Silica core with constant cladding "
                              "refractive index."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0],
              y_ranges=[(35., 110.)])
