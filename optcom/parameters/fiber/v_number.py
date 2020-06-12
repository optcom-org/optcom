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


class VNumber(AbstractParameter):

    def __init__(self, NA: Union[float, Callable], core_radius: float
                 ) -> None:
        r"""
        Parameters
        ----------
        NA :
            The numerical_aperture. If a callable is provided, the
            variable must be angular frequency. :math:`[ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`

        """
        self._core_radius: float = core_radius
        self._NA: Union[float, Callable] = NA
    # ==================================================================
    @property
    def NA(self) -> Union[float, Callable]:

        return self._NA
    # ------------------------------------------------------------------
    @NA.setter
    def NA(self, NA: Union[float, Callable]) -> None:

        self._NA = NA
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the V number.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the V parameter.
        """
        fct = CallableContainer(VNumber.calc_v_number,
                                [omega, self._NA, self._core_radius])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_v_number(omega: float, NA: float, core_radius: float
                      ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_v_number(omega: np.ndarray, NA: np.ndarray, core_radius: float
                      ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_v_number(omega, NA, core_radius):
        r"""Calculate the V parameter.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        NA :
            The numerical aperture.
        core_radius :
            The radius of the core. :math:`[\mu m]`

        Returns
        -------
        :
            Value of the V parameter.

        Notes
        -----
        Considering:

        .. math:: V = k_0 a \text{NA} = \frac{\omega_0}{c} a \text{NA}

        """
        # Unit conversion
        core_radius *= 1e3  # um -> nm

        return NA * core_radius * omega / cst.C


if __name__ == "__main__":
    """Plot the V fiber number as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import math
    from typing import List

    import numpy as np

    import optcom as oc

    # With float
    omega: float = oc.lambda_to_omega(1552.0)
    core_radius: float = 5.0
    n_clad: float = 1.44
    sellmeier: oc.Sellmeier = oc.Sellmeier("sio2")
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    print(v_nbr(omega))
    NA: float = NA_inst(omega)
    v_nbr = oc.VNumber(NA, core_radius)
    print(v_nbr(omega))
    print(oc.VNumber.calc_v_number(omega, NA, core_radius))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900, 1550, 10)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    NA_inst = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr = oc.VNumber(NA_inst, core_radius)
    res: np.ndarray = v_nbr(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['V Number']
    plot_titles: List[str] = ["V number as a function of the wavelength \n"
                              "for Silica core with constant cladding "
                              "refractive index."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0],
              y_ranges=[(2., 7.)])
