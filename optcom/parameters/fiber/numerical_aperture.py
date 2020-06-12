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


class NumericalAperture(AbstractParameter):

    def __init__(self, n_core: Union[float, Callable],
                 n_clad: Union[float, Callable]) -> None:
        r"""
        Parameters
        ----------
        n_core :
            The refractive index of the core.  If a callable is
            provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
        n_clad :
            The refractive index of the cladding.  If a callable is
            provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`

        """
        self._n_core: Union[float, Callable] = n_core
        self._n_clad: Union[float, Callable] = n_clad
    # ==================================================================
    @property
    def n_core(self) -> Union[float, Callable]:

        return self._n_core
    # ------------------------------------------------------------------
    @n_core.setter
    def n_core(self, n_core: Union[float, Callable]) -> None:

        self._n_core = n_core
    # ==================================================================
    @property
    def n_clad(self) -> Union[float, Callable]:

        return self._n_clad
    # ------------------------------------------------------------------
    @n_clad.setter
    def n_clad(self, n_clad: Union[float, Callable]) -> None:

        self._n_clad = n_clad
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
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`


        Returns
        -------
        :
            The numerical aperture.

        """
        fct = CallableContainer(NumericalAperture.calc_NA,
                                [self._n_core, self._n_clad])

        return fct(omega)
    # ==================================================================
    # Static methods ===================================================
    # ==================================================================
    @overload
    @staticmethod
    def calc_NA(n_core: float, n_clad: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_NA(n_core: float, n_clad: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_NA(n_core: np.ndarray, n_clad: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_NA(n_core: np.ndarray, n_clad: np.ndarray
                ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_NA(n_core, n_clad):
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        n_core :
            The refractive index of the core.
        n_clad :
            The refractive index of the cladding.

        Returns
        -------
        :
            The numerical aperture.

        Notes
        -----

        .. math::  NA = \sqrt{n_{co}^2 - n_{cl}^2}

        """
        if (isinstance(n_core, float) and isinstance(n_clad, float)):

            return math.sqrt(n_core**2 - n_clad**2)
        else:

            return np.sqrt(np.square(n_core) - np.square(n_clad))
    # ==================================================================
    @overload
    @staticmethod
    def calc_n_core(NA: float, n_clad: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_core(NA: float, n_clad: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_core(NA: np.ndarray, n_clad: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_core(NA: np.ndarray, n_clad: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_n_core(NA, n_clad):
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        NA :
            The numerical aperture.
        n_clad :
            The refractive index of the cladding.

        Returns
        -------
        :
            The refractive index of the core.

        Notes
        -----

        .. math::  n_{co} = \sqrt{NA^2 + n_{cl}^2}

        """
        if (isinstance(n_clad, float) and isinstance(NA, float)):

            return math.sqrt(NA**2 + n_clad**2)
        else:

            return np.sqrt(np.square(NA) + np.square(n_clad))
    # ==================================================================
    @overload
    @staticmethod
    def calc_n_clad(NA: float, n_core: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_clad(NA: float, n_core: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_clad(NA: np.ndarray, n_core: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_clad(NA: np.ndarray, n_core: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_n_clad(NA, n_core):
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        NA :
            The numerical aperture.
        n_core :
            The refractive index of the core.

        Returns
        -------
        :
            The refractive index of the cladding.

        Notes
        -----

        .. math::  n_{cl} = \sqrt{n_{co}^2 - NA^2}

        """
        if (isinstance(n_core, float) and isinstance(NA, float)):

            return math.sqrt(n_core**2 - NA**2)
        else:

            return np.sqrt(np.square(n_core) - np.square(NA))


if __name__ == "__main__":
    """Plot the numerical aperture as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    # With float
    n_clad: float = 1.44
    omega: float = oc.lambda_to_omega(1050.0)
    sellmeier: oc.Sellmeier = oc.Sellmeier("sio2")
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    print(NA_inst(omega))

    n_core: float = sellmeier(omega)
    NA_inst = oc.NumericalAperture(n_core, n_clad)
    NA: float = NA_inst(omega)
    print(NA, oc.NumericalAperture.calc_NA(n_core, n_clad))
    print(n_core, oc.NumericalAperture.calc_n_core(NA, n_clad))
    print(n_clad, oc.NumericalAperture.calc_n_clad(NA, n_core))

    # With numpy ndarray
    lambdas: np.ndarray = np.linspace(900, 1550, 100)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    NA_inst = oc.NumericalAperture(sellmeier, n_clad)
    res: np.ndarray = NA_inst(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['Numerical aperture']
    plot_titles: List[str] = ["Numercial aperture as a function of the "
                              "wavelength \n for Silica core with constant "
                              "cladding refractive index."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0],
              y_ranges=[(0.1, 0.2)])
