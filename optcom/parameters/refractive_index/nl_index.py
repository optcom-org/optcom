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
from typing import Dict, List, Optional, overload, Union, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.parameters.abstract_parameter import AbstractParameter


# Ref: Salceda-Delgado, G., Martinez-Rios, A., Ilan, B. and
# Monzon-Hernandez, D., 2012.  Raman response function an Raman fraction
# of phosphosilicate fibers.  Optical and Quantum Electronics, 44(14),
# pp.657-671.
# poly_coeff = {medium: [(coeff, exponant)], ...}
MEDIA: List[str] = ["sio2"]
POLY_FACTOR: Dict[str, float] = {"sio2": 1.000055}
POLY_COEFF: Dict[str, List[Tuple[float,float]]]
POLY_COEFF = {"sio2": [(8.30608, 0.0), (-27.79971, 1.0), (59.66014, 2.0),
              (-69.24258, 3.0), (45.22437, 4.0), (-15.63666, 5.0),
              (2.22585, 6.0)]}


class NLIndex(AbstractParameter):

    def __init__(self, medium: str = cst.DEF_FIBER_MEDIUM,
                 factor: Optional[float] = None,
                 coefficients: Optional[List[Tuple[float, float]]] = None
                 ) -> None:
        r"""The non linear index.

        Parameters
        ----------
        medium :
            The medium in which the wave propagates.
        factor :
            Number which will multiply the fitting formula.
        coefficients :
            Coefficients of the fitting formula. [(coeff, expo), ..]

        """
        self._medium: str = util.check_attr_value(medium.lower(), MEDIA,
                                                 cst.DEF_FIBER_MEDIUM)
        self._factor: float
        self._coefficients: List[Tuple[float, float]]
        if (factor is not None and coefficients is not None):
            self._factor = factor
            self._coefficients = coefficients
        else:
            self._factor = POLY_FACTOR[self._medium]
            self._coefficients = POLY_COEFF[self._medium]
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the non linear index.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the non linear index. :math:`[m^2\cdot W^{-1}]`

        """

        return NLIndex.calc_nl_index(omega, self._factor, self._coefficients)
    # ==================================================================
    @overload
    @staticmethod
    def calc_nl_index(omega: float, factor: float,
                      coefficients: List[Tuple[float, float]]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_nl_index(omega: np.ndarray, factor: float,
                      coefficients: List[Tuple[float, float]]
                      ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_nl_index(omega, factor, coefficients):
        r"""Calculate the non linear index by help of fitting formula.
        The non linear is assumed to be the sum of the electronic and
        Raman contributions at zero frequency shift. [10]_

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        factor :
            Number which will multiply the fitting formula.
        coefficients :
            Coefficients of the fitting formula. [(coeff, expo), ..]

        Returns
        -------
        :
            Value of the non linear index. :math:`[m^2\cdot W^{-1}]`

        Notes
        -----
        Considering:

        .. math:: n_2(\lambda) = 1.000055 (8.30608 - 27.79971\lambda
                  + 59.66014\lambda^2 - 69.24258\lambda^3
                  + 45.22437\lambda^4 - 15.63666\lambda^5
                  + 2.22585\lambda^6)

        with :math:`\lambda` in :math:`nm` and return with factor
        :math:`10^{-20}`

        References
        ----------
        .. [10] Salceda-Delgado, G., Martinez-Rios, A., Ilan, B. and
               Monzon-Hernandez, D., 2012. Raman response function an
               Raman fraction of phosphosilicate fibers. Optical and
               Quantum Electronics, 44(14), pp.657-671.

        """
        if (isinstance(omega, float)):
            res = 0.0
        else:
            res = np.zeros_like(omega)
        Lambda = Domain.omega_to_lambda(omega)
        Lambda *= 1e-3  # nm -> um
        if (isinstance(omega, float)):
            for elem in coefficients:
                res += elem[0] * Lambda**elem[1]
        else:
            for elem in coefficients:
                res += elem[0] * np.power(Lambda, elem[1])
        res *= factor

        return res * 1e-20  # from fitting formula to m^2 w^-1


if __name__ == "__main__":
    """Plot the non-linear index as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    # With float
    omega: float = oc.lambda_to_omega(1552.0)
    nl_index: oc.NLIndex = oc.NLIndex(medium="SiO2")
    print(nl_index(omega))
    # With numpy ndarray
    lambdas: np.ndarray = np.linspace(500, 1600, 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    res: np.ndarray = nl_index(omegas)

    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['n_2']
    plot_titles: List[str] = ["Non linear index as a function of the "
                              "wavelength for Silica core."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
               plot_titles=plot_titles, line_opacities=[0.0],
               y_ranges=[(2.5*1e-20, 3.2*1e-20)])
