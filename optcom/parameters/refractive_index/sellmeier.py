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

import copy
import math
from typing import Dict, List, Optional, overload, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.parameters.abstract_parameter import AbstractParameter

# ref for As/lambdas values: Brückner, V., 2011. To the use of Sellmeier
# formula. Senior Experten Service (SES) Bonn and HfT Leipzig, Germany,
# 42, pp.242-250.
MEDIA: List[str] = ["sio2"]
# COEFF_VALUES = {medium: (As, lambdas), ...}
COEFF_VALUES: Dict[str, Tuple[List[float], List[float]]]
COEFF_VALUES = {"sio2": ([0.6961663, 0.4079426, 0.8974794],
                         [0.068404, 0.1162414, 9.896161])}
DOPANTS: List[str] = ["geo2", "p2o5", "b2O3", "f"]
# DOPANT_VALUES = {dopant: slope, ...}
DOPANT_VALUES: Dict[str, float]
DOPANT_VALUES = {"geo2": 1.4145e-3, "p2o5": 1.652e-3, "b2O3": -3.760e-4,
                 "f": -4.665e-3}


class Sellmeier(AbstractParameter):
    r"""Sellmeier equations.

    Represent the Sellmeier equations which are empirical relationship
    between refractive index and wavelength for a specific medium. [12]_

    References
    ----------
    .. [12] Malitson, I.H., 1965. Interspecimen comparison of the
           refractive index of fused silica. Josa, 55(10), pp.1205-1209.

    """

    def __init__(self, medium: str = cst.DEF_FIBER_MEDIUM,
                 dopant: str = cst.FIBER_CORE_DOPANT,
                 dopant_concent: float = 0.0) -> None:
        r"""
        Parameters
        ----------
        medium :
            The medium in which the wave propagates.
        dopant :
            The dopant of the medium.
        dopant_concent :
            The concentration of the dopant. [mole%]

        """
        self._medium = util.check_attr_value(medium.lower(), MEDIA,
                                             cst.DEF_FIBER_MEDIUM)
        self._dopant = util.check_attr_value(dopant.lower(), DOPANTS,
                                             cst.FIBER_CORE_DOPANT)
        self._As: List[float] = COEFF_VALUES[self._medium][0]
        self._lambdas: List[float] = COEFF_VALUES[self._medium][1]
        self._dopant_slope: float = DOPANT_VALUES[self._dopant]
        self._dopant_concent: float = dopant_concent
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> float: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the refractive index from the Sellmeier equations.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the refractive index

        """

        return Sellmeier.calc_ref_index(omega, self._As, self._lambdas,
                                        self._dopant_slope,
                                        self._dopant_concent)
    # ==================================================================
    @overload
    @staticmethod
    def calc_ref_index(omega: float, As: List[float], lambdas: List[float],
                       dopant_slope: float, dopant_concent: float
                       ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_ref_index(omega: np.ndarray, As: List[float],
                       lambdas: List[float], dopant_slope: float,
                       dopant_concent: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_ref_index(omega, As, lambdas, dopant_slope=0., dopant_concent=0.):
        r"""Compute the Sellmeier equations.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        As :
            The A coefficients for the sellmeier equations.
        lambdas :
            The wavelength coefficients for the sellmeier equations.
        dopant_slope :
            The slope coefficient of the dopant linear fitting.
        dopant_concent :
            The concentration of the dopant. [mole%]

        Returns
        -------
        :
            The refractive index.

        Notes
        -----

        .. math:: n^2(\lambda) = 1 + \sum_i A_i
                                 \frac{\lambda^2}{\lambda^2 - \lambda_i}

        If dopant is specified, use linear fitting parameters from
        empirical data at 1300 nm.

        .. math:: n_{new}(\lambda, d) = n(\lambda) + a d

        where :math:`a` is the slope of the linear fitting and :math:`d`
        is the dopant concentration.

        """
        Lambda = Domain.omega_to_lambda(omega)  # nm
        # Lambda in micrometer in formula
        Lambda = Lambda * 1e-3  # um
        if (isinstance(Lambda, float)):
            res = 1.0
            for i in range(len(As)):
                res += (As[i]*Lambda**2) / (Lambda**2 - lambdas[i]**2)
            res = math.sqrt(res)
        else:   # numpy.ndarray
            res = np.ones(Lambda.shape)
            for i in range(len(As)):
                res += (As[i]*Lambda**2) / (Lambda**2 - lambdas[i]**2)
            res = np.sqrt(res)

        res = res + dopant_slope * dopant_concent

        return res


if __name__ == "__main__":
    """Plot the refractive index as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc
    import optcom.utils.constants as cst

    medium: str = "sio2"
    sellmeier: oc.Sellmeier = oc.Sellmeier(medium)
    # With float
    omega: float = oc.lambda_to_omega(1550.0)
    print(sellmeier(omega))
    sellmeier = oc.Sellmeier(medium, cst.FIBER_CORE_DOPANT,
                             cst.CORE_DOPANT_CONCENT)
    n_core: float = sellmeier(omega)
    sellmeier = oc.Sellmeier(medium, cst.FIBER_CLAD_DOPANT,
                             cst.CLAD_DOPANT_CONCENT)
    n_clad: float = sellmeier(omega)
    print(n_core, n_clad, oc.NumericalAperture.calc_NA(n_core, n_clad))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(120., 2120., 2000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    sellmeier = oc.Sellmeier(medium)
    res: List[np.ndarray] = [sellmeier(omegas)]
    line_labels: List[Optional[str]] = ["no dopants"]

    sellmeier = oc.Sellmeier(medium, cst.FIBER_CORE_DOPANT,
                             cst.CORE_DOPANT_CONCENT)
    res.append(sellmeier(omegas))
    line_labels.append("{} mole% of {}"
                       .format(cst.CORE_DOPANT_CONCENT,
                               cst.FIBER_CORE_DOPANT))
    sellmeier = oc.Sellmeier(medium, cst.FIBER_CLAD_DOPANT,
                             cst.CLAD_DOPANT_CONCENT)
    res.append(sellmeier(omegas))
    line_labels.append("{} mole% of {}"
                       .format(cst.CLAD_DOPANT_CONCENT,
                               cst.FIBER_CLAD_DOPANT))
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['Refractive index']
    plot_titles: List[str] = ["Refractive index of Silica from Sellmeier"
                              " equations."]

    oc.plot2d([lambdas], res, x_labels=x_labels, y_labels=y_labels,
              line_labels=line_labels, plot_titles=plot_titles,
              line_opacities=[0.0], split=False, y_ranges=[(1.4, 3.)])
