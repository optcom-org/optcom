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
from typing import Callable, Dict, List, Optional, overload, Union, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.mccumber import McCumber
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


DOPANTS: List[str] = ["yb", "er"]
MEDIA: List[str] = ["sio2"]
# DOPANT_RANGE = {dopant:(bottom, up), \ldots} # in nm
DOPANT_RANGE: Dict[str, Tuple[float, float]]
DOPANT_RANGE = {"yb": (900., 1100.), "er": (1400., 1650.)}
# ref: Valley, G.C., 2001. Modeling cladding-pumped Er/Yb fiber
# amplifiers.  Optical Fiber Technology, 7(1), pp.21-44.
# DOPANT_FIT_COEFF = {dopant:[[factor, l_c, l_w, n],\ldots], \ldots}
DOPANT_FIT_COEFF: Dict[str, List[List[float]]]
DOPANT_FIT_COEFF = {"yb": [[0.09,913.0,8.0,2.0], [0.13,950.0,40.0,4.0],
                           [0.2,968.0,40.0,2.4], [1.08,975.8,3.0,1.5]],
                    "er": [[0.221,1493.0,16.5,2.2], [0.342,1534.0,4.5,1.4],
                           [0.158,1534.0,30.0,4.0], [0.037,1534.0,85.0,4.0],
                           [0.132,1541.0,8.0,0.8]]}


class AbsorptionSection(AbstractParameter):

    def __init__(self, dopant: str = cst.DEF_FIBER_DOPANT,
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 T: float = cst.TEMPERATURE,
                 sigma_e: Optional[Union[float, Callable]] = None) -> None:
        r"""
        Parameters
        ----------
        dopant :
            The doping medium.
        medium :
            The fiber medium.
        T :
            The absolute temperature. :math:`[K]`
        sigma_e :
            A callable which return the emission cross section.
            :math:`[nm^2]` If a callable, the parameters must be the
            angular frequencies. :math:`[ps^{-1}]` If none is provided,
            use fitting formulas.

        """
        self._dopant: str = util.check_attr_value(dopant.lower(), DOPANTS,
                                                  cst.DEF_FIBER_DOPANT)
        self._medium: str = util.check_attr_value(medium.lower(), MEDIA,
                                                  cst.DEF_FIBER_MEDIUM)
        self._T: float = T
        self._sigma_e: Optional[Union[float, Callable]] = sigma_e
        self._coefficients = DOPANT_FIT_COEFF[self._dopant]
        self._lambda_range = DOPANT_RANGE[self._dopant]
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the derivatives of the absorption cross sections.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The nth derivative of the propagation constant.
            :math:`[ps^{i}\cdot km^{-1}]`

        """
        if (self._sigma_e is None):

            return AbsorptionSection.calc_sigma(omega, self._coefficients,
                                                self._lambda_range)
        else:
            mc = McCumber(dopant=self._dopant, medium=self._medium)
            fct = CallableContainer(mc.cross_section_absorption,
                                    [omega, self._sigma_e, self._T])

            return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_sigma(omega: float, coefficients: List[List[float]],
                   lambda_range: Optional[Tuple[float]]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_sigma(omega: np.ndarray, coefficients: List[List[float]],
                   lambda_range: Optional[Tuple[float]]) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_sigma(omega, coefficients, lambda_range=None):
        r"""Calculate the absorption cross section. Calculate with the
        fitting formula from ref. [1]_ .

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        coefficients :
            The coefficients for the fitting formula.
            :math:`[\text{factor}, l_c, l_w, n]`
        lambda_range :
            The range of wavelength in which the fitting formula is
            valid. (lower_bound, upper_bound)

        Returns
        -------
        float or numpy.ndarray of float
            Value of the absorption cross section. :math:`[nm^2]`

        Notes
        -----
        Considering:

        .. math:: f(\lambda, \lambda_c, \lambda_w, n)
                  = \exp\Big[-\Big\lvert
                  \frac{\lambda-\lambda_c}{\lambda_w}\Big\rvert^n \Big]

        With :math:`\lambda`, :math:`\lambda_c` and :math:`\lambda_w`
        in :math:`[nm]`

        For Ytterbium:

        .. math:: \sigma^{Yb}(\lambda) = 0.09f(\lambda,913,8,2)
                  + 0.13 f(\lambda,950,40,4)
                  + 0.2 f(\lambda, 968, 40, 2.4)
                  + 1.08 f(\lambda, 975.8,3,1.5)

        For Erbium:

        .. math:: \begin{split}
                    \sigma^{Er}(\lambda) &= 0.221
                    f(\lambda, 1493, 16.5, 2.2)
                    + 0.342 f(\lambda, 1534, 4.5, 1.4)\\
                    &\quad + 0.158f(\lambda, 1534, 30,4)
                    + 0.037 f(\lambda, 1534, 85, 4)
                    + 0.132 f(\lambda, 1541, 8,0.8)
                  \end{split}

        References
        ----------
        .. [1] Valley, G.C., 2001. Modeling cladding-pumped Er/Yb fiber
               amplifiers. Optical Fiber Technology, 7(1), pp.21-44.

        """
        Lambda = Domain.omega_to_lambda(omega)
        res = 0.0 if isinstance(Lambda, float) else np.zeros_like(Lambda)
        if (isinstance(Lambda, float)):
            if (lambda_range is None):
                res = AbsorptionSection._formula(Lambda, coefficients)
            else:
                if (Lambda < lambda_range[1] and Lambda > lambda_range[0]):
                    res = AbsorptionSection._formula(Lambda, coefficients)
        else:
            if (lambda_range is not None):
                Lambda_down = Lambda[Lambda < lambda_range[0]]
                Lambda_up = Lambda[Lambda > lambda_range[1]]
                if (len(Lambda_up)):
                    Lambda_in = Lambda[len(Lambda_down):-len(Lambda_up)]
                else:
                    Lambda_in = Lambda[len(Lambda_down):]
                res = np.array([])
                if (Lambda_in.size):
                    res = AbsorptionSection._formula(Lambda_in, coefficients)
                res = np.hstack((np.zeros_like(Lambda_down),
                                 res, np.zeros_like(Lambda_up)))
            else:
                res = AbsorptionSection._formula(Lambda, coefficients)
        res *= 1e-6      # 10^-20 cm^2 -> nm^2

        return res
    # ==================================================================
    @overload
    @staticmethod
    def _formula(Lambda: float, coefficients: List[float]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def _formula(Lambda: np.ndarray, coefficients: List[float]
                ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def _formula(Lambda, coefficients):

        def f(l, lc, lw, n):
            if (isinstance(l, float)):

                return math.exp(-(abs((l-lc)/lw)**n))
            else:

                return np.exp(-(np.abs((l-lc)/lw)**n))

        if (isinstance(Lambda, float)):
            res = 0.0
        else:
            res = np.zeros(Lambda.shape)

        for coeff in coefficients:
            res += coeff[0] * f(Lambda, coeff[1], coeff[2], coeff[3])

        return res



if __name__ == "__main__":
    """Plot the absorption cross sections from formula and file. A file
    with absorption cross section data must have been provided with
    Optcom.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List

    import numpy as np

    import optcom as oc
    from optcom.parameters.fiber.absorption_section import DOPANT_RANGE
    from optcom.parameters.fiber.absorption_section import DOPANTS

    x_data: List[np.ndarray] = []
    y_data: List[np.ndarray] = []
    plot_titles: List[str] = []
    T: float = 293.1
    medium: str = 'sio2'
    nbr_samples: int = 1000
    omegas: np.ndarray
    absorp: oc.AbsorptionSection
    sigmas: np.ndarray
    lambdas: np.ndarray
    dopant_name: str
    # From formula
    for dopant in DOPANTS:
        lambdas = np.linspace(DOPANT_RANGE[dopant][0],
                              DOPANT_RANGE[dopant][1],
                              nbr_samples)
        omegas = oc.omega_to_lambda(lambdas)
        absorp = oc.AbsorptionSection(dopant=dopant, medium=medium, T=T)
        sigmas = absorp(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = dopant[0].upper()+dopant[1:]
        plot_titles.append("Cross sections {} from formula"
                           .format(dopant_name))
    # From file
    file_sigma_a: str = './data/fiber_amp/cross_section/absorption/yb.txt'
    csv_sigma_a: Callable = oc.CSVFit(file_sigma_a, conv_factor=[1e9, 1e18],
                                      conv_func=[oc.Domain.lambda_to_omega])
    lambdas = np.linspace(DOPANT_RANGE['yb'][0], DOPANT_RANGE['yb'][1],
                          nbr_samples)
    omegas = oc.omega_to_lambda(lambdas)
    sigmas = csv_sigma_a(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    plot_titles.append("Cross sections Yb from file.")
    # From file and McCumber relations
    file_sigma_e: str = './data/fiber_amp/cross_section/emission/yb.txt'
    csv_sigma_e: Callable = oc.CSVFit(file_sigma_e, conv_factor=[1e9, 1e18],
                                     conv_func=[oc.Domain.lambda_to_omega])
    absorp = oc.AbsorptionSection(dopant='yb', medium=medium, T=T,
                                 sigma_e=csv_sigma_e)
    sigmas = absorp(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    plot_titles.append('Cross sections from emission Yb file and McCumber.')

    oc.plot2d(x_data, y_data, x_labels=['Lambda'],
              y_labels=[r'Absorption cross section, $\,\sigma_a\,(nm^2)$'],
              split=True, line_colors=['red'], plot_titles=plot_titles,
              line_styles=['-.'], line_opacities=[0.0])
