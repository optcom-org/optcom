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
from optcom.parameters.refractive_index.sellmeier import Sellmeier


class ChromaticDisp(AbstractParameter):

    def __init__(self, ref_index: Union[float, Callable]) -> None:
        r"""
        Parameters
        ----------
        ref_index :
            The refractive index. If a callable is provided, variable
            must be angular frequency. :math:`[ps^{-1}]`

        """
        self._ref_index: Union[float, Callable] = ref_index
    # ==================================================================
    @property
    def ref_index(self) -> Union[float, Callable]:

        return self._ref_index
    # ------------------------------------------------------------------
    @ref_index.setter
    def ref_index(self, ref_index: Union[float, Callable]) -> None:

        self._ref_index = ref_index
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> np.ndarray: ...
    @overload
    def __call__(self, omega: float, order: int) -> np.ndarray: ...
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    @overload
    def __call__(self, omega: np.ndarray, order: int) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega, order=0):
        r"""Compute the derivatives of the chromatic dispersion
        propagation constant.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        order :
            The order of the highest derivative.

        Returns
        -------
        :
            The nth derivatives of the propagation constant.
            :math:`[ps^{i}\cdot km^{-1}]`

        """

        return ChromaticDisp.calc_beta(omega, order, self._ref_index)
    # ==================================================================
    @overload
    @staticmethod
    def calc_beta(omega: float, order: int,
                  ref_index: Union[float, Callable]) -> List[float]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_beta(omega: np.ndarray, order: int,
                  ref_index: Union[float, Callable]) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_beta(omega, order, ref_index):
        r"""Calcul the nth first derivatives of the propagation
        constant. (valid only for TEM)

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        order :
            The order of the highest dispersion coefficient.
        ref_index :
            The refractive index.  If a callable is provided, the
            variable must be angular frequency. :math:`[ps^{-1}]`

        Returns
        -------
        :
            The nth derivative of the propagation constant.
            :math:`[ps^{i}\cdot km^{-1}]`

        Notes
        -----

        .. math:: \beta_i(\omega) = \frac{d^i \beta(\omega)}{d\omega^i}
                  = \frac{1}{c}\bigg(i
                  \frac{d^{(i-1)} n(\omega)}{d\omega^{(i-1)}}
                  + \omega \frac{d^i n(\omega)}{d\omega^i}\bigg)

        for :math:`i = 0, \ldots, \text{order}`

        """
        C = cst.C * 1e-12   # nm/ps -> km/ps
        if (isinstance(omega, float)):
            res = [0.0 for i in range(order+1)]
        else:
            res = np.zeros((order+1, len(omega)))
        if (callable(ref_index)):
            predict_ref_index = ref_index
        else:
            predict_ref_index = lambda omega: ref_index
        prec_n_deriv = 0.
        for i in range(order+1):
            current_n_deriv = util.deriv(predict_ref_index, omega, i)
            res[i] = (i*prec_n_deriv + omega*current_n_deriv)/C
            prec_n_deriv = current_n_deriv

        return res
    # ==================================================================
    @overload
    @staticmethod
    def calc_dispersion(Lambda: float, beta_2: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_dispersion(Lambda: np.ndarray, beta_2: np.ndarray
                        ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_dispersion(Lambda, beta_2):
        r"""Calculate the dispersion parameter.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[ps \cdot nm^{-1} \cdot km^{-1}]`

        Notes
        -----

        .. math::  D = \frac{d}{d\lambda}\Big(\frac{1}{v_g}\Big)
                   = \frac{d}{d\lambda} \beta_1
                   = -\frac{2\pi c}{\lambda^2} \beta_2

        """
        if (isinstance(Lambda, float)):
            factor = (2.0 * cst.PI * cst.C) / (Lambda**2)
        else:
            factor = (2.0 * cst.PI * cst.C) / np.square(Lambda)

        return -1 * factor * beta_2
    # ==================================================================
    @overload
    @staticmethod
    def calc_dispersion_slope(Lambda: float, beta_2: float, beta_3: float
                              ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_dispersion_slope(Lambda: np.ndarray, beta_2: np.ndarray,
                              beta_3: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_dispersion_slope(Lambda, beta_2, beta_3):
        r"""Calculate the dispersio slope.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`
        beta_3 :
            The third order dispersion term. :math:`[ps^3\cdot km^{-1}]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[ps \cdot nm^{-2} \cdot km^{-1}]`

        Notes
        -----

        .. math::  S = \frac{d D}{d\lambda}
                   = \beta_2 \frac{d}{d\lambda} \Big(-\frac{2\pi c}
                   {\lambda^2}\Big) - \frac{2\pi c}
                   {\lambda^2} \frac{d\beta_2}{d\lambda}
                   = \frac{4\pi c}{\lambda^3} \beta_2
                   + \Big(\frac{2\pi c}{\lambda^2}\Big)^2 \beta_3

        """
        if (isinstance(Lambda, float)):
            factor = (2.0 * cst.PI * cst.C) / (Lambda**2)
        else:
            factor = (2.0 * cst.PI * cst.C) / np.square(Lambda)

        return (2 * factor / Lambda * beta_2) + (factor * factor * beta_3)
    # ==================================================================
    @overload
    @staticmethod
    def calc_RDS(Lambda: float, beta_2: float, beta_3: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_RDS(Lambda: np.ndarray, beta_2: np.ndarray,
                 beta_3: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_RDS(Lambda, beta_2, beta_3):
        r"""Calculate the relative dispersion slope.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`
        beta_3 :
            The third order dispersion term. :math:`[ps^3\cdot km^{-1}]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[nm^{-1}]`

        Notes
        -----

        .. math::  RDS = \frac{S}{D}

        """

        return (Dispersion.calc_dispersion_slope(beta_2, beta_3, Lambda)
                / Dispersion.calc_dispersion(beta_2, Lambda))
    # ==================================================================
    @overload
    @staticmethod
    def calc_accumulated_dispersion(Lambda: float, beta_2: float, length: float
                                    ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_accumulated_dispersion(Lambda: np.ndarray, beta_2: np.ndarray,
                                    length: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_accumulated_dispersion(Lambda, beta_2, length):
        r"""Calculate the dispersion parameter.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`
        length :
            The length over which dispersion is considered. :math:`[km]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[ps \cdot nm^{-1}]`

        Notes
        -----

        .. math::  D_{acc} = D \cdot L

        """

        return Dispersion.calc_dispersion(beta_2, Lambda) * length
    # ==================================================================
    @overload
    @staticmethod
    def calc_dispersion_length(width: float, beta_2: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_dispersion_length(width: np.ndarray, beta_2: np.ndarray
                               ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_dispersion_length(width, beta_2):
        r"""Calculate dispersion length.

        Parameters
        ----------
        width :
            The power. :math:`[ps]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`

        Returns
        -------
        :
            The dispersion length :math:`[km]`

        Notes
        -----

        .. math::  L_{D} = \frac{T_0^2}{|\beta_2|}

        """

        if (isinstance(width, float)):

            return width**2 / abs(beta_2)
        else:

            return np.square(width) / np.abs(beta_2)


if __name__ == "__main__":
    """Plot the GVD, dispersion and dispersion slope as a function of
    the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    center_omega: float = oc.lambda_to_omega(976.0)
    medium: str = "Sio2"
    # Betas values
    sellmeier: oc.Sellmeier = oc.Sellmeier(medium)
    disp: oc.ChromaticDisp = oc.ChromaticDisp(sellmeier)
    print('betas: ', disp(center_omega, 13))
    print('\n betas with callable: ',
          oc.ChromaticDisp.calc_beta(center_omega, 13, sellmeier))
    n_core = sellmeier(center_omega)
    print('\n betas with constant: ',
          oc.ChromaticDisp.calc_beta(center_omega, 13, n_core))
    # Dispersion coeff.
    lambdas: np.ndarray = np.linspace(900., 1600., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    beta_2: np.ndarray = oc.ChromaticDisp.calc_beta(omegas, 2, sellmeier)[2]
    x_data: List[np.ndarray] = [lambdas]
    y_data: List[np.ndarray] = [beta_2]
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['beta2']
    plot_titles: List[str] = ["Group velocity dispersion coefficients in Silica"]
    disp = oc.ChromaticDisp.calc_dispersion(lambdas, beta_2)
    x_data.append(lambdas)
    y_data.append(disp)
    x_labels.append('Lambda')
    y_labels.append('dispersion')
    plot_titles.append('Dispersion of Silica')
    # Dispersion slope coefficient
    beta_3: np.ndarray = oc.ChromaticDisp.calc_beta(omegas, 3, sellmeier)[3]
    slope: np.ndarray = oc.ChromaticDisp.calc_dispersion_slope(lambdas, beta_2,
                                                               beta_3)
    x_data.append(lambdas)
    y_data.append(slope)
    x_labels.append('Lambda')
    y_labels.append('dispersion_slope')
    plot_titles.append('Dispersion slope of Silica')

    oc.plot2d(x_data, y_data, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, split=True, line_opacities=[0.0])
