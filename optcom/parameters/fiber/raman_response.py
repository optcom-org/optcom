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
from typing import List, Optional, overload, Union, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.fft import FFT


class RamanResponse(AbstractParameter):

    def __init__(self, tau_1: float = cst.TAU_1, tau_2: float = cst.TAU_2,
                 tau_b: float = cst.TAU_B, f_a: float = cst.F_A,
                 f_b: float = cst.F_B, f_c: float = cst.F_C) -> None:
        r"""
        Parameters
        ----------
        tau_1 :
            The inverse of vibrational frequency of the fiber core
            molecules. :math:`[ps]`
        tau_2 :
            The damping time of vibrations. :math:`[ps]`
        tau_b :
            The spectral width of the Boson peak. :math:`[ps]`
        f_a :
            The fractional contribution of the isotropic part to the
            isotropic Raman function.
        f_b :
            The fractional contribution of the anisotropic part to the
            anisotropic Raman function.
        f_c :
            The fractional contributian of the isotropic part to the
            anisotropic Raman function.

        """
        self._tau_1: float = tau_1
        self._tau_2: float = tau_2
        self._tau_b: float = tau_b
        self._f_a: float = f_a
        self._f_b: float = f_b
        self._f_c: float = f_c
    # ==================================================================
    @overload
    def __call__(self, time: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, time: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, time):
        r"""Compute the Raman response function.

        Parameters
        ----------
        time :
            The time. :math:`[ps]`

        Returns
        -------
        :
            The values of the Raman response function. :math:`[ps^{-1}]`

        """

        return RamanResponse.calc_h_R(time, self._tau_1, self._tau_2,
                                      self._tau_b, self._f_a, self._f_b,
                                      self._f_c)
    # ==================================================================
    @staticmethod
    def calc_h_R(time: np.ndarray, tau_1: float = cst.TAU_1,
                 tau_2: float = cst.TAU_2, tau_b: float = cst.TAU_B,
                 f_a: float = cst.F_A, f_b: float = cst.F_B,
                 f_c: float = cst.F_C) -> np.ndarray:
        r"""Calculate the Raman response function as the sum of the
        isotropic :math:`R_a(t)` and anisotropic :math:`R_b(t)`
        (Boson peak) Raman response functions. [7]_

        Parameters
        ----------
        time :
            The time value(s). :math:`[ps]`
        tau_1 :
            The inverse of vibrational frequency of the fiber core
            molecules. :math:`[ps]`
        tau_2 :
            The damping time of vibrations. :math:`[ps]`
        tau_b :
            The spectral width of the Boson peak. :math:`[ps]`
        f_a :
            The fractional contribution of the isotropic part to the
            isotropic Raman function.
        f_b :
            The fractional contribution of the anisotropic part to the
            anisotropic Raman function.
        f_c :
            The fractional contributian of the isotropic part to the
            anisotropic Raman function.

        Returns
        -------
        :
            Value of the Raman response function at the time given as
            parameter. :math:`[ps^{-1}]`

        Notes
        -----

        .. math:: \begin{split}
                  h_R(t) &= R_a(t) + R_b(t)\\
                  &= (f_a h_a(t)) + (f_b h_b(t) + f_c h_a(t))\\
                  &= (f_a+f_c)h_a(t) + f_b h_b(t)
                  \end{split}

        References
        ----------
        .. [7] Lin, Q. and Agrawal, G.P., 2006. Raman response function
           for silica fibers. Optics letters, 31(21), pp.3086-3088.

        """

        return ((f_a+f_c) * RamanResponse.calc_h_a(time, tau_1, tau_2)
                + f_b * RamanResponse.calc_h_b(time, tau_b))
    # ==================================================================
    @staticmethod
    def calc_h_a(time: np.ndarray, tau_1: float = cst.TAU_1,
                 tau_2: float = cst.TAU_2) -> np.ndarray:
        r"""Calculate the isotropic part of the Raman response function.
        [8]_

        Parameters
        ----------
        time :
            The time value(s). :math:`[ps]`
        tau_1 :
            The inverse of vibrational frequency of the fiber core
            molecules. :math:`[ps]`
        tau_2 :
            The damping time of vibrations. :math:`[ps]`

        Returns
        -------
        :
            Value of the isotropic part of the Raman response function
            at the time given as parameter. :math:`[ps^{-1}]`

        Notes
        -----

        .. math:: h_a(t) = \frac{\tau_1}{\tau_1^2 + \tau_2^2}
                  e^{\frac{-t}{\tau_2}} \sin\big(\frac{t}{\tau_1}\big)

        References
        ----------
        .. [8] Lin, Q. and Agrawal, G.P., 2006. Raman response function
           for silica fibers. Optics letters, 31(21), pp.3086-3088.

        """

        #return ((tau_2**2 + tau_1**2) / (tau_1 * tau_2**2)
        return (tau_1 / (tau_2**2 + tau_1**2)
                * np.exp(-time/tau_2) * np.sin(time/tau_1))
    # ==================================================================
    @staticmethod
    def calc_h_b(time: np.ndarray, tau_b: float = cst.TAU_B
                 ) -> np.ndarray:
        r"""Calculate the anisotropic part (boson peak) of the Raman
        response function. [9]_

        Parameters
        ----------
        time :
            The time value(s). :math:`[ps]`
        tau_b :
            The spectral width of the Boson peak. :math:`[ps]`

        Returns
        -------
        :
            Value of the anisotropic part of the Raman response function
            at the time given as parameter. :math:`[ps^{-1}]`

        Notes
        -----

        .. math:: h_b(t) = \frac{2\tau_b - t}{\tau_b^2}
                  e^{\frac{-t}{\tau_b}}

        References
        ----------
        .. [9] Lin, Q. and Agrawal, G.P., 2006. Raman response function
           for silica fibers. Optics letters, 31(21), pp.3086-3088.

        """

        return ((2*tau_b - time) / (tau_b**2) * np.exp(-time/tau_b))
    # ==================================================================
    @staticmethod
    def calc_raman_gain(omega_bw: float, h_R: np.ndarray,
                        center_omega: float, n_0: float, f_R: float,
                        n_2: float) -> np.ndarray:
        r"""Calculate the Raman gain.

        Parameters
        ----------
        omega_bw :
            The angular frequency bandwidth. :math:`[ps^{-1}]`
        h_R :
            The raman response function. :math:`[ps^{-1}]`
        center_omega :
            The center angular frequency. :math:`[ps^{-1}]`
        n_0 :
            The refractive index.
        f_R :
            The fractional contribution of the delayed Raman response.
            :math:`[]`
        n_2 :
            The non-linear index. :math:`[m^2\cdot W^{-1}]`

        Returns
        -------
        :
            The Raman gain.

        Notes
        -----

        .. math::  g_R(\Delta \omega) = \frac{\omega_0}{c n_0} f_R
                   \chi^{(3)} \Im{\hat{h}_R(\Delta \omega)}

        where:

        .. math::  \chi^{(3)} = \frac{4\epsilon_0 c n^2}{3} n_2

        :math:`\chi^{(3)}` is in :math:`[m^2 \cdot V^{-2}]`.

        """

        n_2 *= 1e36  # m^2 W^{-1} = s^3 kg^-1 -> ps^3 kg^-1
        chi_3 = 4 * cst.EPS_0 * cst.C * n_0**2 * n_2 / 3

        return (center_omega / cst.C / n_0 * f_R * chi_3
                * np.imag(FFT.fft(h_R)))


if __name__ == "__main__":
    """Plot the Raman response as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List, Optional

    import numpy as np

    import optcom as oc
    import optcom.utils.constants as cst

    samples: int = 1000
    time: np.ndarray
    dtime: float
    time, dtime = np.linspace(0.0, 0.3, samples, False, True)  # ps

    x_data: List[np.ndarray] = []
    y_data: List[np.ndarray] = []
    line_labels: List[Optional[str]] = []

    f_R: float = cst.F_R
    n_0: float = 1.40
    center_omega: float = oc.lambda_to_omega(1550.0)

    f_a: float = cst.F_A
    f_b: float = cst.F_B
    f_c: float = cst.F_C
    x_data.append(time)
    h_R: oc.RamanResponse = oc.RamanResponse.calc_h_R(time, f_a=f_a, f_b=f_b,
                                                      f_c=f_c)
    y_data.append(h_R)
    line_labels.append('Isotropic and anisotropic part')
    f_a = 1.0
    f_b = 0.0
    f_c = 1.0
    x_data.append(time)
    h_R = oc.RamanResponse.calc_h_R(time, f_a=f_a, f_b=f_b, f_c=f_c)
    y_data.append(h_R)
    line_labels.append('W/o anisotropic part')
    f_a = 0.0
    f_b = 1.0
    f_c = 0.0
    x_data.append(time)
    h_R = oc.RamanResponse.calc_h_R(time, f_a=f_a, f_b=f_b, f_c=f_c)
    y_data.append(h_R)
    line_labels.append('W/o isotropic part')
    plot_titles: List[str] = ['Raman response function', 'Raman gain']

    oc.plot2d(x_data, y_data, x_labels=['t'], y_labels=['h_R'],
              plot_groups=[0,0,0], plot_titles=plot_titles,
              line_opacities=[0.0], line_labels=line_labels)
