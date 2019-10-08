# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

""".. moduleauthor:: Sacha Medaer"""

from typing import Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.fft import FFT


class Raman(AbstractEffect):
    r"""Generic class for effect object.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    center_omega : numpy.ndarray of float
        The center angular frequency. :math:`[ps^{-1}]`

    """

    def __init__(self, T_R: float = cst.RAMAN_COEFF,
                 eta: float = cst.XPM_COEFF,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 cross_term: bool = False,
                 tau_1: float = cst.TAU_1, tau_2: float = cst.TAU_2,
                 tau_b: float = cst.TAU_B, f_a: float = cst.F_A,
                 f_b: float = cst.F_B, f_c: float = cst.F_C,
                 h_R: Optional[Array[float]] = None,
                 time: Optional[Array[float]] = None,
                 omega: Optional[Array[float]] = None) -> None:
        r"""
        Parameters
        ----------
        T_R :
            The raman coefficient. :math:`[]`
        eta :
            Positive term multiplying the cross terms in the effect.
        approx_type :
            The type of the NLSE approximation.
        cross_term :
            If True, trigger the cross-term influence in the effect.
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
        h_R :
            The Raman response function values.
        time :
            The time values of the field.
        omega:
            The angular frequencies values of the field.

        """
        super().__init__(omega, time)
        self._T_R: float = T_R
        self._eta: float = eta
        self._approx_type: int = approx_type
        self._cross_term: float = cross_term
        self._tau_1: float = tau_1
        self._tau_2: float = tau_2
        self._tau_b: float = tau_b
        self._f_a: float = f_a
        self._f_b: float = f_b
        self._f_c: float = f_c
        self._h_R: Optional[Array[float]] = h_R
        self._fft_h_R = None
    # ==================================================================
    @property
    def h_R(self) -> Array[float]:
        if (self._h_R is None):
            if (self._time is None):
                util.warning_terminal("Must specified time array to"
                    "calculate the Raman response function")
            else:
                self._h_R = self.calc_h_R(self._time, self._tau_1, self._tau_2,
                                          self._tau_b, self._f_a, self._f_b,
                                          self._f_c)
                # Save fft of h_R bcs might be reused and not change
                self._fft_h_R = FFT.fft(self._h_R)
        return self._h_R
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The non approximated operator of the Raman effect."""
        square_mod = waves[id]*np.conj(waves[id])

        return (1j * FFT.conv_to_fft(self.h_R, square_mod))#, self._fft_h_R))
    # ==================================================================
    def op_approx(self, waves: Array[cst.NPFT], id: int,
                  corr_wave: Optional[Array[cst.NPFT]] = None
                  ) -> Array[cst.NPFT]:
        """The approximation of the operator of the Raman effect."""
        res = self.op_approx_self(waves, id, corr_wave)
        if (self._cross_term):

            res += self.op_approx_cross(waves, id, corr_wave)

        return res
    # ==================================================================
    def op_approx_self(self, waves: Array[cst.NPFT], id: int,
                       corr_wave: Optional[Array[cst.NPFT]] = None
                       ) -> Array[cst.NPFT]:
        """The approximation of the operator of the Raman effect for the
        considered wave."""
        A = waves[id]
        res = np.zeros(A.shape, dtype=cst.NPFT)

        if (self._approx_type == cst.approx_type_1
                or self._approx_type == cst.approx_type_2):

            res =  FFT.dt_to_fft(A*np.conj(A), self._omega, 1)

        if (self._approx_type == cst.approx_type_3):

            res = (np.conj(A)*FFT.dt_to_fft(A, self._omega, 1)
                   + A*FFT.dt_to_fft(np.conj(A), self._omega, 1))

        return -1j * self._T_R * res
    # ==================================================================
    def op_approx_cross(self, waves: Array[cst.NPFT], id: int,
                        corr_wave: Optional[Array[cst.NPFT]] = None
                        ) -> Array[cst.NPFT]:
        """The approximation operator of the cross terms of the Raman
        effect."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        if (self._approx_type == cst.approx_type_1
                or self._approx_type == cst.approx_type_2
                or self._approx_type == cst.approx_type_3):
            for i in range(len(waves)):
                if (i != id):
                    res += waves[i]*np.conj(waves[id])
            res = FFT.dt_to_fft(res, self._omega, 1)

        return -1j * self._T_R * self._eta * res
    # ==================================================================
    @staticmethod
    def calc_h_R(time: Array[float], tau_1: float = cst.TAU_1,
                 tau_2: float = cst.TAU_2, tau_b: float = cst.TAU_B,
                 f_a: float = cst.F_A, f_b: float = cst.F_B,
                 f_c: float = cst.F_C) -> Array[float]:
        r"""Calculate the Raman response function as the sum of the
        isotropic :math:`R_a(t)` and anisotropic :math:`R_b(t)`
        (Boson peak) Raman response functions. [1]_

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
        .. [1] Lin, Q. and Agrawal, G.P., 2006. Raman response function
           for silica fibers. Optics letters, 31(21), pp.3086-3088.

        """

        return ((f_a+f_c) * Raman.calc_h_a(time, tau_1, tau_2)
                + f_b * Raman.calc_h_b(time, tau_b))
    # ==================================================================
    @staticmethod
    def calc_h_a(time: Array[float], tau_1: float = cst.TAU_1,
                 tau_2: float = cst.TAU_2) -> Array[float]:
        r"""Calculate the isotropic part of the Raman response function.
        [2]_

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
        .. [2] Lin, Q. and Agrawal, G.P., 2006. Raman response function
           for silica fibers. Optics letters, 31(21), pp.3086-3088.

        """

        #return ((tau_2**2 + tau_1**2) / (tau_1 * tau_2**2)
        return (tau_1 / (tau_2**2 + tau_1**2)
                * np.exp(-time/tau_2) * np.sin(time/tau_1))
    # ==================================================================
    @staticmethod
    def calc_h_b(time: Array[float], tau_b: float = cst.TAU_B
                 ) -> Array[float]:
        r"""Calculate the anisotropic part (boson peak) of the Raman
        response function. [3]_

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
        .. [3] Lin, Q. and Agrawal, G.P., 2006. Raman response function
           for silica fibers. Optics letters, 31(21), pp.3086-3088.

        """

        return ((2*tau_b - time) / (tau_b**2) * np.exp(-time/tau_b))
    # ==================================================================
    @staticmethod
    def calc_raman_gain(omega_bw: float, h_R: Array[float],
                        center_omega: float, n_0: float, f_R: float,
                        n_2: float) -> Array[float]:
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
        chi_3 = 4 * cst.EPS_0 * cst.LIGHT_SPEED * n_0**2 * n_2 / 3

        return (center_omega / cst.LIGHT_SPEED / n_0 * f_R * chi_3 *
                np.imag(FFT.fft(h_R)))


if __name__ == "__main__":

    import optcom.utils.constants as cst
    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.utils.fft import FFT

    samples = 1000
    time, dtime = np.linspace(0.0, 0.3, samples, False, True)  # ps
    freq_window = 1.0 / dtime

    x_data = []
    y_data = []
    plot_labels = []

    f_R: float = cst.F_R
    n_0: float = 1.40
    center_omega = Domain.lambda_to_omega(1550.0)

    f_a: float = cst.F_A
    f_b: float = cst.F_B
    f_c: float = cst.F_C
    x_data.append(time)
    h_R = Raman.calc_h_R(time, f_a=f_a, f_b=f_b, f_c=f_c)
    y_data.append(h_R)
    plot_labels.append('Isotropic and anisotropic part')
    f_a = 1.0
    f_b = 0.0
    f_c = 1.0
    x_data.append(time)
    h_R = Raman.calc_h_R(time, f_a=f_a, f_b=f_b, f_c=f_c)
    y_data.append(h_R)
    plot_labels.append('W/o anisotropic part')
    f_a = 0.0
    f_b = 1.0
    f_c = 0.0
    x_data.append(time)
    h_R = Raman.calc_h_R(time, f_a=f_a, f_b=f_b, f_c=f_c)
    y_data.append(h_R)
    plot_labels.append('W/o isotropic part')

    plot_titles = ['Raman response function', 'Raman gain']

    plot.plot(x_data, y_data, x_labels=['t'], y_labels=['h_R'],
              plot_groups=[0,0,0], plot_titles=plot_titles, opacity=0.0,
              plot_labels=plot_labels)
