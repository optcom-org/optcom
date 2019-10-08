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

from __future__ import annotations

from typing import Any, overload

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util


class Domain(object):
    r"""Contain shared information about fields propagating in the
    layout.

    Attributes
    ----------
    bits : int
        Number of bits to consider.
    samples : int
        Total number of samples per signal.
    time : numpy.ndarray of float
        Absolute time values for any signal.
    dtime : float
        The time step size.
    time_window: float
        The time window.
    omega : numpy.ndarray of float
        Absolute angular frequency values for any signal.
    domega : float
        The angular frequency step size.
    omega_window: float
        The angular frequency window.
    nu : numpy.ndarray of float
        Absolute frequency values for any signal.
    dnu : float
        The frequency step size.
    nu_window: float
        The frequency window.

    """

    def __init__(self, bits: int = 1, bit_width: float = 100.0,
                 samples_per_bit: int = 512, memory_storage: float = 1.0
                 ) -> None:
        r"""
        Parameters
        ----------
        bits :
            Number of bits to consider.
        bit_width :
            The width of one bit. :math:`[ps]`
        samples_per_bit :
            Number of samples per bit.
        memory_storage :
            Max memory available if recording all steps of computation.
            Will be used if the attribute :attr:`save_all` of
            :class:`AbstractComponent` is True. :math:`[Gb]`

        """
        # Attr types check ---------------------------------------------
        util.check_attr_type(bits, 'bits', int)
        util.check_attr_type(bit_width, 'bit_width', int, float)
        util.check_attr_type(samples_per_bit, 'samples_per_bit', int)
        util.check_attr_type(memory_storage, 'memory_storage', int, float)
        # Attr range check ---------------------------------------------
        util.check_attr_range(bits, 'bits', cst.MIN_BITS, cst.MAX_BITS)
        util.check_attr_range(samples_per_bit, 'samples_per_bit',
                              cst.MIN_SAMPLES_PER_BIT, cst.MAX_SAMPLES_PER_BIT)
        util.check_attr_range(bit_width, 'bit_width', cst.MIN_BIT_WIDTH,
                              cst.MAX_BIT_WIDTH)
        # Attr ---------------------------------------------------------
        self._memory_storage: float = memory_storage
        self._bits: int = bits
        self._samples_per_bit: int = samples_per_bit
        self._bit_width: float = bit_width
        self._samples: int = int(self._bits * self._samples_per_bit)
        # Time ---------------------------------------------------------
        self._time_window: float = self._bits * self._bit_width
        self._time, self._dtime = np.linspace(0.0, self._time_window,
                                              self._samples, False, True)
        #  Angular Frequency -------------------------------------------
        # log for omega compared to t
        self._omega_window = Domain.nu_to_omega(1.0 / self._dtime)
        self._omega, self._domega = np.linspace(-0.5*self._omega_window,
                                                0.5*self._omega_window,
                                                self._samples, False,
                                                True)
    # ==================================================================
    # Getters ==========================================================
    # ==================================================================
    # NB: only getters, attribute should be immutable
    @property
    def memory(self) -> float:

        return self._memory_storage
    # ==================================================================
    @property
    def bits(self) -> int:

        return self._bits
    # ==================================================================
    @property
    def samples(self) -> int:

        return self._samples
    # ==================================================================
    @property
    def time(self) -> Array[float, 1,...]:

        return self._time
    # ==================================================================
    @property
    def dtime(self) -> float:

        return self._dtime
    # ==================================================================
    @property
    def time_window(self) -> float:

        return self._time_window
    # ==================================================================
    @property
    def omega(self) -> Array[float, 1,...]:

        return self._omega
    # ==================================================================
    @property
    def domega(self) -> float:

        return self._domega
    # ==================================================================
    @property
    def omega_window(self) -> float:

        return self._omega_window
    # ==================================================================
    @property
    def nu(self) -> Array[float, 1,...]:

        return Domain.omega_to_nu(self._omega)
    # ==================================================================
    @property
    def dnu(self) -> float:

        return Domain.omega_to_nu(self._domega)
    # ==================================================================
    @property
    def nu_window(self) -> float:

        return Domain.omega_to_nu(self._omega_window)
    # Not considering wavelength as an absolute value shared among all
    # fields (see AOC lab 2 for details).  Need the center wavelength
    # to convert nu/omega window to lambda window which can be different
    # for each channels of each fields.
    # ==================================================================
    # nu-omega-Lambda static conversion ================================
    # ==================================================================
    # NB for typing: the most specific type must be first for overload
    @overload
    @staticmethod
    def nu_to_omega(nu: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def nu_to_omega(nu: Array[float, 1,...]) -> Array[float, 1,...]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def nu_to_omega(nu):
        r"""Convert frequency to angular frequency.

        Parameters
        ----------
        nu : float or numpy.ndarray of float
            The frequency. :math:`[THz]`

        Returns
        -------
        float numpy.ndarray of float
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Notes
        -----

        .. math:: \omega = 2\pi\nu

        """

        return 2.0 * cst.PI * nu
    # ==================================================================
    @overload
    @staticmethod
    def nu_to_lambda(nu: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def nu_to_lambda(nu: Array[float, 1,...]) -> Array[float, 1,...]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def nu_to_lambda(nu):
        r"""Convert frequency to wavelength.

        Parameters
        ----------
        nu : float or numpy.ndarray of float
            The frequency. :math:`[THz]`

        Returns
        -------
        float or numpy.ndarray of float
            The wavelength. :math:`[nm]`

        Notes
        -----

        .. math:: \lambda = \frac{c}{\nu}

        """

        return cst.LIGHT_SPEED / nu
    # ==================================================================
    @overload
    @staticmethod
    def omega_to_nu(omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def omega_to_nu(omega: Array[float, 1,...]) -> Array[float, 1,...]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def omega_to_nu(omega):
        r"""Convert angular frequency to frequency.

        Parameters
        ----------
        omega : float or numpy.ndarray of float
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        float or numpy.ndarray of float
            The frequency. :math:`[THz]`

        Notes
        -----

        .. math:: \nu = \frac{\omega}{2\pi}

        """

        return omega /  (2.0 * cst.PI)
    # ==================================================================
    @overload
    @staticmethod
    def omega_to_lambda(omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def omega_to_lambda(omega: Array[float, 1,...]
                        ) -> Array[float, 1,...]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def omega_to_lambda(omega):
        r"""Convert angular frequency to wavelength.

        Parameters
        ----------
        omega : float or numpy.ndarray of float
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        float or numpy.ndarray of float
            The wavelength. :math:`[nm]`

        Notes
        -----

        .. math:: \lambda = \frac{2\pi c}{\omega}

        """


        return 2.0 * cst.PI * cst.LIGHT_SPEED / omega
    # ==================================================================
    # lambda is reserved word in python
    @overload
    @staticmethod
    def lambda_to_nu(Lambda: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def lambda_to_nu(Lambda: Array[float, 1,...]) -> Array[float, 1,...]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def lambda_to_nu(Lambda):
        r"""Convert wavelength to frequency.

        Parameters
        ----------
        Lambda : float or numpy.ndarray of float
            The wavelength. :math:`[nm]`

        Returns
        -------
        float or numpy.ndarray of float
            The frequency. :math:`[THz]`

        Notes
        -----

        .. math:: \nu = \frac{c}{\lambda}

        """

        return cst.LIGHT_SPEED / Lambda
    # ==================================================================
    # lambda is reserved word in python
    @overload
    @staticmethod
    def lambda_to_omega(Lambda: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def lambda_to_omega(Lambda: Array[float, 1,...]
                        ) -> Array[float, 1,...]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def lambda_to_omega(Lambda):
        r"""Convert wavelength to angular frequency.

        Parameters
        ----------
        Lambda : float or numpy.ndarray of float
            The wavelength. :math:`[nm]`

        Returns
        -------
        float or numpy.ndarray of float
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Notes
        -----

        .. math:: \omega = \frac{2\pi c}{\lambda}

        """

        return 2.0 * cst.PI * cst.LIGHT_SPEED / Lambda
    # ==================================================================
    @staticmethod
    def nu_bw_to_lambda_bw(nu_bw: float, center_nu: float) -> float:
        r"""Convert the frequency bandwidth to wavelength bandwidth.

        Parameters
        ----------
        nu_bw :
            The frequency bandwidth. :math:`[THz]`
        center_nu :
            The center frequency. :math:`[THz]`


        Returns
        -------
        :
            The wavelength bandwidth. :math:`[nm]`

        Notes
        -----

        .. math:: \Delta \lambda = \frac{c}{\nu_0^2}\Delta\nu

        """

        return cst.LIGHT_SPEED * nu_bw / center_nu**2
    # ==================================================================
    @staticmethod
    def lambda_bw_to_nu_bw(lambda_bw: float, center_lambda: float) -> float:
        r"""Convert the wavelength bandwidth to frequency bandwidth.

        Parameters
        ----------
        lambda_bw :
            The wavelength bandwidth. :math:`[nm]`
        center_lambda :
            The center wavelength. :math:`[nm]`


        Returns
        -------
        :
            The frequency bandwidth. :math:`[THz]`

        Notes
        -----

        .. math:: \Delta \nu = \frac{c}{\lambda_0^2}\Delta\lambda

        """

        return cst.LIGHT_SPEED * lambda_bw / center_lambda**2
    # ==================================================================
    @staticmethod
    def omega_bw_to_lambda_bw(omega_bw: float, center_omega: float) -> float:
        r"""Convert the angular frequency bandwidth to wavelength
        bandwidth.

        Parameters
        ----------
        omega_bw :
            The angular frequency bandwidth. :math:`[rad\cdot ps^{-1}]`
        center_omega :
            The center angular frequency. :math:`[rad\cdot ps^{-1}]`


        Returns
        -------
        :
            The wavelength bandwidth. :math:`[nm]`

        Notes
        -----

        .. math:: \Delta\lambda = \frac{2\pi c}{\omega_0^2}\Delta\omega

        """

        return 2 * cst.PI * cst.LIGHT_SPEED * omega_bw / center_omega**2
    # ==================================================================
    @staticmethod
    def lambda_bw_to_omega_bw(lambda_bw: float, center_lambda: float) -> float:
        r"""Convert the wavelength bandwidth to angular frequency
        bandwidth.

        Parameters
        ----------
        lambda_bw :
            The wavelength bandwidth. :math:`[nm]`
        center_lambda :
            The center wavelength. :math:`[nm]`


        Returns
        -------
        :
            The angular frequency bandwidth. :math:`[rad\cdot ps^{-1}]`

        Notes
        -----

        .. math:: \Delta\omega = \frac{2\pi c}{\lambda_0^2}\Delta\lambda

        """

        return 2 * cst.PI * cst.LIGHT_SPEED * lambda_bw / center_lambda**2
    # ==================================================================
    # Others ===========================================================
    # ==================================================================
    def get_shift_time(self, rel_pos: float) -> Array[float, 1, ...]:

        return self.time - rel_pos*self.time_window

if __name__ == "__main__":

    nu: float = 1550.0
    print(Domain.nu_to_omega(nu))
    print(Domain.nu_to_lambda(nu))
    print(Domain.lambda_to_nu(nu))
    print(Domain.lambda_to_omega(nu))
    print(Domain.omega_to_nu(nu))
    print(Domain.omega_to_lambda(nu))
