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

from __future__ import annotations

from typing import Any, overload, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util


class Domain(object):
    r"""Contain shared information about fields propagating in the
    same layout. Contain static methods for main physics variable
    conversion.

    Attributes
    ----------
    bits : int
        Number of bits to consider.
    samples : int
        Total number of samples per signal.
    time : numpy.ndarray of float
        Absolute time values for any signal. :math:`[ps]`
    dtime : float
        The time step size. :math:`[ps]`
    time_window: float
        The time window. :math:`[ps]`
    omega : numpy.ndarray of float
        Absolute angular frequency values for any signal.
        :math:`[ps^{-1}]`
    domega : float
        The angular frequency step size. :math:`[ps^{-1}]`
    omega_window: float
        The angular frequency window. :math:`[ps^{-1}]`
    nu : numpy.ndarray of float
        Absolute frequency values for any signal. :math:`[ps^{-1}]`
    dnu : float
        The frequency step size. :math:`[ps^{-1}]`
    nu_window: float
        The frequency window. :math:`[ps^{-1}]`
    noise_samples :
        The number of samples in the noise wavelength range.
    noise_domega :
        The angular frequency step size of the noise. :math:`[ps^{-1}]`
    noise_omega_window :
        The angular frequency window of the noise. :math:`[ps^{-1}]`
    noise_omega :
        The angular frequency values of the noise :math:`[ps^{-1}]`

    """

    def __init__(self, bits: int = 1, bit_width: float = 100.0,
                 samples_per_bit: int = 512, memory_storage: float = 1.0,
                 noise_range: Tuple[float, float] = (900.0,1600.0),
                 noise_samples: int = 250) -> None:
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
        noise_range :
            The wavelength range in which the noise must be considered.
            :math:`[nm]`
        noise_samples :
            The number of samples in the noise wavelength range.

        """
        # Attr types check ---------------------------------------------
        util.check_attr_type(bits, 'bits', int)
        util.check_attr_type(bit_width, 'bit_width', int, float)
        util.check_attr_type(samples_per_bit, 'samples_per_bit', int)
        util.check_attr_type(memory_storage, 'memory_storage', int, float)
        util.check_attr_type(noise_range, 'noise_range', tuple)
        util.check_attr_type(noise_samples, 'noise_samples', int)
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
        # Angular Frequency --------------------------------------------
        # log for omega compared to t
        self._omega_window = Domain.nu_to_omega(1.0 / self._dtime)
        self._omega, self._domega = np.linspace(-0.5*self._omega_window,
                                                0.5*self._omega_window,
                                                self._samples, False,
                                                True)
        # Noise --------------------------------------------------------
        omega_noise_lower = Domain.lambda_to_omega(noise_range[1])
        omega_noise_upper = Domain.lambda_to_omega(noise_range[0])
        self._noise_omega_window: float = omega_noise_upper - omega_noise_lower
        self._noise_samples: int = noise_samples
        self._noise_omega, self._noise_domega = np.linspace(omega_noise_lower,
                                                            omega_noise_upper,
                                                            noise_samples,
                                                            False, True)
    # ==================================================================
    # In-build methods =================================================
    # ==================================================================
    def __eq__(self, operand) -> bool:
        """Two domains are equal if they share the same characteristics.
        """
        if (not isinstance(operand, Domain)):

            return False
        else:

            return ((self.memory == operand.memory)
                    and (self.bits == operand.bits)
                    and (self.samples == operand.samples)
                    and (np.array_equal(self.time, operand.time))
                    and (self.dtime == operand.dtime)
                    and (self.time_window == operand.time_window)
                    and (np.array_equal(self.omega, operand.omega))
                    and (self.domega == operand.domega)
                    and (self.omega_window == operand.omega_window)
                    and (self.noise_samples == operand.noise_samples)
                    and (self.noise_omega_window == operand.noise_omega_window)
                    and (np.array_equal(self.noise_omega, operand.noise_omega))
                    and (self.noise_domega == operand.noise_domega))
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
    def time(self) -> np.ndarray:

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
    def omega(self) -> np.ndarray:

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
    def nu(self) -> np.ndarray:

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
    @property
    def noise_samples(self) -> int:

        return self._noise_samples
    # ==================================================================
    @property
    def noise_domega(self) -> float:

        return self._noise_domega
    # ==================================================================
    @property
    def noise_omega_window(self) -> float:

        return self._noise_omega_window
    # ==================================================================
    @property
    def noise_nu(self) -> np.ndarray:

        return Domain.omega_to_nu(self._noise_omega)
    # ==================================================================
    @property
    def noise_omega(self) -> np.ndarray:

        return self._noise_omega
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
    def nu_to_omega(nu: np.ndarray) -> np.ndarray: ...
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
    def nu_to_lambda(nu: np.ndarray) -> np.ndarray: ...
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

        return cst.C / nu
    # ==================================================================
    @overload
    @staticmethod
    def omega_to_nu(omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def omega_to_nu(omega: np.ndarray) -> np.ndarray: ...
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
    def omega_to_lambda(omega: np.ndarray) -> np.ndarray: ...
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


        return 2.0 * cst.PI * cst.C / omega
    # ==================================================================
    # lambda is reserved word in python
    @overload
    @staticmethod
    def lambda_to_nu(Lambda: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def lambda_to_nu(Lambda: np.ndarray) -> np.ndarray: ...
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

        return cst.C / Lambda
    # ==================================================================
    # lambda is reserved word in python
    @overload
    @staticmethod
    def lambda_to_omega(Lambda: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def lambda_to_omega(Lambda: np.ndarray) -> np.ndarray: ...
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

        return 2.0 * cst.PI * cst.C / Lambda
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

        return cst.C * nu_bw / center_nu**2
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

        return cst.C * lambda_bw / center_lambda**2
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

        return 2 * cst.PI * cst.C * omega_bw / center_omega**2
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

        return 2 * cst.PI * cst.C * lambda_bw / center_lambda**2
    # ==================================================================
    # Others ===========================================================
    # ==================================================================
    def get_shift_time(self, rel_pos: float) -> np.ndarray:

        return self.time - rel_pos*self.time_window


if __name__ == "__main__":

    nu: float = 1550.0
    print(Domain.nu_to_omega(nu))
    print(Domain.nu_to_lambda(nu))
    print(Domain.lambda_to_nu(nu))
    print(Domain.lambda_to_omega(nu))
    print(Domain.omega_to_nu(nu))
    print(Domain.omega_to_lambda(nu))
