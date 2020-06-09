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

import copy
import operator
import warnings
from typing import Any, List, Union, overload, Optional, Tuple

import numpy as np

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.utils.fft import FFT


# Exceptions
class FieldError(Exception):
    pass

class ShapeError(FieldError):
    pass

class OperatorError(FieldError):
    pass

class FieldWarning(UserWarning):
    pass

class DimWarning(FieldWarning):
    pass

class IndexWarning(FieldWarning):
    pass

class CompatibilityWarning(FieldWarning):
    pass

# Decorators
class IOperator():

    def __init__(self):

        return None

    def __call__(self, ioperator):

        def wrap_ioperator(obj, operand):

            iop = ioperator(obj, operand)
            if (isinstance(operand, Field)):
                if (obj.has_equal_charact(operand)):
                    map, map_ = Field.get_common_channels_maps(obj, operand)
                    obj[:][map] = iop(obj[:][map], operand[:][map_])
                else:

                    raise OperatorError("Two fields must have same "
                        "characteristics to support global operators.")
            else:
                iop(obj[:], operand)

            return obj

        return wrap_ioperator


class Operator():

    def __init__(self):

        return None

    def __call__(self, operator):

        def wrap_operator(obj, operand):

            op = operator(obj, operand)
            field = copy.deepcopy(obj)
            if (isinstance(operand, Field)):
                if (obj.has_equal_charact(operand)):
                    map, map_ = Field.get_common_channels_maps(field, operand)
                    field[:][map] = op(field[:][map], operand[:][map_])
                else:

                    raise OperatorError("Two fields must have same "
                        "characteristics to support global operators.")
            else:
                field[:] = op(field[:], operand)

            return field

        return wrap_operator


class ROperator():

    def __init__(self):

        return None

    def __call__(self, operator):

        def wrap_operator(obj, operand):

            op = operator(obj, operand)
            field = copy.deepcopy(obj)
            if (isinstance(operand, Field)):
                if (obj.has_equal_charact(operand)):
                    map, map_ = Field.get_common_channels_maps(field, operand)
                    field[:][map] = op(operand[:][map_], field[:][map])
                else:

                    raise OperatorError("Two fields must have same "
                        "characteristics to support global operators.")
            else:
                field[:] = op(operand, field[:])

            return field

        return wrap_operator


_ioperator = IOperator()
_operator = Operator()
_roperator = ROperator()


class Field(object):
    """Represent a field made of several channels. At each channel is
    associated a delay, a repetition frequency and a center omega.
    A field is associated with one noise array. The channels and noise
    characteristics must be consistent the provided domain. This is to
    ensure consistency between all fields propagating in the layout
    sharing equivalent domains.  N.B. 1 : please pay attention that
    the in-built operator __eq__ is overwritten, i.e. 'neq', 'not in',
    'in', 'eq' operators are depending on the overwritten definition.
    N.B. 2 : A field can contain different channels center at the same
    angular frequencies and having the same repetition frequency,
    however, most of the Field methods will assume that each tuple of
    angular frequency and repetition frequency is unique. (e.g. math
    operators overload).

    Attributes
    ----------
    type : str
        The type of the fields.
        See :mod:`optcom/utils/constant_values/field_types`.
    name : str
        The name of the field.

    """

    __nbr_default_names: int = 0
    # Trick needed to call the roperator on Field with numpy arrays
    __array_ufunc__ = None

    def __init__(self, domain: Domain, type: str,
                 name: str = cst.DEFAULT_FIELD_NAME) -> None:
        """
        Parameters
        ----------
        domain : optcom.domain.Domain
            The domain which the field is bound to.
        type :
            The type of the fields.
            See :mod:`optcom/utils/constant_values/field_types`.
        name :
            The name of the field.

        """
        self._domain: Domain = domain
        self._samples: int = domain.samples
        self.type: str = type
        self.name: str = ''
        if (name == cst.DEFAULT_FIELD_NAME or not name):
            Field.__nbr_default_names += 1
            self.name = (cst.DEFAULT_FIELD_NAME + ' '
                         + str(Field.__nbr_default_names))
        else:
            self.name = name
        self._nbr_channels: int = 0
        #self._channels: np.ndarray[cst.NPFT, self._nbr_channels, self._samples]
        self._channels: np.ndarray = np.array([], dtype=cst.NPFT)
        #self._center_omega: np.ndarray[float, self._nbr_channels]
        self._center_omega: np.ndarray = np.array([], dtype=float)
        #self._delays: np.ndarray[float, self._nbr_channels]
        self._delays: np.ndarray = np.array([], dtype=float)
        #self._noise: np.ndarray[float, domain.noise_samples]
        self._noise: np.ndarray = np.zeros(domain.noise_samples, dtype=float)
        #self._rep_freq: np.ndarray[float, self._nbr_channels]
        self._rep_freq: np.ndarray = np.array([], dtype=float)
    # ==================================================================
    # In-build methods =================================================
    # ==================================================================
    def __str__(self) -> str:

        return str(self[:])
    # ==================================================================
    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:

        return self._channels[key]
    # ==================================================================
    def __setitem__(self, key: int, channel: np.ndarray) -> None:

        if (len(channel) == len(self._channels[key])):
            self._channels[key] = channel.astype(cst.NPFT)
        else:
            warning_message: str = ("All channels must have the same "
                "dimension, can not change channel.")
            warnings.warn(warning_message, DimWarning)
    # ==================================================================
    def __delitem__(self, key: int) -> None:

        self._channels = np.delete(self._channels, key, axis=0)
        self._center_omega = np.delete(self._center_omega, key)
        self._delays = np.delete(self._delays, key)
        self._rep_freq = np.delete(self._rep_freq, key)
        self._nbr_channels -= 1
    # ==================================================================
    def __len__(self) -> int:

        return self._channels.shape[0]
    # ==================================================================
    def __eq__(self, operand) -> bool:
        """Two fields are equal if they share the same characteristics,
        i.e. they do not need to have the same values of delays, noises
        and channels, but need to have the same center angular
        frequencies, repetition frequencies, samples, domain and type.
        """
        if (not isinstance(operand, Field)):

            return False
        else:
            equal: bool
            equal = ((self.domain == operand.domain)
                     and (self.nbr_channels == operand.nbr_channels)
                     and (self.samples == operand.samples)
                     and (self.type == operand.type)
                     and (set(self.center_omega) == set(operand.center_omega))
                     and (set(self.rep_freq) == set(operand.rep_freq)))
            if (equal):     # Check if rep_freq equal order center omega
                order_map: np.ndarray = util.array_equal_map(
                    self.center_omega, operand.center_omega)
                equal = (equal and np.array_equal(self.rep_freq,
                                                  operand.rep_freq[order_map]))

            return equal
    # ==================================================================
    def has_equal_charact(self, field: Field) -> bool:
        """Return True if the current and provided field have the same
        characteristics, i.e. equivalent domains, same number of
        samples and type.
        """

        return ((self._samples == field.samples) and (self.type == field.type)
                and (self.domain == field.domain))
    # ==================================================================
    # Operators overload and management ================================
    # ==================================================================
    @_ioperator
    def __iadd__(self, operand):

        return operator.__iadd__
    # ==================================================================
    @_ioperator
    def __isub__(self, operand):

        return operator.__isub__
    # ==================================================================
    @_ioperator
    def __imul__(self, operand):

        return operator.__imul__
    # ==================================================================
    @_ioperator
    def __itruediv__(self, operand):

        return operator.__itruediv__
    # ==================================================================
    @_operator
    def __add__(self, operand):

        return operator.__add__
    # ==================================================================
    @_operator
    def __sub__(self, operand):

        return operator.__sub__
    # ==================================================================
    @_operator
    def __mul__(self, operand):

        return operator.__mul__
    # ==================================================================
    @_operator
    def __truediv__(self, operand):

        return operator.__truediv__
    # ==================================================================
    @_roperator
    def __radd__(self, operand):

        return operator.__add__
    # ==================================================================
    @_roperator
    def __rsub__(self, operand):

        return operator.__sub__
    # ==================================================================
    @_roperator
    def __rmul__(self, operand):

        return operator.__mul__
    # ==================================================================
    @_roperator
    def __rtruediv__(self, operand):

        return operator.__truediv__
    # ==================================================================
    @staticmethod
    def get_common_channels_maps(field_1: Field, field_2: Field
                                 )-> Tuple[np.ndarray, np.ndarray]:
        r"""Return an index map for each of the field where their
        channels match depending on config flags
        `FIELD_OP_MATCHING_OMEGA` and `FIELD_OP_MATCHING_REP_FREQ`

        Parameters
        ----------
        field_1 : optcom.field.Field
            The first field of the operation.
        field_2 : optcom.field.Field
            The second field of the operation.

        """
        map_co: np.ndarray
        map_co_: np.ndarray
        map_rp: np.ndarray
        map_rp_: np.ndarray
        map_res: np.ndarray
        map_res_: np.ndarray
        map_co, map_co_ = util.array_equal_duo_map(field_1.center_omega,
                                                   field_2.center_omega)
        map_rp, map_rp_ = util.array_equal_duo_map(field_1.rep_freq,
                                                   field_2.rep_freq)
        if (cfg.FIELD_OP_MATCHING_OMEGA and cfg.FIELD_OP_MATCHING_REP_FREQ):
            map_res = np.intersect1d(map_co, map_rp)
            map_res_ = np.intersect1d(map_co_, map_rp_)
        elif (cfg.FIELD_OP_MATCHING_OMEGA):
            map_res = map_co
            map_res_ = map_co_
        elif (cfg.FIELD_OP_MATCHING_REP_FREQ):
            map_res = map_rp
            map_res_ = map_rp_
        else:
            map_res = np.zeros(0, dtype=int)
            map_res_ = np.zeros(0, dtype=int)

        return map_res, map_res_
    # ==================================================================
    def operator_or_extend(self, operator: str, field_operand: Field) -> Field:
        """Execute the operation 'operator' between current Field and
        field operand for their common channels. The channels of field
        operand which do not have corresponding channels in the current
        field will be added to it.

        Parameters
        ----------
        operator :
            An operator name, must be a valid name from python
            `operator` module.
        field_operand : optcom.field.Field
            The operand of the operator (must be a Field).

        """
        map: np.ndarray
        _, map = Field.get_common_channels_maps(self, field_operand)
        res: Field = getattr(self, operator)(field_operand)
        for i in range(len(field_operand[:])):
            if (i not in map):
                res.add_channel(field_operand[i],
                                field_operand.center_omega[i],
                                field_operand.rep_freq[i],
                                field_operand.delays[i])

        return res
    # ==================================================================
    # Getters - setters - deleter ======================================
    # ==================================================================
    @property
    def time(self) -> np.ndarray:
        time = np.zeros((self._nbr_channels, self._samples))
        if (self._nbr_channels):
            for i in range(len(self._delays)):
                time[i] = (self._domain.time+self._delays[i])

        return time
    # ==================================================================
    @property
    def omega(self) -> np.ndarray:
        omega = np.zeros((self._nbr_channels, self._samples))
        if (self._nbr_channels):
            for i in range(len(self._center_omega)):
                omega[i] = (self._domain.omega+self._center_omega[i])

        return omega
    # ==================================================================
    @property
    def nu(self) -> np.ndarray:
        nu = np.zeros((self._nbr_channels, self._samples))
        if (self._nbr_channels):
            omega = self.omega
            for i in range(omega.shape[0]):
                nu[i] = Domain.omega_to_nu(omega[i])

        return nu
    # ==================================================================
    @property
    def domain(self) -> Domain:

        return self._domain
    # ==================================================================
    @property
    def center_omega(self) -> np.ndarray:

        return self._center_omega
    # ==================================================================
    @property
    def rep_freq(self) -> np.ndarray:

        return self._rep_freq
    # ==================================================================
    @property
    def delays(self) -> np.ndarray:

        return self._delays
    # ==================================================================
    @property
    def channels(self) -> np.ndarray:

        return self._channels
    # ==================================================================
    @property
    def samples(self) -> int:

        return self._samples
    # ==================================================================
    @property
    def nbr_channels(self) -> int:

        return self._nbr_channels
    # ==================================================================
    @property
    def noise(self) -> np.ndarray:

        return self._noise
    # ------------------------------------------------------------------
    @noise.setter
    def noise(self, noise: np.ndarray) -> None:
        if ((noise.ndim == 1) and (noise.shape[0] == self.noise_samples)):
            self._noise = noise
        else:

            raise ShapeError("The shape of the provided noise to field '{}' "
                "does not match with the number of noise samples in the "
                "domain, noise will not be set.".format(self.name))
    # ==================================================================
    @property
    def noise_samples(self) -> np.ndarray:

        return self.domain.noise_samples
    # ==================================================================
    # Methods ==========================================================
    # ==================================================================
    def add_delay(self, delays: np.ndarray) -> None:

        self._delays += delays
    # ==================================================================
    def extend(self, field: Field, allow_duplicate: bool = False) -> None:
        """Extend the current field with the channels of the provided
        field.

        Parameters
        ----------
        field : optcom.field.Field
            The field from which to take the channels to extend.

        """
        if (self.has_equal_charact(field)):
            for j in range(len(field)):
                self.add_channel(field[j], field.center_omega[j],
                                 field.rep_freq[j], field.delays[j])
            self._noise += field.noise
        else:
            warning_message: str = ("Two fields must have same number of "
                "samples, type and equivalent domain to be extended."
                "Operation aborted.")
            warnings.warn(warning_message, CompatibilityWarning)
    # ==================================================================
    def add_channel(self, channel: np.ndarray, center_omega: float,
                    rep_freq: float, delay: float = 0.0):
        """Aadd a channel to the field.  The channel must comply with
        the field characteristics.
        """
        success: bool = True
        if (channel.shape[0] == self.samples):
            if (self._channels.size):
                self._channels = np.vstack((self._channels,
                                            channel.astype(cst.NPFT)))
            else:
                self._channels = np.array([channel.astype(cst.NPFT)])
        else:
            warning_message: str = ("All channels must have the same "
                " dimension, can not add new channel.")
            warnings.warn(warning_message, DimWarning)
            success = False
        if (success):
            self._center_omega = np.append(self._center_omega, center_omega)
            if (self._delays.size):
                self._delays = np.append(self._delays, delay)
            else:
                self._delays = np.array([delay])
            if (self._rep_freq.size):
                self._rep_freq = np.append(self._rep_freq, rep_freq)
            else:
                self._rep_freq = np.array([rep_freq])
            self._nbr_channels += 1
    # ==================================================================
    def reset_channels(self, channel_nbr: int = None) -> None:
        """Reset (set to 0) channel_nbr, all if channel_nbr is None.

        Parameters
        ----------
        channel_nbr :
            The channel number to consider. If None, reset all channels.

        """
        if (channel_nbr is not None):
            if (channel_nbr < self._channels.shape[0]):
                self._channels[channel_nbr] =\
                    np.zeros(self._channels[channel_nbr].shape, dtype=cst.NPFT)
            else:
                warning_message: str = ("The channel number {} requested to "
                    "be reset does not exist".format(channel_nbr))
                warnings.warn(warning_message, IndexWarning)
        else:
            for i in range(self._channels.shape[0]):
                self._channels[i] =\
                    np.zeros(self._channels[i].shape, dtype=cst.NPFT)
    # ==================================================================
    def reset_delays(self, channel_nbr: int = None) -> None:
        """Reset (set to 0) channel_nbr delay, all if channel_nbr is
        None.

        Parameters
        ----------
        channel_nbr :
            The channel number to consider. If None, reset all channels.

        """
        if (channel_nbr is not None):
            if (channel_nbr < self._channels.shape[0]):
                self._delays[channel_nbr] = 0.0
            else:
                warning_message: str = ("The channel number {} requested to "
                    "reset delays does not exist".format(channel_nbr))
                warnings.warn(warning_message, IndexWarning)
        else:
            for i in range(self._channels.shape[0]):
                self._delays[i] = 0.0
    # ==================================================================
    def reset_noise(self) -> None:
        """Reset (set to 0) the noise figure."""
        self._noise = np.zeros(self.noise_samples, dtype=float)
    # ==================================================================
    def get_copy(self, new_name: Optional[str] = None,
                 reset_channels: bool = False, reset_noise: bool = False,
                 reset_delays: bool = False) -> Field:
        """Return a copy of self.

        Parameters
        ----------
        new_name :
            The name of the copied field.
        reset_channels :
            If True, reset all channels.
        reset_noise :
            If True, reset the noise.
        reset_delays :
            If True ,reset all delays.

        """
        self_copy = copy.deepcopy(self)
        if (new_name is None):
            self_copy.name = 'copy_of_' + self.name
        else:
            self_copy.name = new_name
        if (reset_channels):
            self_copy.reset_channels()
        if (reset_noise):
            self_copy.reset_noise()
        if (reset_delays):
            self_copy.reset_delays()

        return self_copy
    # ==================================================================
    # Static and class methods =========================================
    # ==================================================================
    # Field methods ----------------------------------------------------
    @staticmethod
    def sq_mod(A: np.ndarray) -> np.ndarray:

        return A * np.conj(A)
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def temporal_phase(A: List[np.ndarray]) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_phase(A: List[np.ndarray], unwrap: bool
                       ) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_phase(A: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_phase(A: np.ndarray, unwrap: bool) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def temporal_phase(A, unwrap=False):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        unwrap :
            If True, unwrap the phase array (see numpy.unwrap).

        Returns
        -------
        :
            The temporal phase of the pulse.

        Notes
        -----

        .. math::  \phi(t) = Arg\big(A(t)\big)


        """
        if (isinstance(A, List)):
            res: List[np.ndarray] = []
            for i in range(len(A)):
                res.append(Field.temporal_phase(A[i], unwrap))

            return res
        else:
            phase: np.ndarray = np.zeros(A.shape)
            if (A.ndim > 1):
                for i in range(A.shape[0]):
                    phase[i] = np.angle(A[i])
                    if (unwrap):
                        phase[i] = np.unwrap(phase[i])
            else:
                phase = np.angle(A)
                if (unwrap):
                    phase = np.unwrap(phase[i])

            return phase
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def spectral_phase(A: List[np.ndarray]) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_phase(A: List[np.ndarray], unwrap: bool
                       ) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_phase(A: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_phase(A: np.ndarray, unwrap: bool) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def spectral_phase(A, unwrap=False):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        unwrap :
            If True, unwrap the phase array (see numpy.unwrap).

        Returns
        -------
        :
            The spectral phase of the pulse.

        Notes
        -----

        .. math::  \phi(\lambda) = Arg\big(\mathcal{F}\{A(t)\}\big)


        """
        if (isinstance(A, List)):
            res: List[np.ndarray] = []
            for i in range(len(A)):
                res.append(Field.spectral_phase(A[i], unwrap))

            return res
        else:
            phase: np.ndarray = np.zeros(A.shape)
            if (A.ndim > 1):
                for i in range(A.shape[0]):
                    phase[i] = np.angle(FFT.fft(A[i]))
                    if (unwrap):
                        phase[i] = np.unwrap(phase[i])
                    phase[i] = FFT.ifftshift(phase[i])
            else:
                phase = np.angle(FFT.fft(A))
                if (unwrap):
                    phase = np.unwrap(phase[i])
                phase = FFT.ifftshift(phase)

            return phase
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def temporal_power(A: List[np.ndarray]) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_power(A: List[np.ndarray], normalize: bool
                       ) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_power(A: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_power(A: np.ndarray, normalize: bool) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def temporal_power(A, normalize=False):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        normalize :
            If True, normalize the power array.

        Returns
        -------
        :
            The temporal power of the pulse. :math:`[W]`

        Notes
        -----

        .. math::  P^t(t) = |A(t)|^2


        """
        if (isinstance(A, List)):
            res: List[np.ndarray] = []
            for i in range(len(A)):
                res.append(Field.temporal_power(A[i], normalize))

            return res
        else:
            P: np.ndarray = np.zeros(A.shape)
            if (A.ndim > 1):
                for i in range(A.shape[0]):
                    P[i]  = np.real(Field.sq_mod(A[i]))
                    if (normalize):
                        P[i] /= np.amax(P[i])
            else:
                P = np.real(Field.sq_mod(A))
                if (normalize):
                    P /= np.amax(P)

            return P
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def spectral_power(A: List[np.ndarray]) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_power(A: List[np.ndarray], normalize: bool
                       ) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_power(A: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_power(A: np.ndarray, normalize: bool
                       ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def spectral_power(A, normalize=False):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        normalize :
            If True, normalize the power array.

        Returns
        -------
        :
            The spectral power density of the pulse. :math:`[a.u.]`

        Notes
        -----

        .. math::  P^\lambda(\lambda) = |\mathcal{F}\{A(t)\}|^2


        """
        if (isinstance(A, List)):
            res: List[np.ndarray] = []
            for i in range(len(A)):
                res.append(Field.spectral_power(A[i], normalize))

            return res
        else:
            P: np.ndarray = np.zeros(A.shape)
            if (A.ndim > 1):    # multidimensional
                for i in range(A.shape[0]):
                    P[i] = np.real(Field.sq_mod(FFT.fft(A[i])))
                    if (normalize):
                        P[i] /= np.amax(P[i])
                    P[i] = FFT.ifftshift(P[i])
            else:
                P = np.real(Field.sq_mod(FFT.fft(A))) # np.real to remove 0j
                if (normalize):
                    P /= np.amax(P)
                P = FFT.ifftshift(P)

            return np.real(P)  # np.real to remove 0j
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def energy(A: List[np.ndarray], dpos: float
               ) -> Union[List[np.ndarray], List[float]]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def energy(A: np.ndarray, dpos: float) -> Union[np.ndarray, float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def energy(A, dpos):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        dpos :
            The step interval. :math:`[ps]` if dtime, :math:`[ps^{-1}]`
            if omega.

        Returns
        -------
        :
            The energy of the pulse. :math:`[J]`

        Notes
        -----

        Equations:

        .. math::  E_{pulse} = \int_0^T P(t)\, dt

        Implementation:

        .. math::  E(A) = \Delta t\sum_{t=0}^{N_t -1}P^t(A(t))

        where :math:`\Delta t` is the time represented by one time
        sample, :math:`N_t` is the number of time samples and
        :math:`P^t` depicts the temporal power.


        """
        dpos_ = dpos * 1e-12  # ps -> s
        if (isinstance(A, List)):
            res: List[Union[float, np.ndarray]] = []
            for i in range(len(A)):
                res.append(Field.energy(A[i], dpos))

            return res
        else:
            E: Union[float, np.ndarray]
            if (A.ndim > 1):    # multidimensional
                E = np.sum(Field.temporal_power(A, False), axis=1) * dpos_
            else:
                E = np.sum(Field.temporal_power(A, False)) * dpos_

            return E
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def temporal_peak_power(A: List[np.ndarray]
                            ) -> Union[List[np.ndarray], List[float]]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def temporal_peak_power(A: np.ndarray) -> Union[np.ndarray, float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def temporal_peak_power(A):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.

        Returns
        -------
        :
            The temporal peak power of the pulse. :math:`[W]`

        Notes
        -----

        .. math::  P_{peak} = \underset{t}{\max} \quad P^t(A(t))

        where :math:`P^t` depicts the temporal power.


        """
        if (isinstance(A, List)):
            res: List[Union[float, np.ndarray]] = []
            for i in range(len(A)):
                res.append(Field.temporal_peak_power(A[i]))

            return res
        else:
            P_peak: Union[float, np.ndarray]
            if (A.ndim > 1):    # multidimensional
                P_peak = np.max(Field.temporal_power(A, False), axis=1)
            else:
                P_peak = np.max(Field.temporal_power(A, False))

            return P_peak
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def spectral_peak_power(A: List[np.ndarray]
                            ) -> Union[List[np.ndarray], List[float]]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def spectral_peak_power(A: np.ndarray) -> Union[np.ndarray, float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def spectral_peak_power(A):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.

        Returns
        -------
        :
            The spectral peak power of the pulse. :math:`[a.u.]`

        Notes
        -----

        .. math::  P_{peak} = \underset{\lambda}{\max} \quad
                              P^\lambda(A(t))

        where :math:`P^\lambda` depicts the spectral power.


        """
        if (isinstance(A, List)):
            res: List[Union[float, np.ndarray]] = []
            for i in range(len(A)):
                res.append(Field.spectral_peak_power(A[i]))

            return res
        else:
            P_peak: Union[float, np.ndarray]
            if (A.ndim > 1):    # multidimensional
                P_peak = np.max(Field.spectral_power(A, False), axis=1)
            else:
                P_peak = np.max(Field.spectral_power(A, False))

            return P_peak
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def average_power(A: np.ndarray, dpos: float, rep_freq: float
                      ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def average_power(A: List[np.ndarray], dpos: float, rep_freq: List[float]
                      ) -> List[float]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def average_power(A: List[np.ndarray], dpos: float,
                      rep_freq: List[np.ndarray]
                      ) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def average_power(A: np.ndarray, dpos: float, rep_freq: np.ndarray
                      ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def average_power(A, dpos, rep_freq):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        dpos :
            The step interval. :math:`[ps]` if dtime, :math:`[ps^{-1}]`
            if omega.
        rep_freq :
            The repetition frequency. :math:`[THz]`


        Returns
        -------
        :
            The average power of the pulse. :math:`[W]`

        Notes
        -----

        Equations:

        .. math::  P_{avg} = \frac{1}{T} \int_0^T P(t)\, dt
                           = \frac{E_{pulse}}{T} = E_{pulse} f_{rep}

        where :math:`E_{pulse}` is the energy of the pulse, :math:`T`
        is the pulse period and :math:`f_{rep}` is the repetition
        frequency.

        Implementation:

        If :math:`f_{rep}=0`: (i.e. no repetition frequency specified
        by the user)

        .. math::  P_{avg} = 0

        If :math:`f_{rep}=Nan`: (i.e. it is a CW pulse)

        .. math::  P_{avg} = \frac{1}{N_t}\sum_{t=0}^{N_t -1} P^t(A(t))

        where :math:`N_t` is the number of time samples and :math:`P^t`
        depicts the temporal power.

        If :math:`f_{rep} > 0`: (i.e. no repetition frequency specified
        by the user)

        .. math::  P_{avg} = E(A) f_{rep}
                           = f_{rep}\Delta t\sum_{t=0}^{N_t -1}P^t(A(t))

        where :math:`\Delta t` is the time represented by one time
        sample, :math:`N_t` is the number of time samples and
        :math:`P^t` depicts the temporal power.

        """
        if (isinstance(A, List)):
            res: List[Union[float, np.ndarray]] = []
            if (not isinstance(rep_freq, List) or len(rep_freq) != len(A)):
                warning_message: str = ("The length of the list of pulses and "
                    "repetion frequencies must be equal in order to "
                    "calculate the average power, return zeros.")
                warnings.warn(warning_message, DimWarning)

                return [0 for i in range(len(A))]
            for i in range(len(A)):
                res.append(Field.average_power(A[i], dpos, rep_freq[i]))

            return res
        else:
            P_avg: Union[float, np.ndarray]
            rep_freq_ = rep_freq * 1e12     # THz -> Hz
            if (A.ndim > 1):    # multidimensional
                P_avg = np.zeros(A.shape[0])
                for i in range(A.shape[0]):
                    if (np.isnan(rep_freq_[i])):
                        P_avg[i] = np.mean(Field.temporal_power(A, False)[i])
                    elif (not rep_freq_[i]):
                        P_avg[i] = np.zeros(A.shape[0])
                    else:
                        P_avg[i] = Field.energy(A[i], dpos) * rep_freq_[i]
            else:
                if (np.isnan(rep_freq)):
                    P_avg = np.mean(Field.temporal_power(A, False))
                elif (not rep_freq):
                    P_avg = 0.0
                else:
                    P_avg = Field.energy(A, dpos) * rep_freq_

            return P_avg
    # ==================================================================
    # N.B.: Can not use np.ndarray in overload bcst List[np.ndarray]
    #       and np.ndarray both eval to Any
    @overload
    @staticmethod
    def fwhm(A: List[np.ndarray], dpos: float, peak_position: Optional[int]
             ) -> List[np.ndarray]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def fwhm(A: np.ndarray, dpos: float, peak_position: Optional[int]
             ) -> np.ndarray: ...
    @staticmethod
    def fwhm(A, dpos, peak_position=None):
        r"""
        Parameters
        ----------
        A :
            Pulse field envelope.
        dpos :
            The step interval. :math:`[ps]` if dtime, :math:`[ps^{-1}]`
            if omega.
        peak_position :
            The position of the peak if not centered.


        Returns
        -------
        :
            The average power of the pulse. :math:`[W]`

        """
        if (isinstance(A, List)):
            res = []
            for i in range(len(A)):
                res.append(Field.fwhm(A[i], dpos, peak_position))

            return res
        else:
            peak_pos: int
            fwhm: np.ndarray = np.array([])
            if (A.ndim > 1):    # multidimensional
                fwhm = np.zeros(len(A))
            else:
                fwhm = np.zeros(1)
            for i in range(len(fwhm)):
                A_ = A[i] if (A.ndim > 1) else A
                if (peak_position is None):
                    peak_pos = np.argmax(A_)
                else:
                    peak_pos = peak_position
                peak_val = A_[peak_pos]
                half_val = peak_val / 2.
                ind_left = peak_pos
                ind_right = peak_pos
                while (ind_left and (A_[ind_left] > half_val)):
                    ind_left -= 1
                while ((ind_right < len(A_)-1) and (A_[ind_right] > half_val)):
                    ind_right += 1
                diff_left = A_[ind_left+1] - A_[ind_left]
                diff_right = A_[ind_right] - A_[ind_right-1]
                # Linear interpolation
                pt_left = ind_left + ((half_val-A_[ind_left]) / diff_left)
                pt_right = ind_right + ((half_val-A_[ind_right]) / diff_right)

                fwhm[i] = (pt_right - pt_left) * dpos

            return fwhm
