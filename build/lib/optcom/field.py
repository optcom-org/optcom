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

import copy
import operator
from typing import Any, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.utils.fft import FFT


class IOperator():

    def __init__(self):

        return None

    def __call__(self, ioperator):

        def wrap_ioperator(obj, operand):

            iop = ioperator(obj, operand)
            if (isinstance(operand, Field)):
                if (obj.type == operand.type):
                    if (len(obj) == len(operand)):
                        iop(obj[:], operand[:])
                    else:
                        util.warning_terminal("Two fields must have the same "
                            "number of channels to support global operators.")
                else:
                    util.warning_terminal("Two fields must be of the same "
                        "type to support global operators.")
            else:
                iop(obj[:], operand)

            return obj

        return wrap_ioperator


class Operator():

    def __init__(self):

        return None

    def __call__(self, operator):

        def wrap_operator(obj, operand):

            iop = operator(obj, operand)
            field = copy.deepcopy(obj)
            iop(field, operand)

            return field

        return wrap_operator

class ROperator():

    def __init__(self):

        return None

    def __call__(self, roperator):

        def wrap_roperator(obj, operand):

            # Operand can only be an int, float, complex, ...  Can not
            # be Field, otherwise operator would have been called bcs
            # it is left operand
            # BUT can be np number for example! -> bug found
            # with numpy dtype object complex as left operand, then
            # return a numpy array and not a field object
            rop = roperator(obj, operand)
            field = copy.deepcopy(obj)
            field[:] = rop(operand, obj[:])

            return field

        return wrap_roperator


_ioperator = IOperator()
_operator = Operator()
_roperator = ROperator()


class Field(object):
    """Represent a field made of several channels.

    Attributes
    ----------
    type : str
        The type of the fields.
        See :mod:`optcom/utils/constant_values/field_types`.

    """

    def __init__(self, domain: Domain, type: str) -> None:
        """
        Parameters
        ----------
        domain : Domain
            The domain which the field is bound to.
        type :
            The type of the fields.
            See :mod:`optcom/utils/constant_values/field_types`.

        """
        self._domain: Domain = domain
        self._samples: int = domain.samples
        self.type: str = type
        self._nbr_channels: int = 0
        self._channels: Array[cst.NPFT, self._nbr_channels, self._samples] =\
            np.array([], dtype=cst.NPFT)
        self._storage: Array[cst.NPFT, ..., self._nbr_channels,
                              self._samples] = np.array([], dtype=cst.NPFT)
        self._center_omega: Array[float, 1, self._nbr_channels] =\
            np.array([], dtype=float)
        self._delay_time: Array[float, 1, self._nbr_channels] =\
            np.array([], dtype=float)
    # ==================================================================
    def __getitem__(self, key: Union[int, slice]) -> Array[cst.NPFT, 1,...]:

        return self._channels[key]
    # ==================================================================
    def __setitem__(self, key: int, channel: Array[cst.NPFT, 1,...]) -> None:

        if (len(channel) == len(self._channels[key])):
            self._channels[key] = channel.astype(cst.NPFT)
        else:
            util.warning_terminal("All channels must have the same dimension, "
                "can not change channel.")
    # ==================================================================
    def __delitem__(self, key: int) -> None:

        self._channels = np.delete(self._channels, key, axis=0)
        self._center_omega = np.delete(self._center_omega, key, axis=0)
        self._delay_time = np.delete(self._delay_time, key, axis=0)
        self._nbr_channels -= 1
    # ==================================================================
    def __len__(self) -> int:

        return self._channels.shape[0]
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

        return operator.__iadd__
    # ==================================================================
    @_operator
    def __sub__(self, operand):

        return operator.__isub__
    # ==================================================================
    @_operator
    def __mul__(self, operand):

        return operator.__imul__
    # ==================================================================
    @_operator
    def __truediv__(self, operand):

        return operator.__itruediv__
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
    # Getters - setters - deleter ======================================
    # ==================================================================
    @property
    def time(self) -> Array[float]:
        time = np.zeros((self._nbr_channels, self._samples))
        if (self._nbr_channels):
            for i in range(len(self._delay_time)):
                time[i] = (self._domain.time+self._delay_time[i])

        return time
    # ==================================================================
    @property
    def omega(self) -> Array[float]:
        omega = np.zeros((self._nbr_channels, self._samples))
        if (self._nbr_channels):
            for i in range(len(self._center_omega)):
                omega[i] = (self._domain.omega+self._center_omega[i])

        return omega
    # ==================================================================
    @property
    def nu(self) -> Array[float]:
        nu = np.zeros((self._nbr_channels, self._samples))
        if (self._nbr_channels):
            omega = self.omega
            for i in range(omega.shape[0]):
                nu[i] = Domain.omega_to_nu(omega[i])

        return nu
    # ==================================================================
    @property
    def center_omega(self):

        return self._center_omega
    # ==================================================================
    @property
    def delay_time(self):

        return self._delay_time
    # ==================================================================
    @property
    def channels(self):

        return self._channels
    # ==================================================================
    @property
    def samples(self):

        return self._samples
    # ==================================================================
    @property
    def nbr_channels(self):

        return self._nbr_channels
    # ==================================================================
    @property
    def storage(self):

        return self._storage
    # ==================================================================
    @storage.setter
    def storage(self, storage):
        self._storage = storage
    # ==================================================================
    # Methods ==========================================================
    # ==================================================================
    def delay(self, delay_time: Array[float, 1,...]) -> None:
        self._delay_time += delay_time
    # ==================================================================
    def extend(self, field: Field) -> None:
        if (self._samples == field.samples):
            if (self.type == field.type):
                self._nbr_channels += field.nbr_channels
                self._channels = np.vstack((self._channels, field.channels))
                self._center_omega = np.hstack((self._center_omega,
                                                field.center_omega))
                self._delay_time = np.hstack((self._delay_time,
                                              field.delay_time))
                self._storage = np.hstack((self._storage, field.storage))
            else:
                util.warning_terminal("Two fields of different types can "
                    "not be extended to each other")
        else:
            util.warning_terminal("Two fields with different number of "
                "samples can not be extended to each other.")
    # ==================================================================
    def add(self, field: Field) -> None:
        if (self._samples == field.samples):
            if (self.type == field.type):
                field_comega = field.center_omega
                for j in range(len(field)):
                    if (field_comega[j] in self.center_omega):
                        index = np.where(self.center_omega==field_comega[j]
                                         )[0][0]
                        self._channels[index] += field[j]
                    else:
                        self.append(field[j], field_comega[j])
            else:
                util.warning_terminal("Two fields of different types can "
                    "not be added to each other")
        else:
            util.warning_terminal("Two fields with different number of "
                "samples can not be added to each other.")
    # ==================================================================
    def append(self, channel: Array[cst.NPFT, 1, ...],
               center_omega: float, delay_time: float = 0.0):

        success = True
        if (self._channels.size):
            if (len(channel) == self._channels.shape[1]):
                self._channels = np.vstack((self._channels,
                                            channel.astype(cst.NPFT)))
            else:
                util.warning_terminal("All channels must have the same "
                    " dimension, can not add new channel.")
                success = False
        else:
            self._channels = np.array([channel.astype(cst.NPFT)])
        if (success):
            self._center_omega = np.append(self._center_omega, center_omega)
            self._delay_time = np.append(self._delay_time, delay_time)
            self._nbr_channels += 1
    # ==================================================================
    def reset_channel(self, channel_nbr: int = None) -> None:
        """Reset (set to 0) channel_nbr, all if channel_nbr is None"""

        if (channel_nbr is not None):
            if (channel_nbr < self._channels.shape[0]):
                self._channels[channel_nbr] =\
                    np.zeros(self._channels[channel_nbr].shape, dtype=cst.NPFT)
                self._delay_time[channel_nbr] = 0.0
            else:
                util.warning_terminal("The channel number {} requested to be "
                    "reset does not exist".format(channel_nbr))
        else:
            for i in range(self._channels.shape[0]):
                self._channels[i] =\
                    np.zeros(self._channels[i].shape, dtype=cst.NPFT)
                self._delay_time[i] = 0.0
    # ==================================================================
    # Static and class methods =========================================
    # ==================================================================
    # Field methods ----------------------------------------------------
    @staticmethod
    def sq_mod(A: Array) -> Array:

        return A * np.conj(A)
    # ==================================================================
    @staticmethod
    def phase(A: Array) -> Array:

        if (A is not None):
            if (A.ndim > 1):
                phase = np.zeros(A.shape)
                for i in range(A.shape[0]):
                    phase[i] = np.angle(A[i])
            else:
                phase = np.angle(A)

            return phase

        return None
    # ==================================================================
    @staticmethod
    def temporal_power(A: Array, normalize: bool = False) -> Array:

        if (A is not None):
            if (A.ndim > 1):
                P = np.zeros(A.shape)
                for i in range(A.shape[0]):
                    P[i]  = np.real(Field.sq_mod(A[i]))
                    if (normalize):
                        P[i] /= np.amax(P[i])
            else:
                P = np.real(Field.sq_mod(A))
                if (normalize):
                    P /= np.amax(P)

            return P
        else:
            util.warning_terminal("Can not get temporal power of a "
                "nonexistent field, request ignored, return null field")

        return None
    # ==================================================================
    @staticmethod
    def spectral_power(A: Array, normalize: bool = False):

        if (A is not None):
            if (A.ndim > 1):    # multidimensional
                P = np.zeros(A.shape)
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
        else:
            util.warning_terminal("Can not get spectral power of a "
                "nonexistent field, request ignored, return null field")

        return None


class EmptyField(Field):
    """Dummy field object to represent an empty field. Rewrite all
    getters from Field and return None. Rewrite all operations
    overloading and set emptyfield as the neutral elements for all.
    """

    def __init__(self) -> None:

        return None
    # ==================================================================
    def __getitem__(self, key):

        return None
    # ==================================================================
    def __setitem__(self, key, channel):

        return None
    # ==================================================================
    def __delitem__(self, key):

        return None
    # ==================================================================
    def __iadd__(self, operand):

        return operand
    # ==================================================================
    def __isub__(self, operand):

        return operand
    # ==================================================================
    def __imul__(self, operand):

        return operand
    # ==================================================================
    def __itruediv__(self, operand):

        return operand
    # ==================================================================
    def __add__(self, operand):

        return operand
    # ==================================================================
    def __sub__(self, operand):

        return operand
    # ==================================================================
    def __mul__(self, operand):

        return operand
    # ==================================================================
    def __truediv__(self, operand):

        return operand
    # ==================================================================
    def __radd__(self, operand):

        return operand
    # ==================================================================
    def __rsub__(self, operand):

        return operand
    # ==================================================================
    def __rmul__(self, operand):

        return operand
    # ==================================================================
    def __rtruediv__(self, operand):

        return operand
    # ==================================================================
    @property
    def time(self):

        return None
    # ==================================================================
    @property
    def omega(self):

        return None
    # ==================================================================
    @property
    def nu(self):

        return None
    # ==================================================================
    @property
    def center_omega(self):

        return None
    # ==================================================================
    @property
    def delay_time(self):

        return None
    # ==================================================================
    @property
    def channels(self):

        return None
    # ==================================================================
    @property
    def samples(self):

        return None
    # ==================================================================
    @property
    def nbr_channels(self):

        return None
    # ==================================================================
    @property
    def storage(self):

        return None
    # ==================================================================
    @storage.setter
    def storage(self, storage):

        return None
