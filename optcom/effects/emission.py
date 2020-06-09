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

from typing import Callable, Union

from optcom.effects.active_fiber_photon_process import ActiveFiberPhotonProcess


class Emission(ActiveFiberPhotonProcess):
    r"""The emission effect.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    domega : float
        The angular frequency step. :math:`[ps^{-1}]`
    dtime : float
        The time step. :math:`[ps]`

    """

    def __init__(self, sigma: Union[float, Callable],
                 Gamma: Union[float, Callable], doped_area: float,
                 UNI_OMEGA: bool = True) -> None:
        """
        Parameters
        ----------
        sigma :
            The absorption cross sections. :math:`[nm^2]`  If a callable
            is prodived, variable must be wavelength. :math:`[nm]`
        Gamma :
            The overlap factor. If a callable is provided, variable must
            be angular frequency. :math:`[ps^{-1}]`
        doped_area :
            The doped area. :math:`[\mu m^2]`
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.

        """
        super().__init__(sigma, Gamma, doped_area, UNI_OMEGA)
