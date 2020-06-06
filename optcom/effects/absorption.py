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

from typing import Callable, Union

from optcom.effects.active_fiber_photon_process import ActiveFiberPhotonProcess


class Absorption(ActiveFiberPhotonProcess):
    r"""The absorption effect.

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
        r"""
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
