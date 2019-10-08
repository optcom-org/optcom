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

import math
from typing import List, Optional, overload

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter


class NumericalAperture(AbstractParameter):

    def __init__(self):

        return None
    # ==================================================================
    @overload
    def __call__(self, n_core: float, n_clad: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, n_core: Array[float], n_clad: Array[float]
                 ) -> Array[float]: ...
    # ------------------------------------------------------------------
    def __call__(self, n_core, n_clad) -> float:
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        n_core :
            The refractive index of the core.
        n_clad :
            The refractive index of the cladding.

        Returns
        -------
        :
            The numerical aperture.

        """

        return NumericalAperture.calc_NA(n_core, n_clad)
    # ==================================================================
    # Static methods ===================================================
    # ==================================================================
    @overload
    @staticmethod
    def calc_NA(n_core: float, n_clad: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_NA(n_core: Array[float], n_clad: Array[float]
                ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_NA(n_core, n_clad):
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        n_core :
            The refractive index of the core.
        n_clad :
            The refractive index of the cladding.

        Returns
        -------
        :
            The numerical aperture.

        Notes
        -----

        .. math::  NA = \sqrt{n_{co}^2 - n_{cl}^2}

        """
        if (isinstance(n_core, float)):

            return math.sqrt(n_core**2 - n_clad**2)
        else:

            return np.sqrt(np.square(n_core) - np.square(n_clad))
    # ==================================================================
    @overload
    @staticmethod
    def calc_n_core(NA: float, n_clad: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_core(NA: Array[float], n_clad: Array[float]
                    ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_n_core(NA, n_clad):
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        NA :
            The numerical aperture.
        n_clad :
            The refractive index of the cladding.

        Returns
        -------
        :
            The refractive index of the core.

        Notes
        -----

        .. math::  n_{co} = \sqrt{NA^2 + n_{cl}^2}

        """
        if (isinstance(n_clad, float)):

            return math.sqrt(NA**2 + n_clad**2)
        else:

            return np.sqrt(np.square(NA) + np.square(n_clad))
    # ==================================================================
    @overload
    @staticmethod
    def calc_n_clad(NA: float, n_core: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_n_clad(NA: Array[float], n_core: Array[float]
                    ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_n_clad(NA, n_core):
        r"""Calculate the numerical aperture.

        Parameters
        ----------
        NA :
            The numerical aperture.
        n_core :
            The refractive index of the core.

        Returns
        -------
        :
            The refractive index of the cladding.

        Notes
        -----

        .. math::  n_{cl} = \sqrt{n_{co}^2 - NA^2}

        """
        if (isinstance(n_core, float)):

            return math.sqrt(n_core**2 - NA**2)
        else:

            return np.sqrt(np.square(n_core) - np.square(NA))

if __name__ == "__main__":

    import numpy as np
    from optcom.domain import Domain

    # With float
    n_core = 1.43
    n_clad = 1.425

    NA_inst = NumericalAperture()
    NA = NumericalAperture.calc_NA(n_core, n_clad)
    print(NA, NA_inst(n_core, n_clad))
    print(n_core, NumericalAperture.calc_n_core(NA, n_clad))
    print(n_clad, NumericalAperture.calc_n_clad(NA, n_core))

    # With numpy ndarray
    n_core = np.linspace(1.42, 1.43, 10)
    n_clad = np.linspace(1.415, 1.425, 10)

    NA_inst = NumericalAperture()
    NA = NumericalAperture.calc_NA(n_core, n_clad)
    print(NA, NA_inst(n_core, n_clad))
    print(n_core, NumericalAperture.calc_n_core(NA, n_clad))
    print(n_clad, NumericalAperture.calc_n_clad(NA, n_core))
