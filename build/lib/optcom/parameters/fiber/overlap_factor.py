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


class OverlapFactor(AbstractParameter):

    def __init__(self, doped_area: float) -> None:
        """
        Parameters
        ----------
        doped_area :
            The doped area. :math:`[\mu m^2]`

        """
        self._doped_area = doped_area
    # ==================================================================
    @overload
    def __call__(self, eff_area: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, eff_area: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def __call__(self, eff_area):
        r"""Calculate the overlap factor.

        Parameters
        ----------
        eff_area :
            The effective area. :math:`[\mu m^2]`

        Returns
        -------
        :
            Value of the overlap factor.

        """
        return OverlapFactor.calc_overlap_factor(self._doped_area, eff_area)
    # ==================================================================
    @overload
    @staticmethod
    def calc_overlap_factor(doped_area: float, eff_area: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_overlap_factor(doped_area: Array[float], eff_area: Array[float]
                            ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_overlap_factor(doped_area, eff_area):
        r"""Calculate the overlap factor.

        Parameters
        ----------
        doped_area :
            The doped area. :math:`[\mu m^2]`
        eff_area :
            The effective area. :math:`[\mu m^2]`

        Returns
        -------
        :
            Value of the overlap factor.

        Notes
        -----
        Considering:

        .. math:: \Gamma  = 1 - e^{-2\frac{A_{doped}}{A_{eff}}}

        """
        if (isinstance(eff_area, float)):
            res = 1 - math.exp(-2*doped_area/eff_area)
        else:
            res = 1 - np.exp(-2*doped_area/eff_area)

        return res


if __name__ == "__main__":

    import numpy as np
    import optcom.utils.constants as cst

    # With float
    A_doped = cst.PI*25.0
    A_eff = 50.0
    of = OverlapFactor(A_doped)
    print(of(A_eff))
    print(of.calc_overlap_factor(A_doped, A_eff))
    # With numpy array
    A_doped_ = cst.PI*25.0*np.ones(10)
    A_eff_ = np.linspace(45, 55, 10)
    of = OverlapFactor(A_doped_)
    print(of(A_eff_))
    print(of.calc_overlap_factor(A_doped_, A_eff_))
