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

from typing import List

from optcom.equations.abstract_field_equation import AbstractEquation


class AbstractRE(AbstractEquation):

    def __init__(self, nbr_eqs: int, N_T: float):
        r"""
        Parameters
        ----------
        nbr_eqs :
            The number of rate equations.
        N_T :
            The total doping concentration. :math:`[nm^{-3}]`

        """
        self._N_T = N_T
        self._nbr_levels = nbr_eqs
