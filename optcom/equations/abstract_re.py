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
