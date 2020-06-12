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

import math
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter


class SEPower(AbstractParameter):

    def __init__(self, domega: float) -> None:
        r"""
        Parameters
        ----------
        domega :
            The angular frequency step. :math:`[rad\cdot ps^{-1}]`

        """
        self._domega: float = domega
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the energy saturation.

        Parameters
        ----------
        domega :
            The angular frequency step. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the spontaneous emission power. :math:`[W]`

        """

        return SEPower.calc_se_power(omega, self._domega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_se_power(omega: float, domega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_se_power(omega: np.ndarray, domega: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_se_power(omega, domega):
        r"""Calculate the spontaneous emission power.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        domega :
            The angular frequency step. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the spontaneous emission power. :math:`[W]`

        Notes
        -----

        .. math:: P_{SE} = \frac{h\omega \Delta\omega}{2\pi^{2}}

        """
        res = cst.H * omega * domega / (2.0 * cst.PI**2)
        res *= 1e18 # nm^2 kg ps^{-3} -> m^2 kg s^{-3} = W

        return res



if __name__ == "__main__":
    """Plot the overlap factor as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    domega:float = 0.02
    # With float
    omega: float = oc.lambda_to_omega(1552.0)
    se_power: oc.SEPower = oc.SEPower(domega)
    print(se_power(omega))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(500., 1600., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    res: np.ndarray = se_power(omegas)

    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = [r'SE Power $\,P_{SE}\,(W)$']
    plot_titles: List[str] = ["Spontaneous emission power as a function of "
                              "the wavelength."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0])
