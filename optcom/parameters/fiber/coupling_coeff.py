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
from optcom.domain import Domain
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


class CouplingCoeff(AbstractParameter):

    def __init__(self, v_nbr: Union[float, Callable], a: float, d: float,
                 ref_index: Union[float, Callable]) -> None:
        r"""
        Parameters
        ----------
        v_nbr :
            The V number.  If a callable is provided, variable must be
            angular frequency. :math:`[ps^{-1}]`
        a :
            The core radius. :math:`[\mu m]`
        d :
            The center to center spacing between the two cores.
            :math:`[\mu m]`
        ref_index :
            The refractive index outside of the two fiber cores.  If a
            callable is provided, variable must be angular frequency.
            :math:`[ps^{-1}]`

        """
        self._v_nbr: Union[float, Callable] = v_nbr
        self._a: float = a
        self._d: float = d
        self._ref_index: Union[float, Callable] = ref_index
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the coupling coefficient.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the coupling coefficient. :math:`[km^{-1}]`

        """
        fct = CallableContainer(CouplingCoeff.calc_kappa,
                                [omega, self._v_nbr, self._a, self._d,
                                 self._ref_index])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_kappa(omega: float, v_nbr: float, a: float, d: float,
                   ref_index: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_kappa(omega: np.ndarray, v_nbr: np.ndarray, a: float, d: float,
                   ref_index: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_kappa(omega, v_nbr, a, d, ref_index):
        r"""Calculate the coupling coefficient for the parameters given
        for two waveguides. (assuming the two waveguides are
        symmetric) [3]_

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        v_nbr :
            The fiber parameter.
        a :
            The core radius. :math:`[\mu m]`
        d :
            The center to center spacing between the two cores.
            :math:`[\mu m]`
        ref_index :
            The refractive index outside of the two fiber cores.

        Returns
        -------
        :
            Value of the coupling coefficient. :math:`[km^{-1}]`

        Notes
        -----

        .. math:: \kappa = \frac{\pi V}{2 k_0 n_0 a^2}
                            e^{-(c_0 + c_1 d + c_2 d^2)}

        This equation is accurate to within 1% for values of the fiber
        parameter :math:`1.5\leq V\leq 2.5` and of the normalized
        center-to-center spacing :math:`\bar{d} = d/a`,
        :math:`2\leq \bar{d}\leq 4.5` .

        References
        ----------
        .. [3] Govind Agrawal, Chapter 2: Fibers Couplers,
           Applications of Nonlinear Fiber Optics (Second Edition),
           Academic Press, 2008, Page 59.

        """
        lambda_0 = Domain.omega_to_lambda(omega)
        a *= 1e-9   # um -> km
        d *= 1e-9   # um -> km
        lambda_0 *= 1e-12   # nm -> km
        norm_d = d/a
        k_0  = 2*cst.PI/lambda_0
        c_0 = 5.2789 - 3.663*v_nbr + 0.3841*v_nbr**2
        c_1 = -0.7769 + 1.2252*v_nbr - 0.0152*v_nbr**2
        c_2 = -0.0175 - 0.0064*v_nbr - 0.0009*v_nbr**2
        # Check validity of formula paramater --------------------------
        if (np.mean(v_nbr) < 1.5 or np.mean(v_nbr) > 2.5):
            util.warning_terminal("Value of the fiber parameter is V = {}, "
                "kappa fitting formula is valid only for : 1.5 <= V <= 2.5, "
                "might lead to unrealistic results.".format(np.mean(v_nbr)))
        if (norm_d < 2.0 or norm_d > 4.5):
            util.warning_terminal("Value of the normalized spacing is d = {}, "
                "kappa fitting formula is valid only for : 2.0 <= d <= 4.5, "
                "might lead to unrealistic results.".format(norm_d))
        # Formula ------------------------------------------------------
        if (isinstance(omega, float)):
            res = (cst.PI*v_nbr / (2*k_0*ref_index*a**2)
                   * math.exp(-(c_0 + c_1*norm_d + c_2*norm_d**2)))
        else:
            res = (cst.PI*v_nbr / (2*k_0*ref_index*a**2)
                   * np.exp(-(c_0 + c_1*norm_d + c_2*norm_d**2)))

        return res


if __name__ == "__main__":
    """Plot the coupling coefficient as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    v_nbr: float = 2.0
    a: float = 5.0
    d: float = 15.0
    ref_index: oc.Sellmeier = oc.Sellmeier(medium='sio2')
    norm_d: float = d/a
    cpl: oc.CouplingCoeff = oc.CouplingCoeff(v_nbr, a, d, ref_index)
    # With float
    Lambda: float = 1550.0
    omega: float = oc.lambda_to_omega(Lambda)
    print(cpl(omega))
    # With np.ndarray
    Lambdas: np.ndarray = np.linspace(900, 1600, 3)
    omegas: np.ndarray = oc.lambda_to_omega(Lambdas)
    print(cpl(omegas))
    Lambdas = np.linspace(900, 1600, 1000)
    omegas = oc.lambda_to_omega(Lambdas)
    kappas: np.ndarray = cpl(omegas)

    plot_titles: List[str] = ["Coupling coefficient as a function of the "
                              "wavelength for norm. spacing = {}"
                              .format(norm_d)]

    oc.plot2d(np.array([Lambdas]), np.array([kappas]), x_labels=['Lambda'],
              y_labels=[r'Kappa $km^{-1}$'], plot_titles=plot_titles,
              split=True, line_opacities=[0.0], y_ranges=[(200000., 420000.)])
