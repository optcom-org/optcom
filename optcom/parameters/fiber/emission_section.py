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
from typing import Callable, List, Optional, overload, Union, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.mccumber import McCumber
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.parameters.fiber.absorption_section import AbsorptionSection
from optcom.utils.callable_container import CallableContainer


DOPANTS: List[str] = ["yb"]
MEDIA: List[str] = ["sio2"]


class EmissionSection(AbstractParameter):

    def __init__(self, dopant: str = cst.DEF_FIBER_DOPANT,
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 T: float = cst.TEMPERATURE,
                 sigma_a: Optional[Union[float, Callable]] = None) -> None:
        r"""
        Parameters
        ----------
        dopant :
            The doping medium.
        medium :
            The fiber medium.
        T :
            The absolute temperature. :math:`[K]`
        sigma_a :
            A callable which return the absorption cross section.
            :math:`[nm^2]` If a callable is provided, the parameters
            must be the angular frequencies. :math:`[ps^{-1}]` If none
            is provided, will construct a :class:`parameters.fiber.
            absorption_section.AbsorptionSection` object.

        """
        self._dopant: str = util.check_attr_value(dopant.lower(), DOPANTS,
                                                  cst.DEF_FIBER_DOPANT)
        self._dopant = dopant
        self._medium: str = util.check_attr_value(medium.lower(), MEDIA,
                                                  cst.DEF_FIBER_MEDIUM)
        self._T: float = T
        self._sigma_a: Union[float, Callable]
        if (sigma_a is None):
            self._sigma_a = AbsorptionSection(dopant)
        else:
            self._sigma_a = sigma_a
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the derivatives of the emission cross sections.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The nth derivative of the propagation constant.
            :math:`[ps^{i}\cdot km^{-1}]`

        """
        fct = CallableContainer(EmissionSection.calc_sigma,
                                [omega, self._sigma_a, self._T, self._medium,
                                 self._dopant])

        return fct(omega)
    # ==================================================================
    # N.B.: No analytical fct for emission cross sections.  Can use
    # however the McCumber relations to draw emission from absorption.
    @overload
    @staticmethod
    def calc_sigma(omega: float, sigma_a: float, T: float, medium: str,
                   dopant: str) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_sigma(omega: np.ndarray, sigma_a: np.ndarray, T: float,
                   medium: str, dopant: str) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_sigma(omega, sigma_a, T, medium, dopant):
        r"""Calculate the emission cross section. There is no analytical
        formula for the emission cross section. Calculate first the
        absorption cross sections with the fitting formula from ref.
        [5]_ and then use the McCumber relations to deduce the emission
        cross sections.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        medium :
            The type of the doped medium.
        center_omega :
            The center angular frequency.  :math:`[rad\cdot ps^{-1}]`
        N_0 :
            The population in ground state. :math:`[nm^{-3}]`
        N_1 :
            The population in the excited state. :math:`[nm^{-3}]`
        T :
            The temperature. :math:`[K]`

        Returns
        -------
        :
            Value of the emission cross section. :math:`[nm^2]`

        References
        ----------
        .. [5] Valley, G.C., 2001. Modeling cladding-pumped Er/Yb fiber
               amplifiers. Optical Fiber Technology, 7(1), pp.21-44.

        """
        mc = McCumber(dopant=dopant, medium=medium)

        return mc.cross_section_emission(omega, sigma_a, T)


if __name__ == "__main__":
    """Plot the absorption cross sections from formula and file. A file
    with absorption cross section data must have been provided with
    Optcom.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc
    from optcom.parameters.fiber.absorption_section import DOPANT_RANGE
    from optcom.parameters.fiber.emission_section import DOPANTS

    x_data: List[np.ndarray] = []
    y_data: List[np.ndarray] = []
    plot_titles: List[str] = []
    line_labels: List[Optional[str]] = []
    line_colors: List[str] = []
    T: float = 293.1
    medium: str = 'sio2'
    nbr_samples: int = 1000
    omegas: np.ndarray
    absorp: oc.AbsorptionSection
    emission: oc.EmissionSection
    sigmas: np.ndarray
    lambdas: np.ndarray
    dopant_name: str
    # From formula and McCumber
    for dopant in DOPANTS:
        lambdas = np.linspace(DOPANT_RANGE[dopant][0],
                              DOPANT_RANGE[dopant][1],
                              nbr_samples)
        omegas = oc.omega_to_lambda(lambdas)
        absorp = oc.AbsorptionSection(dopant)
        sigmas = absorp(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        line_labels.append('absorption')
        line_colors.append('blue')
        emission = oc.EmissionSection(dopant=dopant, medium=medium, T=T,
                                     sigma_a=absorp)
        sigmas = emission(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = dopant[0].upper()+dopant[1:]
        line_labels.append('emission')
        line_colors.append('red')
        plot_titles.append("Cross sections {} from formula"
                           .format(dopant_name)
                           + "\n (absorption sections + McCumber relations)")
    # From file
    file_sigma_a: str = './data/fiber_amp/cross_section/absorption/yb.txt'
    csv_sigma_a: Callable = oc.CSVFit(file_sigma_a, conv_factor=[1e9, 1e18],
                                      conv_func=[oc.lambda_to_omega])
    file_sigma_e: str = './data/fiber_amp/cross_section/emission/yb.txt'
    csv_sigma_e: Callable = oc.CSVFit(file_sigma_e, conv_factor=[1e9, 1e18],
                                     conv_func=[oc.lambda_to_omega])
    lambdas = np.linspace(DOPANT_RANGE['yb'][0], DOPANT_RANGE['yb'][1],
                          nbr_samples)
    omegas = oc.omega_to_lambda(lambdas)
    sigmas = csv_sigma_a(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    line_labels.append('absoprtion')
    line_colors.append('blue')
    sigmas = csv_sigma_e(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    line_labels.append('emission')
    line_colors.append('red')
    plot_titles.append('Cross sections Yb from files.')
    # From file and McCumber
    emission = oc.EmissionSection(dopant='yb', medium=medium, T=T,
                                  sigma_a=csv_sigma_a)
    sigmas = csv_sigma_a(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    line_labels.append('absoprtion')
    line_colors.append('blue')
    sigmas = emission(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    line_labels.append('emission')
    line_colors.append('red')
    plot_titles.append("Cross sections from absorption Yb file and McCumber")

    oc.plot2d(x_data, y_data, x_labels=['Lambda'],
              y_labels=[r'Emission cross section, $\,\sigma_e\,(nm^2)$'],
              line_colors=line_colors, line_labels=line_labels,
              plot_titles=plot_titles, line_styles=['-.'],
              line_opacities=[0.0], plot_groups=[0,0,1,1,2,2])
