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
        """
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
        """Compute the derivatives of the emission cross sections.

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

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.parameters.fiber.absorption_section import AbsorptionSection
    from optcom.parameters.fiber.absorption_section import DOPANT_RANGE
    from optcom.parameters.fiber.emission_section import EmissionSection
    from optcom.parameters.fiber.emission_section import DOPANTS
    from optcom.utils.utilities_user import CSVFit


    x_data: List[np.ndarray] = []
    y_data: List[np.ndarray] = []
    plot_titles: List[str] = []
    plot_labels: List[Optional[str]] = []
    plot_colors: List[str] = []
    T: float = 293.1
    medium: str = 'sio2'
    nbr_samples: int = 1000
    omegas: np.ndarray
    absorp: AbsorptionSection
    emission: EmissionSection
    sigmas: np.ndarray
    lambdas: np.ndarray
    dopant_name: str
    # From formula and McCumber
    for dopant in DOPANTS:
        lambdas = np.linspace(DOPANT_RANGE[dopant][0],
                              DOPANT_RANGE[dopant][1],
                              nbr_samples)
        omegas = Domain.omega_to_lambda(lambdas)
        absorp = AbsorptionSection(dopant)
        sigmas = absorp(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        plot_labels.append('absorption')
        plot_colors.append('blue')
        emission = EmissionSection(dopant=dopant, medium=medium, T=T,
                                   sigma_a=absorp)
        sigmas = emission(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = dopant[0].upper()+dopant[1:]
        plot_labels.append('emission')
        plot_colors.append('red')
        plot_titles.append("Cross sections {} from formula"
                           .format(dopant_name)
                           + "\n (absorption sections + McCumber relations)")
    # From file
    file_sigma_a: str = './data/fiber_amp/cross_section/absorption/yb.txt'
    csv_sigma_a: Callable = CSVFit(file_sigma_a, conv_factor=[1e9, 1e18],
                                   conv_func=[Domain.lambda_to_omega])
    file_sigma_e: str = './data/fiber_amp/cross_section/emission/yb.txt'
    csv_sigma_e: Callable = CSVFit(file_sigma_e, conv_factor=[1e9, 1e18],
                                   conv_func=[Domain.lambda_to_omega])
    lambdas = np.linspace(DOPANT_RANGE['yb'][0], DOPANT_RANGE['yb'][1],
                          nbr_samples)
    omegas = Domain.omega_to_lambda(lambdas)
    sigmas = csv_sigma_a(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    plot_labels.append('absoprtion')
    plot_colors.append('blue')
    sigmas = csv_sigma_e(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    plot_labels.append('emission')
    plot_colors.append('red')
    plot_titles.append('Cross sections Yb from files.')
    # From file and McCumber
    emission = EmissionSection(dopant='yb', medium=medium, T=T,
                               sigma_a=csv_sigma_a)
    sigmas = csv_sigma_a(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    plot_labels.append('absoprtion')
    plot_colors.append('blue')
    sigmas = emission(omegas)
    x_data.append(lambdas)
    y_data.append(sigmas)
    plot_labels.append('emission')
    plot_colors.append('red')
    plot_titles.append("Cross sections from absorption Yb file and McCumber")

    plot.plot2d(x_data, y_data, x_labels=['Lambda'],
                y_labels=[r'Emission cross section, $\,\sigma_e\,(nm^2)$'],
                plot_colors=plot_colors, plot_labels=plot_labels,
                plot_titles=plot_titles, plot_linestyles=['-.'],
                opacity=[0.0], plot_groups=[0,0,1,1,2,2])