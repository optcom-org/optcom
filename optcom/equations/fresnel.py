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
import cmath
from typing import Optional, Union

import optcom.utils.constants as cst
from optcom.equations.abstract_equation import AbstractEquation


class Fresnel(AbstractEquation):
    r"""Fresnel equations.

    Represent the Fresnel equations which describe the reflection and
    transmission of light at the interface of two media.

    """

    def __init__(self):

        return None
    # ==================================================================
    @staticmethod
    def reflectance_s_pol(n_1: Union[float, complex],
                          n_2: Union[float, complex], theta_i: float,
                          theta_t: Optional[float] = None):
        r"""Calcul the s-polarization reflectance.

        Parameters
        ----------
        n_1 :
            The refractive index of the medium of the incident light.
        n_2 :
            The refractive index of the medium of the transmitted light.
        theta_i :
            The angle of incidence.
        theta_t :
            The angle of transmission.

        Returns
        -------
        :
            The s-polarization reflectance.

        Notes
        -----

        .. math:: R_s = \bigg\lvert \frac{n_1 \cos(\theta_i)
                  - n_2\cos(\theta_t)}{n_1 \cos(\theta_i)
                  + n_2\cos(\theta_t)} \bigg\rvert^2
                  = \Bigg\lvert \frac{n_1\cos(\theta_i)
                  -n_2\sqrt{1-\big(\frac{n_1}{n_2}
                  \sin(\theta_i)\big)^2}}{n_1\cos(\theta_i)
                  + n_2 \sqrt{1-\big(\frac{n_1}{n_2}
                  \sin(\theta_i)\big)^2}} \Bigg\rvert^2

        """
        if (theta_t is not None):
            term_1 = n_1*math.cos(theta_i)
            term_2 = n_2*math.cos(theta_t)

        else:
            term_1 = n_1*math.cos(theta_i)
            term_2 = n_2*cmath.sqrt(1-(n_1/n_2*math.sin(theta_i))**2)

        return (abs((term_1-term_2) / (term_1+term_2)))**2
    # ==================================================================
    @staticmethod
    def reflectance_p_pol(n_1: Union[float, complex],
                          n_2: Union[float, complex], theta_i: float,
                          theta_t: Optional[float] = None):
        r"""Calcul the s-polarization reflectance.

        Parameters
        ----------
        n_1 :
            The refractive index of the medium of the incident light.
        n_2 :
            The refractive index of the medium of the transmitted light.
        theta_i :
            The angle of incidence.
        theta_t :
            The angle of transmission.

        Returns
        -------
        :
            The p-polarization reflectance.

        Notes
        -----

        .. math:: R_p = \bigg\lvert \frac{n_1 \cos(\theta_t)
                  - n_2\cos(\theta_i)}{n_1 \cos(\theta_t)
                  + n_2\cos(\theta_i)} \bigg\rvert^2
                  = \Bigg\lvert\frac{n_1\sqrt{1-\big(\frac{n_1}{n_2}
                  \sin(\theta_i)\big)^2}
                  - n_2\cos(\theta_i)}{n_1\sqrt{1-\big(\frac{n_1}{n_2}
                  \sin(\theta_i)\big)^2}
                  + n_2\cos(\theta_i) } \Bigg\rvert^2

        """
        if (theta_t is not None):
            term_1 = n_1*math.cos(theta_t)
            term_2 = n_2*math.cos(theta_i)

        else:
            term_1 = n_1*cmath.sqrt(1-(n_1/n_2*math.sin(theta_i))**2)
            term_2 = n_2*math.cos(theta_i)

        return (abs((term_1-term_2) / (term_1+term_2)))**2
    # ==================================================================
    @staticmethod
    def irradiance_s_pol(n_1: Union[float, complex],
                         n_2: Union[float, complex], theta_i: float,
                         theta_t: Optional[float] = None):
        r"""Calcul the s-polarization irradiance.

        Parameters
        ----------
        n_1 :
            The refractive index of the medium of the incident light.
        n_2 :
            The refractive index of the medium of the transmitted light.
        theta_i :
            The angle of incidence.
        theta_t :
            The angle of transmission.

        Returns
        -------
        :
            The s-polarization irradiance.

        Notes
        -----

        .. math:: T_s = 1 - R_s

        """

        return 1 - Fresnel.reflectance_s_pol(n_1, n_2, theta_i, theta_t)
    # ==================================================================
    @staticmethod
    def irradiance_p_pol(n_1: Union[float, complex],
                         n_2: Union[float, complex], theta_i: float,
                         theta_t: Optional[float] = None):
        r"""Calcul the p-polarization irradiance.

        Parameters
        ----------
        n_1 :
            The refractive index of the medium of the incident light.
        n_2 :
            The refractive index of the medium of the transmitted light.
        theta_i :
            The angle of incidence.
        theta_t :
            The angle of transmission.

        Returns
        -------
        :
            The p-polarization irradiance.

        Notes
        -----

        .. math:: T_p = 1 - R_p

        """

        return 1 - Fresnel.reflectance_p_pol(n_1, n_2, theta_i, theta_t)


if __name__ == "__main__":

    n_1 = 1.4
    n_2 = 1.6
    theta_i = 0.9
    theta_t = 1.3
    print(Fresnel.reflectance_s_pol(n_1, n_2, theta_i))
    print(Fresnel.reflectance_s_pol(n_1, n_2, theta_i, theta_t))
    print(Fresnel.reflectance_p_pol(n_1, n_2, theta_i))
    print(Fresnel.reflectance_p_pol(n_1, n_2, theta_i, theta_t))
    print(Fresnel.irradiance_s_pol(n_1, n_2, theta_i))
    print(Fresnel.irradiance_s_pol(n_1, n_2, theta_i, theta_t))
    print(Fresnel.irradiance_p_pol(n_1, n_2, theta_i))
    print(Fresnel.irradiance_p_pol(n_1, n_2, theta_i, theta_t))
