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

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_refractive_index import AbstractRefractiveIndex

# ref for Bs/Cs values: Brückner, V., 2011. To the use of Sellmeier
# formula. Senior Experten Service (SES) Bonn and HfT Leipzig, Germany,
# 42, pp.242-250.
# N.B. more significativ numbers for Cs bcs has been raised to square
media = ["sio2"]
# coeff_values = {medium: (Bs, Cs), ...}
coeff_values: Dict[str, Tuple[List[float], List[float]]] \
    = {"sio2": ([0.696750, 0.408218, 0.890815],
                [0.004770112356, 0.013377698244, 98.021068512481])}


class Sellmeier(AbstractRefractiveIndex):
    r"""Sellmeier equations.

    Represent the Sellmeier equations which are empirical relationship
    between refractive index and wavelength for a specific medium. [1]_

    References
    ----------
    .. [1] Malitson, I.H., 1965. Interspecimen comparison of the
           refractive index of fused silica. Josa, 55(10), pp.1205-1209.

    """

    def __init__(self, medium: str = 'sio2', Bs: Optional[List[float]] = None,
                 Cs: Optional[List[float]] = None) -> None:
        r"""
        Parameters
        ----------
        medium :
            The medium in which the wave propagates.
        Bs :
            B coefficients.
        Cs :
            C coefficients.

        """
        super().__init__(medium)
        self._Bs: List[float] = [0.0]
        self._Cs: List[float] = [0.0]
        if (Cs is not None and Bs is not None):
            self._Bs = Bs
            self._Cs = Cs
        elif (self._medium in media):
            self._Bs = coeff_values[self._medium][0]
            self._Cs = coeff_values[self._medium][1]
        else:
            util.warning_terminal("The medium is not recognised or no "
                "coefficients provided in Sellmeier equations, coefficients "
                "set to zero.")
    # ==================================================================
    def n(self, omega): # Headers in parent class
        r"""Compute the Sellmeier equations.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The refractive index.

        Notes
        -----

        .. math:: n^2(\lambda) = 1 + \sum_i B_i
                                 \frac{\lambda^2}{\lambda^2 - C_i}

        """
        Lambda = Domain.omega_to_lambda(omega)  # nm
        # Lambda in micrometer in formula
        Lambda = Lambda * 1e-3  # um
        if (isinstance(Lambda, float)):
            res = 1.0
            for i in range(len(self._Bs)):
                res += (self._Bs[i]*Lambda**2) / (Lambda**2 - self._Cs[i])
            res = math.sqrt(res)
        else:   # numpy.ndarray
            res = np.ones(Lambda.shape)
            for i in range(len(self._Bs)):
                res += (self._Bs[i]*Lambda**2) / (Lambda**2 - self._Cs[i])
            res = np.sqrt(res)

        return res


if __name__ == "__main__":

    import optcom.utils.plot as plot
    from optcom.domain import Domain

    sellmeier = Sellmeier("sio2")

    center_omega = Domain.lambda_to_omega(1050.0)
    print(sellmeier.n(center_omega))

    Lambda = np.linspace(120, 2120, 2000)
    omega = Domain.lambda_to_omega(Lambda)
    n = sellmeier.n(omega)

    x_labels = ['Lambda']
    y_labels = ['Refractive index']
    plot_titles = ["Refractive index of Silica from Sellmeier equations"]

    plot.plot(Lambda, n, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, opacity=0.0)
