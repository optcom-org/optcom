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
from typing import Any, Callable, List, Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.attenuation import Attenuation
from optcom.effects.dispersion import Dispersion
from optcom.effects.gain_saturation import GainSaturation
from optcom.equations.abstract_equation import AbstractEquation
from optcom.equations.abstract_nlse import AbstractNLSE
from optcom.equations.abstract_refractive_index import AbstractRefractiveIndex
from optcom.equations.re_fiber import REFiber
from optcom.field import Field
from optcom.parameters.fiber.nl_coefficient import NLCoefficient
from optcom.parameters.fiber.nl_index import NLIndex


class AbstractAmpNLSE(AbstractNLSE):

    def __init__(self, re: REFiber,
                 alpha: Optional[Union[List[float], Callable]],
                 alpha_order: int,
                 beta: Optional[Union[List[float], Callable]],
                 beta_order: int, gamma: Optional[Union[float, Callable]],
                 gain_order: int, R_0: float, R_L: float,
                 nl_index: Optional[Union[float, Callable]], ATT: bool,
                 DISP: bool, GS: bool, medium: str, dopant: str) -> None:
        r"""
        Parameters
        ----------
        re :
            A fiber rate equation object.
        alpha :
            The derivatives of the attenuation coefficients.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        alpha_order :
            The order of alpha coefficients to take into account. (will
            be ignored if alpha values are provided - no file)
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        beta_order :
            The order of beta coefficients to take into account. (will
            be ignored if beta values are provided - no file)
        gamma :
            The non linear coefficient.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]` If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        gain_order :
            The order of the gain coefficients to take into account.
            (from the Rate Equations resolution)
        R_0 :
            The reflectivity at the fiber start.
        R_L :
            The reflectivity at the fiber end.
        nl_index :
            The non linear index. Used to calculate the non linear
            parameter. :math:`[m^2\cdot W^{-1}]`
        ATT :
            If True, trigger the attenuation.
        DISP :
            If True, trigger the dispersion.
        GS :
            If True, trigger the gain saturation.
        medium :
            The main medium of the fiber amplifier.
        dopant :
            The doped medium of the fiber amplifier.

        """
        # keep all methods from NLSE but bypass constructor
        AbstractEquation.__init__(self) # grand parent constructor
        self._medium: str = medium
        self._dopant: str = dopant
        self._re: REFiber = re
        self._R_L: float = R_L
        self._R_0: float = R_0
        self._delay_time: Array[float]
        self._coprop: bool
        # Effects ------------------------------------------------------
        self._att_ind: int = -1
        self._disp_ind: int = -1
        self._gs_ind: int = -1
        self._gain_ind: int = -1
        self._gain_order: int = gain_order
        self._beta_order: int = beta_order
        if (ATT):
            self._effects_lin.append(Attenuation(alpha, alpha_order))
            self._att_ind = len(self._effects_lin) - 1
        if (DISP):
            self._effects_lin.append(Dispersion(beta, beta_order))
            self._disp_ind = len(self._effects_lin) - 1
        if (GS):
            start_taylor_gain = 1
            self._effects_lin.append(GainSaturation(re, [0.0]))
            self._gs_ind = len(self._effects_lin) - 1
        else:
            start_taylor_gain = 0
        alpha_temp = [0.0 for i in range(self._gain_order+1)]
        self._effects_lin.append(Attenuation(alpha_temp,
                                             start_taylor=start_taylor_gain))
        self._gain_ind = len(self._effects_lin) - 1
        # Gamma --------------------------------------------------------
        self._nl_index: Union[float, Callable] = NLIndex(medium=medium) if\
            (nl_index is None) else nl_index
        self._gamma: Array[float]
        self._predict_gamma: Optional[Callable] = None
        self._custom_gamma: bool = False
        if (gamma is not None):
            if (callable(gamma)):
                self._custom_gamma = True
                self._predict_gamma = gamma
            else:
                self._gamma = np.asarray(util.make_list(gamma))
        else:
            self._predict_gamma = NLCoefficient.calc_nl_coefficient
    # ==================================================================
    class RefractiveIndex(AbstractRefractiveIndex):
        """Container for refractive index computation."""

        def __init__(self, re, step):
            self._re = re
            self._step = step
        # ==============================================================
        def n(self, omega): # headers in parent class
            res = self._re.get_n_core(omega, self._step)

            return res
    # ==================================================================
    def get_criterion(self, waves_f, waves_b):
        criterion = np.sum(Field.temporal_power(waves_f))
        criterion += np.sum(Field.temporal_power(waves_b))

        return criterion
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        # Bypass open in parent to avoid gamma calculation (in set here)
        AbstractEquation.open(self, domain, *fields)
        self._delay_time = np.zeros(self._center_omega.shape)
        # Initiate counter ---------------------------------------------
        self._iter = -1 # -1 bcs increment in the first init. cond.
        self._waves_s_ref = np.zeros((self._len_eq(0), domain.samples))
        self._waves_p_ref = np.zeros((self._len_eq(1), domain.samples))
    # ==================================================================
    def set(self, waves: Array[cst.NPFT], h: float, z: float) -> None:
        super().set(waves, h, z)
        step = int(round(z/h))
        self._call_counter += 1
        if (not self._iter and not self._coprop):
            step = 0
        # Gain ---------------------------------------------------------
        new_alpha = self._re.absorption(step, self._gain_order)
        self._effects_lin[self._gain_ind].alpha = new_alpha
        # Gain saturation ----------------------------------------------
        if (self._gs_ind != -1):
            self._effects_lin[self._gs_ind].alpha = new_alpha[:,0]
        if (self._gs_ind != -1):
            self._effects_lin[self._gs_ind].update(step=step)
        # Non-linear parameter -----------------------------------------
        if (self._predict_gamma is not None):
            self._gamma = np.zeros_like(self._center_omega)
            if (self._custom_gamma):
                for id, omega in enumerate(self._center_omega):
                    self._gamma[id] = self._predict_gamma(omega)
            else:
                for id, omega in enumerate(self._center_omega):
                    eff_area = self._re.get_eff_area(omega, step)
                    nl_index = self._nl_index(omega)\
                        if callable(self._nl_index) else self._nl_index
                    self._gamma[id] = self._predict_gamma(omega, nl_index,
                                                          eff_area)
        # Dispersion ---------------------------------------------------
        if (self._disp_ind != -1):
            class_n = AbstractAmpNLSE.RefractiveIndex(self._re, step)
            self._effects_lin[self._disp_ind].update(class_n=class_n)
    # ==================================================================
    def initial_condition(self, waves: Array[cst.NPFT], end: bool
                          ) -> Array[cst.NPFT]:
        r"""
        Parameters
        ----------
        waves :
            The propagationg waves.
        end :
            If true, boundary counditions at z=L, else at z=0.

        Returns
        -------
        :
            The waves after initial conditions.

        Notes
        -----
        If co-propagating signal and pump:

        .. math:: \begin{split}
                    A_s^+(z=0) &= \sqrt{R_0} A_s^-(z=0)
                                  + P_{s,ref}^+(z=0)\\
                    A_p^+(z=0) &= \sqrt{R_0} A_p^-(z=0)
                                  + P_{p, ref}^+(z=0)\\
                    A_s^-(z=L) &= \sqrt{R_L} A_s^+(z=L)\\
                    A_p^-(z=L) &= \sqrt{R_L} A_s^+(z=L)
                  \end{split}

        If counter-propagating signal and pump:

        .. math:: \begin{split}
                    A_s^+(z=0) &= \sqrt{R_0} A_s^-(z=0)
                                  + P_{s,ref}^+(z=0)\\
                    A_p^+(z=0) &= \sqrt{R_0} A_p^-(z=0)\\
                    A_s^-(z=L) &= \sqrt{R_L} A_s^+(z=L)\\
                    A_p^-(z=L) &= \sqrt{R_L} A_s^+(z=L)
                                  + P_{p, ref}^-(z=0)
                  \end{split}

        """
        # Update counter (per iter) ------------------------------------
        self._call_counter = 0
        self._iter += 1
        if (not self._iter and not self._call_counter):   # Very first call
            self._coprop = not end
        # N.B. : could also include reflection losses
        # Truth table:
        # if coprop :  iter |   z   |   to do
        #              0    |   0   |   init pump
        #              1    |   L   |   CI pump
        #              2    |   0   |   CI pump & init signal
        #              3    |   L   |   CI pump & CI signal
        # if counterprop:   iter|   z   |   to do
        #                   0   |   L   |   init pump
        #                   1   |   0   |   CI pump & init signal
        #                   2   |   L   |   CI pump & CI signal
        # Make shallow copy, does not change waves_prop
        waves_prop_s = self._in_eq_waves(waves, 0)
        waves_prop_p = self._in_eq_waves(waves, 1)
        if (self._iter):
            # Initial conditions ---------------------------------------
            if (end):   # End of the fiber z=L
                if (self._iter > 1):
                    waves_prop_s = math.sqrt(self._R_L) * waves_prop_s
                else:
                    waves_prop_s = np.zeros_like(waves_prop_s)
                waves_prop_p = math.sqrt(self._R_L) * waves_prop_p
                if (not self._coprop):
                    waves_prop_p += self._waves_p_ref
            else:               # begining of the fiber z=0
                if (self._iter > 2):
                    waves_prop_s = (self._waves_s_ref
                                    + math.sqrt(self._R_0)*waves_prop_s)
                elif (self._iter == 1 or self._iter == 2):   # First signal
                    waves_prop_s = self._waves_s_ref
                else:
                    waves_prop_s = np.zeros_like(waves_prop_s)
                waves_prop_p = math.sqrt(self._R_0) * waves_prop_p
                if (self._coprop):
                    waves_prop_p += self._waves_p_ref
        else:
            self._waves_s_ref = copy.deepcopy(waves_prop_s)
            self._waves_p_ref = copy.deepcopy(waves_prop_p)
            waves_prop_s = np.zeros_like(waves_prop_s)

        return np.vstack((waves_prop_s, waves_prop_p))
