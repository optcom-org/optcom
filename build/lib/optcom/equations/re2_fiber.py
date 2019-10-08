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
from typing import Callable, List, Optional, overload, Union

import numpy as np
from nptyping import Array
from scipy import interpolate

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.utils.fft import FFT
from optcom.domain import Domain
from optcom.effects.absorption import Absorption
from optcom.effects.stimulated_emission import StimulatedEmission
from optcom.equations.abstract_refractive_index import AbstractRefractiveIndex
from optcom.equations.re_fiber import REFiber
from optcom.equations.mccumber import McCumber
from optcom.equations.resonant_index import ResonantIndex
from optcom.equations.sellmeier import Sellmeier
from optcom.field import Field
from optcom.parameters.fiber.effective_area import EffectiveArea
from optcom.parameters.fiber.numerical_aperture import NumericalAperture
from optcom.parameters.fiber.overlap_factor import OverlapFactor


class RE2Fiber(REFiber):
    r"""Rate equations for a 2 levels system in fiber amplifier.

    Represent the rate equations of a 2 levels system in steady state
    with pump power :math:`P_p`, signal power :math:`P_s` and inversion
    of population :math:`N_1` as parameters.

    Notes
    -----

    .. math:: \begin{alignat}{1}
                &N_1(z) =  \frac{\frac{1}{hcA_c}\Big[\sum^K_{k=1}
                \Gamma_{s,k}\lambda_k\big[ \sigma_{a,s}(\lambda_k)N_T]
                P_{s,k}^{\pm}(z) + \sum^L_{l=1}
                \Gamma_{p,l}\lambda_l\big[\sigma_{a,p}(\lambda_l)N_T
                \big] P_{p,l}^{\pm}(z)\Big]}
                {\frac{1}{hcA_c}\Big[\sum^K_{k=1}
                \Gamma_{s,k}\lambda_k\big[ \sigma_{a,s}(\lambda_k)
                + \sigma_{e,s}(\lambda_k)\big] P_{s,k}^{\pm}(z)
                + \sum^L_{l=1} \Gamma_{p,l}\lambda_l
                \big[\sigma_{a,p}(\lambda_l)+\sigma_{e,p}(\lambda_l)
                \big] P_{p,l}^{\pm}(z)\Big] - \frac{1}{\tau}}\\
                &\frac{\partial P_{s,k}^{\pm}(z)}{\partial z}
                = \pm \bigg[\Gamma_k\Big[\big(\sigma_{a,s}(\lambda_k)
                + \sigma_{e,s}(\lambda_k)\big)N_1(z)
                - \sigma_{a,s}(\lambda_k)N_T\Big] P_{s,k}^\pm (z)
                - \eta_{s,k} P_{s,k}^\pm (z)
                + \Gamma_k\sigma_{e,s}(\lambda_k) N_1(z)
                \frac{2hc^2 \Delta\lambda}{\lambda_k^3} \bigg]\\
                &\frac{\partial P_{p,l}^{\pm}(z)}{\partial z}
                = \pm \bigg[\Gamma_k\Big[\big(\sigma_{a,p}(\lambda_l)
                + \sigma_{e,p}(\lambda_l)\big)N_1(z)
                - \sigma_{a,p}(\lambda_l)N_T\Big] P_{p,l}^\pm (z)
                - \eta_{p,l} P_{p,l}^\pm (z) \bigg]
              \end{alignat}

    """

    def __init__(self, sigma_a: Optional[Union[List[float], Callable]] = None,
                 sigma_e: Optional[Union[List[float], Callable]] = None,
                 n_core: Optional[Union[float, List[float]]] = None,
                 n_clad: Optional[Union[float, List[float]]] = None,
                 NA: Optional[Union[float, List[float]]] = None,
                 temperature: float = 293.15, tau_meta: float = cst.TAU_META,
                 N_T: float = cst.N_T,
                 core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 area_doped: Optional[float] = None,
                 eta_s: float = cst.ETA_SIGNAL, eta_p: float = cst.ETA_PUMP,
                 R_0: float = cst.R_0, R_L: float = cst.R_L,
                 signal_width: List[float] = [1.0],
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 dopant: str = cst.DEF_FIBER_DOPANT,
                 step_update: bool = False) -> None:
        r"""
        Parameters
        ----------
        sigma_a :
            The absorption cross sections of the signal and the pump
            (1<=len(sigma_a)<=2). :math:`[nm^2]` If a callable is
            provided, varibale must be wavelength. :math:`[nm]`
        sigma_e :
            The emission cross sections of the signal and the pump
            (1<=len(sigma_a)<=2). :math:`[nm^2]` If a callable is
            provided, varibale must be wavelength. :math:`[nm]`
        n_core :
            The refractive index of the core. If the medium is not
            recognised by Optcom, at least two elements should be
            provided out of those three: n_core, n_clad, NA.
        n_clad :
            The refractive index of the cladding. If the medium is not
            recognised by Optcom, at least two elements should be
            provided out of those three: n_core, n_clad, NA.
        NA :
            The numerical aperture. If the medium is not
            recognised by Optcom, at least two elements should be
            provided out of those three: n_core, n_clad, NA.
        temperature :
            The temperature of the medium. :math:`[K]`
        tau_meta :
            The metastable level lifetime. :math:`[\mu s]`
        N_T :
            The total doping concentration. :math:`[nm^{-3}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        area_doped :
            The doped area. :math:`[\mu m^2]` If None, will be
            approximated to the core area.
        eta_s :
            The background signal loss. :math:`[km^{-1}]`
        eta_p :
            The background pump loss. :math:`[km^{-1}]`
        R_0 :
            The reflectivity at the fiber start.
        R_L :
            The reflectivity at the fiber end.
        signal_width :
            The width of each channel of the signal. :math:`[ps]`
        medium :
            The main medium of the fiber amplifier.
        dopant :
            The doped medium of the fiber amplifier.
        step_update :
            If True, update the signal and pump power from the wave
            arrays at each space step.

        """
        super().__init__(2) # 2 equations

        # Variable declaration -----------------------------------------
        self._power_s_f: Array[float] = np.array([])
        self._power_s_b: Array[float] = np.array([])
        self._power_s_ref: Array[float] = np.array([])
        self._power_p_f: Array[float] = np.array([])
        self._power_p_b: Array[float] = np.array([])
        self._power_p_ref: Array[float] = np.array([])
        self._power_ase_f: Array[float] = np.array([])
        self._power_ase_b: Array[float] = np.array([])
        self._N_1: Array[float] = np.array([])
        self._sigma_a_s: Array[float] = np.array([])
        self._sigma_a_p: Array[float] = np.array([])
        self._sigma_e_s: Array[float] = np.array([])
        self._sigma_e_p: Array[float] = np.array([])
        self._n_tot_s: Array[float] = np.array([])
        self._n_tot_p: Array[float] = np.array([])
        self._n_0_s: Array[float] = np.array([])
        self._n_0_p: Array[float] = np.array([])
        self._n_clad_s: Array[float] = np.array([])
        self._n_clad_p: Array[float] = np.array([])
        self._Gamma_s: Array[float] = np.array([])
        self._Gamma_p: Array[float] = np.array([])
        self._A_eff_s: Array[float] = np.array([])
        self._A_eff_p: Array[float] = np.array([])
        # Variable initialization --------------------------------------
        self._coprop: bool = True

        self._step_update: bool = step_update

        self._signal_width = signal_width

        self._medium: str = medium
        self._dopant: str = dopant

        self._T: float = temperature
        self._N_T: float = N_T
        self._tau: float = tau_meta * 1e6   # us -> ps
        self._decay: float = 1/self._tau

        if (area_doped is None):
            area_doped = (cst.PI*core_radius**2)
        # TO DO: make option cladding doped and thus A_d_p = A_cladding
        A_d_p = (cst.PI*core_radius**2)
        A_d_s = area_doped
        self._overlap_s = OverlapFactor(A_d_s)
        self._overlap_p = OverlapFactor(A_d_p)
        self._A_d_s =  A_d_s * 1e6  # um^2 -> nm^2
        # Doping region area, can be cladding or core
        self._A_d_p = area_doped * 1e6  # um^2 -> nm^2

        self._core_radius = core_radius

        self._res_index: ResonantIndex = ResonantIndex(medium=self._dopant)

        # Set n_core and n_clad depending on NA (forget NA afterwards)
        # Assume that only core medium can be calculated with Sellmeier
        # equations.  Could be changed later. Would be easier to have
        # also cladding medium set by Sellmeier but cladding medium
        # not always available. -> To Discuss
        self._calc_n_core: Optional[Sellmeier] = None
        self._calc_n_clad: Optional[Callable] = None
        if (NA is None and n_clad is None):
            util.warning_terminal("Must specify at least NA or n_clad, "
                "n_clad will be set to 0.")
            self._n_clad_s_value = 0.0
            self._n_clad_p_value = 0.0
        if (n_clad is not None):    # default is NA is None
            n_clad_temp = util.make_list(n_clad, 2)
            self._n_clad_s_value = n_clad_temp[0]
            self._n_clad_p_value = n_clad_temp[1]
        if (n_core is None):
            if (NA is not None and n_clad is not None):
                n_clad_temp = util.make_list(n_clad, 2)
                self._n_0_s_value =\
                    NumericalAperture.calc_n_core(NA, n_clad_temp[0])
                self._n_0_p_value =\
                    NumericalAperture.calc_n_core(NA, n_clad_temp[1])
            else:
                self._calc_n_core = Sellmeier(medium=self._medium)
                if (NA is not None and n_clad is None):
                    self._calc_n_clad = NumericalAperture.calc_n_clad
                    NA_temp = util.make_list(NA, 2)
                    self._NA_value_s = NA_temp[0]
                    self._NA_value_p = NA_temp[1]
        else:
            n_core_: List[float] = util.make_list(n_core, 2)
            self._n_0_s_value = n_core_[0]
            self._n_0_p_value = n_core_[1]
            if (NA is not None and n_clad is None):
                self._n_clad_s_value =\
                    NumericalAperture.calc_n_clad(NA, n_core_[0])
                self._n_clad_p_value =\
                    NumericalAperture.calc_n_clad(NA, n_core_[1])

        self._eff_area_s = EffectiveArea(core_radius)
        self._eff_area_p = EffectiveArea(core_radius)
        self._eta_s = eta_s * 1e-12  # km^{-1} -> nm^{-1}
        self._eta_p = eta_p * 1e-12  # km^{-1} -> nm^{-1}
        self._factor_s = 1 / (cst.HBAR*self._A_d_s)
        self._factor_p = 1 / (cst.HBAR*self._A_d_p)

        self._absorp: Optional[Absorption] = None
        if (sigma_a is None):
            self._absorp = Absorption(dopant=dopant)
        else:
            if (callable(sigma_a)):
                self._absorp = Absorption(predict=sigma_a)
            else:
                sigma_a_: List[float] = util.make_list(sigma_a, 2)
                self._sigma_a_s_value = sigma_a_[0]
                self._sigma_a_p_value = sigma_a_[1]

        self._sigma_e_mccumber = False
        self._stimu: Optional[StimulatedEmission] = None
        if (sigma_e is None):
            self._sigma_e_mccumber = True
        else:
            if (callable(sigma_e)):
                self._stimu = StimulatedEmission(predict=sigma_e)
            else:
                sigma_e_: List[float] = util.make_list(sigma_e, 2)
                self._sigma_e_s_value = sigma_e_[0]
                self._sigma_e_p_value = sigma_e_[1]

        self._R_0 = R_0
        self._R_L = R_L
    # ==================================================================
    # Built in getters =================================================
    # ==================================================================
    @overload
    def get_n_0(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_n_0(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_n_0(self, omega):
        """Return the ground refractive index of the core.
        Assume that omega is part of current center omegas.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The ground refractive index of the core.

        """
        # Handle isinstance(omega, float) ------------------------------
        revert = False
        if (isinstance(omega, float)):
            omega = np.array([omega])
            revert = True
        # Getter -------------------------------------------------------
        if (self._calc_n_core is not None):
            res = self._calc_n_core.n(omega)
        else:
            res = np.zeros_like(omega)
            for i in range(len(omega)):
                if (util.is_float_in_list(omega[i], self._center_omega_s)):
                    res[i] = self._n_0_s_value
                else:
                    res[i] = self._n_0_p_value
        # Revert float type of omega -----------------------------------
        if (revert):
            res = res[0]

        return res
    # ==================================================================
    @overload
    def get_n_core(self, omega: float, step: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_n_core(self, omega: Array[float], step: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_n_core(self, omega, step):
        """Return the refractive index of the core.
        Assume that omega is part of current center omegas.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        step :
            The current step of the computation.

        Returns
        -------
        :
            The refractive index of the core at the specified space
            step.

        """
        N_1 = self._N_1[self._step]
        n_0 = self.get_n_0(omega)

        return n_0 + self._res_index.n(omega, n_0, N_1)
    # ==================================================================
    @overload
    def get_n_clad(self, omega: float, step: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_n_clad(self, omega: Array[float], step: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_n_clad(self, omega, step):
        """Return the refractive index of the core.
        Assume that omega is part of current center omegas.
        N.B. : variable step currently not considered, but could be if
        cladding doped.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        step :
            The current step of the computation.

        Returns
        -------
        :
            The refractive index of the core.

        """
        # Handle isinstance(omega, float) ------------------------------
        revert = False
        if (isinstance(omega, float)):
            omega = np.array([omega])
            revert = True
        # Getter -------------------------------------------------------
        res = np.zeros_like(omega)
        for i in range(len(omega)):
            if (util.is_float_in_list(omega[i], self._center_omega_s)):
                if (self._calc_n_clad is None):
                    res[i] = self._n_clad_s_value
                else:
                    res[i] =\
                        self._calc_n_clad(self._NA_value_s,
                                          self.get_n_core(omega[i], step))
            else:
                if (self._calc_n_clad is None):
                    res[i] = self._n_clad_p_value
                else:
                    res[i] =\
                        self._calc_n_clad(self._NA_value_p,
                                          self.get_n_core(omega[i], step))
        # Revert float type of omega -----------------------------------
        if (revert):
            res = res[0]

        return res
    # ==================================================================
    @overload
    def get_eff_area(self, omega: float, step: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_eff_area(self, omega: Array[float], step: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_eff_area(self, omega, step):
        """Return the effective area.
        Assume that omega is part of current center omegas.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        step :
            The current step of the computation.

        Returns
        -------
        :
            The effective area at the specified space step.
            :math:`[\mu m^2]`

        """
        # Handle isinstance(omega, float) ------------------------------
        revert = False
        if (isinstance(omega, float)):
            omega = np.array([omega])
            revert = True
        # Getter -------------------------------------------------------
        NA = NumericalAperture.calc_NA(self.get_n_core(omega, step),
                                       self.get_n_clad(omega, step))
        res = np.zeros_like(omega)
        for i in range(len(omega)):
            if (util.is_float_in_list(omega[i], self._center_omega_s)):
                res[i] = self._eff_area_s(omega[i], NA)
            else:
                res[i] = self._eff_area_p(omega[i], NA)
        # Revert float type of omega -----------------------------------
        if (revert):
            res = res[0]

        return res
    # ==================================================================
    @overload
    def get_sigma_a(self, omega: float, step: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_sigma_a(self, omega: Array[float], step: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_sigma_a(self, omega, step):
        """Return the absorption cross section.
        Assume that omega is part of current center omegas.
        Variable 'step' currently not considered, only for constitency.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        step :
            The current step of the computation.

        Returns
        -------
        :
            The absorption cross section. :math:`[nm^2]`

        """
        # Handle isinstance(omega, float) ------------------------------
        revert = False
        if (isinstance(omega, float)):
            omega = np.array([omega])
            revert = True
        # Getter -------------------------------------------------------
        res = np.zeros_like(omega)
        for i in range(len(omega)):
            if (self._absorp is not None):
                res[i] = self._absorp.get_cross_section(omega[i])
            else:
                if (util.is_float_in_list(omega[i], self._center_omega_s)):
                    res[i] = self._sigma_a_s_value
                else:
                    res[i] = self._sigma_a_p_value
        # Revert float type of omega -----------------------------------
        if (revert):
            res = res[0]

        return res
    # ==================================================================
    @overload
    def get_sigma_e(self, omega: float, step: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_sigma_e(self, omega: Array[float], step: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_sigma_e(self, omega, step):
        """Return the emission cross section.
        Assume that omega is part of current center omegas.
        Variable 'step' currently not considered, only for constitency.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        step :
            The current step of the computation.

        Returns
        -------
        :
            The emission cross section. :math:`[nm^2]`

        """
        # Handle isinstance(omega, float) ------------------------------
        revert = False
        if (isinstance(omega, float)):
            omega = np.array([omega])
            revert = True
        # Getter -------------------------------------------------------
        res = np.zeros_like(omega)
        for i in range(len(omega)):
            if (self._stimu is not None):
                res[i] = self._stimu.get_cross_section(omega[i])
            else:
                if (self._sigma_e_mccumber):
                    # Assume omega = center_omega
                    res[i] = McCumber.calc_cross_section_emission(
                        self.get_sigma_a(omega, step), 0.0, 0.0,
                        self.N_0[step], self.N_1[step], self._T)
                else:
                    if (util.is_float_in_list(omega[i], self._center_omega_s)):
                        res[i] = self._sigma_e_s_value
                    else:
                        res[i] = self._sigma_e_p_value
        # Revert float type of omega -----------------------------------
        if (revert):
            res = res[0]

        return res
    # ==================================================================
    @overload
    def get_Gamma(self, omega: float, step: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_Gamma(self, omega: Array[float], step: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_Gamma(self, omega, step):
        """Return the overlap factor.
        Assume that omega is part of current center omegas.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        step :
            The current step of the computation.

        Returns
        -------
        :
            The overlap factor at the specified space step.

        """
        # Handle isinstance(omega, float) ------------------------------
        revert = False
        if (isinstance(omega, float)):
            omega = np.array([omega])
            revert = True
        # Getter -------------------------------------------------------
        res = np.zeros_like(omega)
        for i in range(len(omega)):
            if (util.is_float_in_list(omega[i], self._center_omega_s)):
                res[i] = self._overlap_s(self.get_eff_area(omega[i], step))
            else:
                res[i] = self._overlap_p(self.get_eff_area(omega[i], step))
        # Revert float type of omega -----------------------------------
        if (revert):
            res = res[0]

        return res
    # ==================================================================
    def get_criterion(self, waves_f: Array[cst.NPFT],
                      waves_p: Array[cst.NPFT]) -> float:

        return np.sum(self._power_s_f[-1]) + np.sum(self._power_s_b[-1])
    # ==================================================================
    # Built in getters =================================================
    # ==================================================================
    @property
    def N_1(self) -> Array[float]:

        return self._N_1
    # ==================================================================
    @property
    def N_0(self) -> Array[float]:

        return (self._N_T - self._N_1)
    # ==================================================================
    @property
    def power_ase_forward(self) -> Array[float]:

        return (np.sum(self._power_ase_f, axis=2)).T
    # ==================================================================
    @property
    def power_ase_backward(self) -> Array[float]:

        return (np.sum(self._power_ase_b, axis=2)).T
    # ==================================================================
    @property
    def power_signal_forward(self) -> Array[float]:

        return (np.sum(self._power_s_f, axis=2)).T
    # ==================================================================
    @property
    def power_signal_backward(self) -> Array[float]:

        return (np.sum(self._power_s_b, axis=2)).T
    # ==================================================================
    @property
    def power_pump_forward(self) -> Array[float]:

        return (np.sum(self._power_p_f, axis=2)).T
    # ==================================================================
    @property
    def power_pump_backward(self) -> Array[float]:

        return (np.sum(self._power_p_b, axis=2)).T
    # ==================================================================
    # Private getters ==================================================
    # ==================================================================
    def _get_power_s(self, waves: Array[cst.NPFT]) -> Array[float]:

        waves_s = self._in_eq_waves(waves, 0)

        return Field.spectral_power(waves_s, False)
    # ==================================================================
    def _get_power_p(self, waves: Array[cst.NPFT]) -> Array[float]:

        waves_p = self._in_eq_waves(waves, 1)

        return Field.spectral_power(waves_p, False)
    # ==================================================================
    # Open - IC - set - call - update - close ==========================
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        super().open(domain, *fields)
        # Initialize angular frequency ---------------------------------
        self._samples = len(self._omega)
        self._omega_s  = self._in_eq_waves(self._omega_all, 0)
        self._omega_p = self._in_eq_waves(self._omega_all, 1)
        # Iniate array for center omegas data --------------------------
        self._center_omega_s = self._in_eq_waves(self._center_omega, 0)
        self._center_omega_p = self._in_eq_waves(self._center_omega, 1)
        # Signal channel width ----------------------------------------
        signal_width = np.array(util.make_list(self._signal_width,
                                               len(self._center_omega_s)))
        self._width_omega_s = (1.0 / signal_width) * 2.0 * cst.PI
        # Initiate the variables array - power and pop. density --------
        self._shape_step_s = (len(self._center_omega_s), self._samples)
        self._power_s_f = np.zeros((1,)+self._shape_step_s)
        self._power_s_b = np.zeros((1,)+self._shape_step_s)
        self._power_s_ref = np.zeros(self._shape_step_s)
        self._power_ase_f = np.zeros((1,)+self._shape_step_s)
        self._power_ase_b = np.zeros((1,)+self._shape_step_s)
        self._shape_step_p = (len(self._center_omega_p), self._samples)
        self._power_p_f = np.zeros((1,)+self._shape_step_p)
        self._power_p_b = np.zeros((1,)+self._shape_step_p)
        self._power_p_ref = np.zeros(self._shape_step_p)
        self._N_1 = np.zeros(1)
        # Initiate refractive index ------------------------------------
        if (self._calc_n_core):
            self._n_0_s = self._calc_n_core.n(self._omega_s)
            self._n_0_p = self._calc_n_core.n(self._omega_p)
        else:
            self._n_0_s = np.ones(self._shape_step_s) * self._n_0_s_value
            self._n_0_p = np.ones(self._shape_step_p) * self._n_0_p_value
        if (self._calc_n_clad):
            NA_s = np.ones(self._shape_step_s) * self._NA_value_s
            NA_p = np.ones(self._shape_step_p) * self._NA_value_p
            self._n_clad_s = self._calc_n_clad(NA_s, self._n_0_s)
            self._n_clad_p = self._calc_n_clad(NA_p, self._n_0_p)
        else:
            self._n_clad_s = np.ones(self._shape_step_s) * self._n_clad_s_value
            self._n_clad_p = np.ones(self._shape_step_p) * self._n_clad_p_value
        self._n_tot_s = np.zeros(self._shape_step_s)
        self._n_tot_p = np.zeros(self._shape_step_p)
        # Initiate cross sections for each frequency -------------------
        self._A_eff_s = np.zeros(self._shape_step_s)
        self._Gamma_s = np.zeros(self._shape_step_s)
        for i in range(len(self._center_omega_s)):
            NA = NumericalAperture.calc_NA(self._n_0_s[i], self._n_clad_s[i])
            self._A_eff_s[i] = self._eff_area_s(self._omega_s[i], NA)
            self._Gamma_s[i] = self._overlap_s(self._A_eff_s[i])
        self._A_eff_p = np.zeros(self._shape_step_p)
        self._Gamma_p = np.zeros(self._shape_step_p)
        for i in range(len(self._center_omega_p)):
            NA = NumericalAperture.calc_NA(self._n_0_p[i], self._n_clad_p[i])
            self._A_eff_p[i] = self._eff_area_s(self._omega_p[i], NA)
            self._Gamma_p[i] = self._overlap_p(self._A_eff_p[i])
        # Initiate cross sections for each frequency -------------------
        if (self._absorp is not None):
            self._sigma_a_s = np.zeros(self._shape_step_s)
            self._sigma_a_p = np.zeros(self._shape_step_p)
            for i in range(len(self._center_omega_s)):
                self._sigma_a_s[i] =\
                    self._absorp.get_cross_section(self._omega_s[i])
            for i in range(len(self._center_omega_p)):
                self._sigma_a_p[i] =\
                    self._absorp.get_cross_section(self._omega_p[i])
        else:
            self._sigma_a_s = np.ones(self._shape_step_s)*self._sigma_a_s_value
            self._sigma_a_p = np.ones(self._shape_step_p)*self._sigma_a_p_value
        if (self._stimu is not None):
            self._sigma_e_s = np.zeros(self._shape_step_s)
            self._sigma_e_p = np.zeros(self._shape_step_p)
            for i in range(len(self._center_omega_s)):
                self._sigma_e_s[i] =\
                    self._stimu.get_cross_section(self._omega_s[i])
            for i in range(len(self._center_omega_p)):
                self._sigma_e_p[i] =\
                    self._stimu.get_cross_section(self._omega_p[i])
        else:
            if (self._sigma_e_mccumber):
                self._sigma_e_s = np.zeros(self._shape_step_s)
                self._sigma_e_p = np.zeros(self._shape_step_p)
                # must initiate here
            else:
                self._sigma_e_s = (np.ones(self._shape_step_s)
                                   * self._sigma_e_s_value)
                self._sigma_e_p = (np.ones(self._shape_step_p)
                                   * self._sigma_e_p_value)
        # Initiate counter ---------------------------------------------
        self._iter = -1 # -1 bcs increment in the first init. cond.
        self._call_counter = 0
        self._step = 0
        self._forward = True
    # ==================================================================
    def initial_condition(self, waves: Array[cst.NPFT], end: bool
                          ) -> Array[cst.NPFT]:
        r"""Initial conditions for shooting method.

        Parameters
        ----------
        waves :
            The propagating waves.
        end :
            If true, boundary counditions at z=L, else at z=0.

        Notes
        -----
        If co-propagating signal and pump:

        .. math:: \begin{split}
                    P_s^+(z=0) &= R_0 P_s^-(z=0) + P_{s,ref}^+(z=0)\\
                    P_p^+(z=0) &= R_0 P_p^-(z=0) + P_{p, ref}^+(z=0)\\
                    P_s^-(z=L) &= R_L P_s^+(z=L)\\
                    P_p^-(z=L) &= R_L P_s^+(z=L)
                  \end{split}

        If counter-propagating signal and pump:

        .. math:: \begin{split}
                    P_s^+(z=0) &= R_0 P_s^-(z=0) + P_{s,ref}^+(z=0)\\
                    P_p^+(z=0) &= R_0 P_p^-(z=0)\\
                    P_s^-(z=L) &= R_L P_s^+(z=L)\\
                    P_p^-(z=L) &= R_L P_s^+(z=L)  + P_{p, ref}^-(z=0)
                  \end{split}

        """
        # Update counter (per iter)-------------------------------------
        self._call_counter = 0
        self._iter += 1     # Began at -1
        if (not self._iter):   # Very first call
            self._coprop = not end
        self._forward = self._coprop ^ (self._iter%2 == 1)
        self._signal_on = ((self._iter > 0 and not self._coprop)
                           or (self._iter > 1 and self._coprop))
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
        if (self._iter):
            # Initial conditions ---------------------------------------
            if (end):   # At z = L
                if (self._iter > 1):
                    self._power_s_b[-1] = self._R_L * self._power_s_f[-1]
                else:
                    self._power_s_b[-1] = np.zeros_like(self._power_s_ref)
                self._power_p_b[-1] = self._R_L * self._power_p_f[-1]
                if (not self._coprop):
                    # TO DO: deal with counterprop
                    #dif = np.sum(self._power_p_ref)-np.sum(self._power_p_f[-1])
                    #factor = 10**(math.floor(math.log10(dif)))
                    #print(self._power_p_b.shape)
                    #to_add = math.copysign(factor, dif) / self._power_p_b.shape[2]
                    #print('to add', to_add, 'sign', dif)
                    #self._power_p_b[-1] += (to_add
                    #                        + self._power_p_ref)
                    self._power_p_b[-1] += self._power_p_ref
            else:   # at z = 0
                if (self._iter > 2):
                    self._power_s_f[0] = (self._R_0*self._power_s_b[0]
                                          + self._power_s_ref)
                # Initiate signal power --------------------------------
                elif (self._iter == 1 or self._iter == 2):   # First signal
                    self._power_s_f[0] = self._power_s_ref
                else:
                    self._power_s_f[0] = np.zeros_like(self._power_s_ref)
                self._power_p_f[0] = self._R_0 * self._power_p_b[0]
                if (self._coprop):
                    self._power_p_f[0] += self._power_p_ref
        else:   # First iteration
            # Initiate pump and signal power ---------------------------
            self._power_s_ref = self._get_power_s(waves)
            self._power_p_ref = self._get_power_p(waves)
            if (end):
                self._power_s_b[-1] = np.zeros_like(self._power_s_ref)
                self._power_p_b[-1] = self._power_p_ref
            else:
                self._power_s_f[0] = np.zeros_like(self._power_s_ref)
                self._power_p_f[0] = self._power_p_ref

        if (end):
            self._N_1[-1] = self._calc_N_1(-1)
        else:
            self._N_1[0] = self._calc_N_1(0)

        return waves
    # ==================================================================
    def set(self, waves: Array[cst.NPFT], h: float, z: float) -> None:
        self._step = int(round(z/h))
        self._call_counter += 1
        # Set parameters -----------------------------------------------
        if (self._sigma_e_mccumber):
            omega = FFT.ifftshift(self._omega)  # could also change in abstract equation
            for i in range(len(self._center_omega_s)):
                self._sigma_e_s[i] = McCumber.calc_cross_section_emission(
                    self._sigma_a_s[i], omega, 0.0,
                    self.N_0[self._step-1], self.N_1[self._step-1], self._T)
            for i in range(len(self._center_omega_p)):
                self._sigma_e_p[i] = McCumber.calc_cross_section_emission(
                    self._sigma_a_p[i], omega, 0.0,
                    self.N_0[self._step-1], self.N_1[self._step-1], self._T)
        # Add pump and signal power space step -------------------------
        if (not self._iter):    # first iteration
            to_add_s = np.zeros((1,)+self._shape_step_s)
            to_add_p = np.zeros((1,)+self._shape_step_p)
            self._power_s_f = np.vstack((self._power_s_f, to_add_s))
            self._power_s_b = np.vstack((to_add_s, self._power_s_b))
            self._power_p_f = np.vstack((self._power_p_f, to_add_p))
            self._power_p_b = np.vstack((to_add_p, self._power_p_b))
            self._power_ase_f = np.vstack((self._power_ase_f, to_add_s))
            self._power_ase_b = np.vstack((to_add_s, self._power_ase_b))
            if (self._coprop):  # First iteration forward
                self._N_1 = np.hstack((self._N_1, [0.0]))
            else:   # First iteration backward -> self._forward is False
                self._N_1 = np.hstack(([0.0], self._N_1))
                self._step = 0  # construct backward array in bottom-up
        # Set signal power ---------------------------------------------
        # Truth table:
        # if coprop :  iter |forward|   to do
        #              0    |   T   |   set pump
        #              1    |   F   |   set pump
        #              2    |   T   |   set pump & set signal
        # if counterprop:   iter|forward|   to do
        #                   0   |   T   |   set pump
        #                   1   |   F   |   set pump & set signal
        if (self._step_update):
            if (self._forward):
                if (self._signal_on):
                    self._power_s_f[self._step-1] = self._get_power_s(waves)
                self._power_p_f[self._step-1] = self._get_power_p(waves)
            else:
                if (self._signal_on):
                    self._power_s_b[self._step+1] = self._get_power_s(waves)
                self._power_p_b[self._step+1] = self._get_power_p(waves)
    # ==================================================================
    def __call__(self, waves: Array[cst.NPFT], h: float, z: float
                 ) -> Array[cst.NPFT]:

        h = h * 1e12   # km -> nm
        # Till first iter and counterprop or second iter and coprop,
        # the signal power are zeros.
        if (self._forward):
            self._power_ase_f[self._step] =\
                self._calc_power_ase(self._step-1, h, True)
            if (self._signal_on):
                self._power_s_f[self._step] =\
                    self._calc_power_s(self._step-1, h, True)
                self._power_s_f[self._step] += self._power_ase_f[self._step]
            self._power_p_f[self._step] =\
                self._calc_power_p(self._step-1, h, True)
        else:
            self._power_ase_b[self._step] =\
                self._calc_power_ase(self._step+1, h, False)
            if (self._signal_on):
                self._power_s_b[self._step] =\
                    self._calc_power_s(self._step+1, h, False)
                self._power_s_b[self._step] += self._power_ase_b[self._step]
            self._power_p_b[self._step] =\
                self._calc_power_p(self._step+1, h, False)

        self._N_1[self._step] = self._calc_N_1(self._step)

        return waves
    # ==================================================================
    def update(self, waves: Array[cst.NPFT], h: float, z: float) -> None:
        # Update the resonant and total index change -------------------
        N_1 = self._N_1[self._step]
        # Signal -------------------------------------------------------
        for i in range(len(self._center_omega_s)):
            n_0 = self._n_0_s[i]
            delta_n_s = self._res_index.n(self._omega_s[i], n_0, N_1)
            self._n_tot_s[i] = n_0 + delta_n_s
            NA = NumericalAperture.calc_NA(self._n_tot_s[i],
                                           self._n_clad_s[i])
            self._A_eff_s[i] = self._eff_area_s(self._omega_s[i], NA)
            self._Gamma_s[i] = self._overlap_s(self._A_eff_s[i])
        # Pump ---------------------------------------------------------
        for i in range(len(self._center_omega_p)):
            n_0 = self._n_0_p[i]
            delta_n_p = self._res_index.n(self._omega_p[i], n_0, N_1)
            self._n_tot_p[i] = n_0 + delta_n_p
            NA = NumericalAperture.calc_NA(self._n_tot_p[i],
                                           self._n_clad_p[i])
            self._A_eff_p[i] = self._eff_area_s(self._omega_p[i], NA)
            self._Gamma_p[i] = self._overlap_p(self._A_eff_p[i])
        # N.B.: A_eff in um^2 , must leave to calculate Gamma,
        # otherwise not enough precision (Gamma is adim. anyway)
    def close(self, domain: Domain, *fields: List[Field]) -> None:
        super().close(domain, *fields)
    # ==================================================================
    # Variable analytical expression ===================================
    # ==================================================================
    def _calc_N_1(self, step: int) -> Array[float]:
        # Numerator ----------------------------------------------------
        # Numerator signal ---------------------------------------------
        to_sum_s_f = self._power_s_f[step]*(self._sigma_a_s/self._omega_s)
        to_sum_s_b = self._power_s_b[step]*(self._sigma_a_s/self._omega_s)
        num_s = (np.sum(self._Gamma_s*to_sum_s_f)
                 + np.sum(self._Gamma_s*to_sum_s_b))
        # Numerator pump -----------------------------------------------
        to_sum_p_f = self._power_p_f[step]*(self._sigma_a_p/self._omega_p)
        to_sum_p_b = self._power_p_b[step]*(self._sigma_a_p/self._omega_p)
        num_p = (np.sum(self._Gamma_p*to_sum_p_f)
                 + np.sum(self._Gamma_p*to_sum_p_b))
        # Numerator final ----------------------------------------------
        num = self._N_T * (self._factor_s*num_s + self._factor_p*num_p)
        # Denominator --------------------------------------------------
        # Denominator signal -------------------------------------------
        to_sum_s_f += self._power_s_f[step]*(self._sigma_e_s/self._omega_s)
        to_sum_s_b += self._power_s_b[step]*(self._sigma_e_s/self._omega_s)
        den_s = (np.sum(self._Gamma_s*to_sum_s_f)
                 + np.sum(self._Gamma_s*to_sum_s_b))
        # Denominator pump ---------------------------------------------
        to_sum_p_f += self._power_p_f[step]*(self._sigma_e_p/self._omega_p)
        to_sum_p_b += self._power_p_b[step]*(self._sigma_e_p/self._omega_p)
        den_p = (np.sum(self._Gamma_p*to_sum_p_f)
                 + np.sum(self._Gamma_p*to_sum_p_b))
        # Denominator final --------------------------------------------
        den = (self._factor_s*den_s + self._factor_p*den_p) - self._decay

        return num / den
    # ==================================================================
    def _calc_power_s(self, step: int, z: float, forward: bool = True
                      ) -> Array[float]:

        exp_arg = ((self._Gamma_s*(self._sigma_a_s+self._sigma_e_s)
                    *self._N_1[step])
                   - (self._Gamma_s*self._N_T*self._sigma_a_s)
                   - self._eta_s)
        if (forward):

            return (self._power_s_f[step]*np.exp(1*exp_arg*z))
        else:

            return (self._power_s_b[step]*np.exp(-1*exp_arg*z))
    # ==================================================================
    def _calc_power_ase(self, step: int, z: float, forward: bool = True
                        ) -> Array[float]:

        ase = np.zeros(self._shape_step_s)
        for i in range(len(self._center_omega_s)):
            ase += (cst.H/(2*cst.PI**2)*self._Gamma_s*self._sigma_e_s
                    *self._N_1[step]*self._omega_s*self._width_omega_s[i]*z)
        ase *= 1e18  #  nm^2 kg ps^-3 -> W = m^2 kg s^-3
        if (forward):

            return ase
        else:

            return -1*ase
    # ==================================================================
    def _calc_power_p(self, step: int, z: float, forward: bool = True
                      ) -> Array[float]:

        exp_arg = ((self._Gamma_p*(self._sigma_a_p+self._sigma_e_p)
                    *self._N_1[step])
                   - (self._Gamma_p*self._N_T*self._sigma_a_p)
                   - self._eta_p)
        if (forward):

            return (self._power_p_f[step]*np.exp(1*exp_arg*z))
        else:

            return (self._power_p_b[step]*np.exp(-1*exp_arg*z))
    # ==================================================================
    # Gain and absorption ==============================================
    # ==================================================================
    def gain(self, step: int, order: int) -> Array[float]:
        """Return gain of the pump and signal. Fit calculated gain
        and predict main value and derivatives for center omegas.

        Parameters
        ----------
        step :
            The current step of the computation.
        order :
            The order of the taylor series expansion (number of deriv.).

        Returns
        -------
        :
            The gain of the fiber at the specified space step.
            :math:`[km^{-1}]`

        """
        res = np.zeros((len(self._center_omega), order+1))
        # N.B.: max order is 3 due to scipy spline function
        # Gain data for signal -----------------------------------------
        data_s = (self._Gamma_s
                  * ((self._sigma_a_s+self._sigma_e_s)*self._N_1[step]
                  - self._sigma_a_s*self._N_T) - self._eta_s)

        for j, omega in enumerate(self._center_omega_s):
            predict_s = interpolate.splrep(self._omega_s[j], data_s[j])
            for i in range(order+1):
                if (i > 3):
                    res[j][i] = 0.0
                else:
                    res[j][i] = interpolate.splev(omega, predict_s, der=i)
        # Gain data for pump -------------------------------------------
        data_p = (self._Gamma_p
                  * ((self._sigma_a_p+self._sigma_e_p)*self._N_1[step]
                    - self._sigma_a_p*self._N_T) - self._eta_p)

        for j, omega in enumerate(self._center_omega_p):
            predict_p = interpolate.splrep(self._omega_p[j], data_p[j])
            j += len(self._center_omega_s)
            for i in range(order+1):
                if (i > 3):
                    res[j][i] = 0.0
                else:
                    res[j][i] = interpolate.splev(omega, predict_p, der=i)

        return res*1e12 # nm^-1 -> km^-1
    # ==================================================================
    def absorption(self, step: int, order: int) -> Array[float]:
        """Return absorption of the pump and signal. Fit calculated
        absorption and predict main value and derivatives for center
        omegas.

        Parameters
        ----------
        step :
            The current step of the computation.
        order :
            The order of the taylor series expansion (number of deriv.).

        Returns
        -------
        :
            The absorption of the fiber at the specified space step.
            :math:`[km^{-1}]`

        """

        return -1*self.gain(step, order)
