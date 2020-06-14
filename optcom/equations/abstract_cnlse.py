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

import copy
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.asymmetry import Asymmetry
from optcom.effects.coupling import Coupling
from optcom.effects.kerr import Kerr
from optcom.effects.raman import Raman
from optcom.effects.raman_approx import RamanApprox
from optcom.equations.abstract_field_equation import AbstractFieldEquation
from optcom.equations.abstract_field_equation import sync_waves_decorator
from optcom.field import Field
from optcom.parameters.fiber.asymmetry_coeff import AsymmetryCoeff
from optcom.parameters.fiber.coupling_coeff import CouplingCoeff
from optcom.parameters.refractive_index.sellmeier import Sellmeier


TAYLOR_COEFF_TYPE = List[Union[List[float], Callable]]
TAYLOR_COUP_COEFF = List[List[Union[List[float], Callable, None]]]


class AbstractCNLSE(AbstractFieldEquation):
    """Coupled non linear Schrodinger equations.

    Represent the different effects in the NLSE as well as the
    interaction of NLSEs propagating along each others. Note that
    automatic calculation of the coupling coefficients rely on a formula
    that is only correct for symmetric coupler.

    Attributes
    ----------
    nbr_eqs : int
        Number of NLSEs in the CNLSE.

    """

    def __init__(self, nbr_fibers: int,
                 beta: TAYLOR_COEFF_TYPE, kappa: TAYLOR_COUP_COEFF,
                 sigma_cross: List[List[float]], eta_cross: List[List[float]],
                 core_radius: List[float], clad_radius: float,
                 c2c_spacing: List[List[float]],
                 n_clad: Union[float, Callable],
                 v_nbr: List[Union[float, Callable]], temperature: float,
                 ASYM: bool, COUP: bool, XPM: bool, FWM: bool, XNL: bool,
                 NOISE: bool, STEP_UPDATE: bool, INTRA_COMP_DELAY: bool,
                 INTRA_PORT_DELAY: bool, INTER_PORT_DELAY: bool,
                 kerr_effects: List[Optional[Kerr]],
                 raman_effects: List[Optional[Union[Raman, RamanApprox]]]
                 ) -> None:
        r"""
        Parameters
        ----------
        nbr_fibers :
            The number of fibers in the coupler.
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]`
        kappa :
            The coupling coefficients. :math:`[km^{-1}]`
        sigma_cross :
            Positive term multiplying the XPM terms of the NLSE
            inbetween the fibers.
        eta_cross :
            Positive term multiplying the cross-non-linear terms of the
            NLSE inbetween the fibers.
        core_radius :
            The core radius. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        c2c_spacing :
            The center to center distance between two cores.
            :math:`[\mu m]`
        n_clad :
            The refractive index of the clading.  If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        v_nbr :
            The V number.  If a callable is provided, variable must be
            angular frequency. :math:`[ps^{-1}]`
        temperature :
            The temperature of the fiber. :math:`[K]`
        ASYM :
            If True, trigger the asymmetry effects between cores.
        COUP :
            If True, trigger the coupling effects between cores.
        XPM :
            If True, trigger the cross-phase modulation.
        FWM :
            If True, trigger the Four-Wave mixing.
        XNL :
            If True, trigger cross-non linear effects.
        NOISE :
            If True, trigger the noise calculation.
        STEP_UPDATE :
            If True, update component parameters at each spatial
            space step by calling the _update_variables method.
        INTRA_COMP_DELAY :
            If True, take into account the relative time difference,
            between all waves, that is acquired while propagating
            in the component.
        INTRA_PORT_DELAY :
            If True, take into account the initial relative time
            difference between channels of all fields but for each port.
        INTER_PORT_DELAY :
            If True, take into account the initial relative time
            difference between channels of all fields of all ports.
        kerr_effects :
            A list with a Kerr effect object for each fiber.
        raman_effects :
            A list with a Raman effect object for each fiber.

        """
        super().__init__(nbr_eqs=nbr_fibers, SHARE_WAVES=False,
                         prop_dir=[True for i in range(nbr_fibers)],
                         NOISE=NOISE, STEP_UPDATE=STEP_UPDATE,
                         INTRA_COMP_DELAY=INTRA_COMP_DELAY,
                         INTRA_PORT_DELAY=INTRA_PORT_DELAY,
                         INTER_PORT_DELAY=INTER_PORT_DELAY)
        kerr_effects_ = util.make_list(kerr_effects, nbr_fibers)
        raman_effects_ = util.make_list(raman_effects, nbr_fibers)
        self._asym: List[List[Asymmetry]] =\
            [[] for i in range(nbr_fibers)]
        self._coup: List[List[Coupling]] =\
            [[] for i in range(nbr_fibers)]
        self._raman: List[List[Union[Raman, RamanApprox]]] =\
            [[] for i in range(nbr_fibers)]
        kappa_: List[List[Union[List[float], Callable]]] =\
            [[[] for i in range(nbr_fibers)] for i in range(nbr_fibers)]
        beta_: Union[List[float], Callable]
        beta_1_: Union[float, Callable]
        beta_2_: Union[float, Callable]
        crt_kerr: Optional[Kerr]
        crt_raman: Optional[Union[Raman, RamanApprox]]
        coup_coeff_: Union[List[float], Callable]
        for i in range(nbr_fibers):
            for j in range(nbr_fibers):
                if (i != j):
                    if (XPM or FWM):
                        crt_kerr = copy.deepcopy(kerr_effects_[i])
                        if (crt_kerr is not None):
                            crt_kerr.SPM = False
                            crt_kerr.XPM = XPM
                            crt_kerr.FWM = FWM
                            crt_kerr.sigma = sigma_cross[i][j]
                            self._add_non_lin_effect(crt_kerr, i, j)
                    if (XNL):
                        crt_raman = copy.deepcopy(raman_effects_[i])
                        if (crt_raman is not None):
                            crt_raman.self_term = False
                            crt_raman.cross_term = True
                            crt_raman.eta = eta_cross[i][j]
                            self._raman[i].append(crt_raman)
                            self._add_non_lin_effect(crt_raman, i, j)
                    if (ASYM):
                        beta_ = beta[i] if callable(beta[i]) else beta[i]
                        beta_1_ = beta_ if callable(beta_) else beta_[0]
                        beta_ = beta[j] if callable(beta[j]) else beta[j]
                        beta_2_ = beta_ if callable(beta_) else beta_[0]
                        delta_a = AsymmetryCoeff(beta_1=beta_1_,
                                                 beta_2=beta_2_)
                        self._asym[i].append(Asymmetry(delta_a))
                        self._add_lin_effect(self._asym[i][-1], i, j)
                    if (COUP):
                        crt_kappa = kappa[i][j]
                        # not kappa_[i][j] bcs symmetric if self constructing
                        if (not kappa_[i][j]):
                            if (crt_kappa is None):
                                coup_coeff_ = CouplingCoeff(v_nbr=v_nbr[i],
                                    a=core_radius[i], d=c2c_spacing[i][j],
                                    ref_index=n_clad)
                                kappa_[j][i] = coup_coeff_
                            else:
                                coup_coeff_ = crt_kappa
                            kappa_[i][j] = coup_coeff_
                        self._coup[i].append(Coupling(kappa_[i][j]))
                        self._add_ind_effect(self._coup[i][-1], i, j)
    # ==================================================================
    @overload
    def __call__(self, waves: np.ndarray, z: float, h: float
                 )-> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, waves: np.ndarray, z: float, h: float, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, *args):
        if (len(args) == 4):
            waves, z, h, ind = args
            res = np.zeros_like(waves[ind], dtype=cst.NPFT)
            res = self.term_ind(waves, ind, z)

            return res
        else:

            raise NotImplementedError()
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        super().open(domain, *fields)
        for i in range(len(self._asym)):
            for asym in self._asym[i]:
                asym.set(self.id_tracker.waves_in_eq_id(self._center_omega, i))
        for i in range(len(self._coup)):
            for coup in self._coup[i]:
                coup.set(self.id_tracker.waves_in_eq_id(self._center_omega,i))
        for i in range(len(self._raman)):
            for raman in self._raman[i]:
                raman.set()
    # ==================================================================
    def get_kappa_for_noise(self):
        # temp harcoding for noise manangement
        # + will fail if COUP is False
        return self._coup_coeff[0]  # sym assumption, both are equal anyway
    # ==================================================================
    def close(self, domain: Domain, *fields: List[Field]) -> None:
        # tem hardcoding -> must change
        self._coup_coeff: List[Callable] = [] # len == 2
        if (self._nbr_eqs == 2):    # hardcoding for 2 -> need to generalize
            for i in range(len(self._coup)): # len == 2
                for coup in self._coup[i]:  # len == 1
                    self._coup_coeff.append(coup.kappa(self._noise_omega)[0])
        # Call to part
        super().close(domain, *fields)
    # ==================================================================
    @sync_waves_decorator
    def op_non_lin(self, waves: np.ndarray, id: int,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        """Non linear operator of the equation."""
        eq_id = self.id_tracker.eq_id_of_wave_id(id)
        rel_wave_id = self.id_tracker.rel_wave_id(id)
        gamma = self._eqs[eq_id][0].gamma[rel_wave_id]

        return (gamma * self._expr_main("op", "non_lin", waves, id, corr_wave)
                + self._expr_sub("op", "non_lin", waves, id, corr_wave))
    # ==================================================================
    @sync_waves_decorator
    def term_non_lin(self, waves: np.ndarray, id: int,
                     corr_wave: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        """Non linear operator of the equation."""
        eq_id = self.id_tracker.eq_id_of_wave_id(id)
        rel_wave_id = self.id_tracker.rel_wave_id(id)
        gamma = self._eqs[eq_id][0].gamma[rel_wave_id]

        return (gamma*self._expr_main("term", "non_lin", waves, id, corr_wave)
                +self._expr_sub("term", "non_lin", waves, id, corr_wave))
