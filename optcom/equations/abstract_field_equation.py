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

from __future__ import annotations

from typing import Callable, List, Optional, overload, Tuple, TYPE_CHECKING,\
                   Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.abstract_effect_taylor import AbstractEffectTaylor
from optcom.equations.abstract_equation import AbstractEquation
from optcom.field import Field
from optcom.solvers.ode_solver import ODESolver
from optcom.utils.callable_litt_expr import CallableLittExpr
from optcom.utils.fft import FFT
from optcom.utils.id_tracker import IdTracker
from optcom.utils.synchroniser import Synchroniser
# To avoid import cycles -> see
# https://mypy.readthedocs.io/en/stable/common_issues.html#import-cycles
if TYPE_CHECKING:   # only for type check, always false at runtime
    from optcom.components.abstract_solver import SOLVER_CALLABLE_TYPE


def sync_waves_decorator(func):
    """Synchronise the waves. The waves should be the first parameter of
    the method, and the wave id the second parameter of the method.
    """
    def func_wrapper(self, *args, **kwargs):
        if (self.is_master()):
            waves = args[0]
            wave_id = args[1]
            sync_waves = self.synchroniser.sync(waves, self.total_delays, wave_id)
            args_ = (sync_waves, ) + args[1:]
        else:
            args_ = args

        return func(self, *args_, **kwargs)

    return func_wrapper


class AbstractFieldEquation(AbstractEquation):
    r"""Parent class of any wave equation.

    Notes
    -----
    Considering :math:`l = 1, \ldots, M` equations with
    :math:`j = 1, \ldots, K_l` channels per equation, the
    AbstractFieldEquation class implement the following
    generic equations:

    if (self.SHARE_WAVES):

    .. math::
        \begin{split}
            \frac{\partial A_{lj}}{\partial z} &=
            \sum_{m = 1}^{M} C_{lm}^{lin} E_{lm}^{lin} \Big(\{A_{m1},
            \ldots, A_{mK_m}\} \cup \bigcup_{v\neq m}^{M}
            \{A_{v1}, \ldots, A_{v K_v}\}\Big)\\
            &\quad + \sum_{m = 1}^{M} C_{lm}^{nlin} E_{lm}^{nlin}
            \Big(\{A_{m1}, \ldots, A_{mK_m}\} \cup \bigcup_{v\neq m}^{M}
            \{A_{v1}, \ldots, A_{v K_v}\}\Big) \\
            &\quad + \sum_{m = 1}^{M} C_{lm}^{ind} E_{lm}^{ind}
            \Big(\{A_{m1}, \ldots, A_{mK_m}\} \cup \bigcup_{v\neq m}^{M}
            \{A_{v1}, \ldots, A_{v K_v}\}\Big) \\
            &\quad + \sum_{s=1}^{S_l} C_{ls}^{f} f_{ls} \Big(\{A_{l1},
            \ldots, A_{lK_l}\} \cup \bigcup_{v\neq l}^{M} \{A_{v1},
            \ldots, A_{v K_v}\}\Big) \qquad \forall j=1,\ldots,K_l
            \quad \forall l = 1,\ldots,M
          \end{split}

    else:

    .. math::
        \begin{split}
            \frac{\partial A_{lj}}{\partial z} &=
            C_{ll}^{lin} E_{ll}^{lin} \Big(\{A_{l1}, \ldots, A_{lK_l}\}
            \Big) + \sum_{m \neq l}^{M} C_{lm}^{lin} E_{lm}^{lin}
            \Big(\{A_{m1}, \ldots, A_{mK_m}\} \cup \{A_{lj}\}\Big)\\
            &\quad + C_{ll}^{nlin} E_{ll}^{nlin} \Big(\{A_{l1}, \ldots,
            A_{lK_l}\}\Big) +  \sum_{m \neq l}^{M} C_{lm}^{nlin}
            E_{lm}^{nlin} \Big(\{A_{m1}, \ldots, A_{mK_m}\
            \cup \{A_{lj}\}\Big) \\
            &\quad + C_{ll}^{ind} E_{ll}^{ind} \Big(\{A_{l1}, \ldots,
            A_{lK_l}\}\Big) +  \sum_{m \neq l}^{M} C_{lm}^{ind}
            E_{lm}^{ind} \Big(\{A_{m1}, \ldots, A_{mK_m}\}
            \cup \{A_{lj}\}\Big) \\
            &\quad + \sum_{s=1}^{S_l} C_{ls}^{f} f_{ls} (\{A_{l1},
            \ldots, A_{lK_l}\}) \qquad \forall j=1,\ldots,K_l
            \quad \forall l = 1,\ldots,M
          \end{split}

    where f is a list of length :math:`M` where each index :math:`m`
    contains :math:`S_m` sub-equations related to equation :math:`m`,
    :math:`E^{lin}` and :math:`E^{nlin}` are matrices of size
    :math:`M \times M` where each index :math:`ml` contains resp. linear
    and non-linear effects of equations :math:`m` w.r.t. to equation
    :math:`l`.

    """

    def __init__(self, nbr_eqs: int, prop_dir: List[bool], SHARE_WAVES: bool,
                 NOISE: bool, STEP_UPDATE: bool, INTRA_COMP_DELAY: bool,
                 INTRA_PORT_DELAY: bool, INTER_PORT_DELAY: bool) -> None:
        r"""
        Parameters
        ----------
        nbr_eqs :
            The number of equations in the system of equations.
        prop_dir :
            If [i] is True, waves in equation [i] are co-propagating
            relatively to equation [0]. If [i] is False, waves in
            equations [i] are counter-propagating relatively to
            equation [0]. prop_dir[0] must be True. len(prop_dir) must
            be equal to nbr_eqs.
        SHARE_WAVES :
            If True, pass all waves to each sub-equations. If False,
            pass only waves, from the fields which were in position i of
            the initial fields list, to equation i.
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

        """
        self._nbr_eqs: int = nbr_eqs
        # Bool flags ---------------------------------------------------
        self.SHARE_WAVES: bool = SHARE_WAVES
        self.STEP_UPDATE: bool = STEP_UPDATE
        self._synchroniser = Synchroniser(INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                                          INTER_PORT_DELAY)
        # Sub-equations and effects ------------------------------------
        self.__eqs: List[List[AbstractFieldEquation]] =\
            [[] for i in range(nbr_eqs)]
        self.__effects_lin: List[List[List[AbstractEffect]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self.__effects_non_lin: List[List[List[AbstractEffect]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self.__effects_ind: List[List[List[AbstractEffect]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self.__effects_delay: List[AbstractEffect] = []
        self.__effects_noise: List[List[SOLVER_CALLABLE_TYPE]] =\
            [[] for i in range(nbr_eqs)]
        # Sub-equations and effects multiplication constants -----------
        self.__eqs_mul_cst: List[List[float]] =\
            [[] for i in range(nbr_eqs)]
        self.__effects_lin_mul_cst: List[List[List[float]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self.__effects_non_lin_mul_cst: List[List[List[float]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self.__effects_ind_mul_cst: List[List[List[float]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self.__effects_delay_mul_cst: List[float] = []
        self.__effects_noise_mul_cst: List[List[float]] =\
            [[] for i in range(nbr_eqs)]
        # --------------------------------------------------------------
        self._call_counter: int = 0
        self._rep_freq: np.ndarray = np.array([])
        self._abs_omega: np.ndarray = np.array([])
        self._center_omega: np.ndarray = np.array([])
        self._omega: np.ndarray = np.array([])
        #self._step_delay: np.ndarray[float, nbr_channels]
        self._step_delay: np.ndarray = np.array([])
        # self._total_delays: np.ndarray[float, nbr_channels, steps]
        self._total_delays: np.ndarray = np.array([])
        # self._init_delays: np.ndarray[float, nbr_channels]
        self._init_delays: np.ndarray = np.array([])
        self._dtime: float = 0.
        self._domega: float = 0.
        self._id_tracker = IdTracker(nbr_eqs, prop_dir)
        # Noise --------------------------------------------------------
        self._noise_omega: np.ndarray = np.array([])
        self._noise_domega: float = 0.
        self._noise_samples: int = 0
        self._NOISE: bool = NOISE
        self._noise_func_per_field: List[CallableLittExpr] = []
        self.__master_eq: AbstractFieldEquation = self
    # ==================================================================
    def _update_variables(self):

        return None
    # ==================================================================
    @property
    def id_tracker(self):

        return self._id_tracker
    # ==================================================================
    @property
    def synchroniser(self):

        return self._synchroniser
    # ==================================================================
    # Delays management ================================================
    # ==================================================================
    @property
    def step_delay(self):

        return self._step_delay
    # ==================================================================
    @property
    def delays(self) -> np.ndarray:
        """Return the delays for each wave of the equation self
        at each step.
        """
        if (self.is_master()):

            return self._total_delays
        else:
            eq_id = self.eq_id_self()
            master_eq: AbstractFieldEquation = self.get_master()

            return master_eq.id_tracker.waves_in_eq_id(master_eq.delays, eq_id)
    # ==================================================================
    @property
    def total_delays(self) -> np.ndarray:
        """Return the delays for all waves at each step."""
        if (self.is_master()):

            return self._total_delays
        else:
            master_eq: AbstractFieldEquation = self.get_master()

            return master_eq.delays
    # ==================================================================
    def reset_delays(self) -> None:
        if (self.is_master()):
            self._total_delays = np.array([])
    # ==================================================================
    # Master equation management =======================================
    # ==================================================================
    def is_master(self) -> bool:

        return (self.__master_eq == self)
    # ==================================================================
    def set_master(self, eq: AbstractFieldEquation) -> None:

        self.__master_eq = eq
    # ==================================================================
    def get_master(self) -> AbstractFieldEquation:

        return self.__master_eq
    # ==================================================================
    def eq_id_self(self) -> int:
        """Return the equation id of the equation calling the method."""
        eq_id: int = -1
        i: int = 0
        master_eq: AbstractFieldEquation = self.get_master()
        while (eq_id < 0 and i < master_eq._nbr_eqs):
            for eq in master_eq._eqs[i]:
                if (eq == self):
                    eq_id = i
            i += 1

        return eq_id
    # ==================================================================
    # Sub-equations and efffects getter and adding =====================
    # ==================================================================
    def _add_eq(self, eq: AbstractFieldEquation, i: int, mul_cst: float = 1.0
                ) -> None:
        eq.set_master(self)
        self.__eqs[i].append(eq)
        self.__eqs_mul_cst[i].append(mul_cst)
    # ==================================================================
    def _add_lin_effect(self, effect: AbstractEffect, i: int, j: int,
                        mul_cst: float = 1.0) -> None:
        self.__effects_lin[i][j].append(effect)
        self.__effects_lin_mul_cst[i][j].append(mul_cst)
    # ==================================================================
    def _add_non_lin_effect(self, effect: AbstractEffect, i: int, j: int,
                            mul_cst: float = 1.0) -> None:
        self.__effects_non_lin[i][j].append(effect)
        self.__effects_non_lin_mul_cst[i][j].append(mul_cst)
    # ==================================================================
    def _add_ind_effect(self, effect: AbstractEffect, i: int, j: int,
                        mul_cst: float = 1.0) -> None:
        self.__effects_ind[i][j].append(effect)
        self.__effects_ind_mul_cst[i][j].append(mul_cst)
    # ==================================================================
    def _add_delay_effect(self, effect: AbstractEffect, mul_cst: float = 1.0
                          ) -> None:
        self.__effects_delay.append(effect)
        self.__effects_delay_mul_cst.append(mul_cst)
    # ==================================================================
    def _add_noise_effect(self, effect: Callable, i: int,
                          mul_cst: float = 1.0) -> None:
        self.__effects_noise[i].append(effect)
        self.__effects_noise_mul_cst[i].append(mul_cst)
    # ==================================================================
    @property
    def _eqs(self) -> List[List[AbstractFieldEquation]]:

        return self.__eqs
    # ==================================================================
    @property
    def _lin_effects(self) -> List[List[List[AbstractEffect]]]:

        return self.__effects_lin
    # ==================================================================
    @property
    def _lin_effects_mul_cst(self) -> List[List[List[float]]]:

        return self.__effects_lin_mul_cst
    # ==================================================================
    @property
    def _non_lin_effects(self) -> List[List[List[AbstractEffect]]]:

        return self.__effects_non_lin
    # ==================================================================
    @property
    def _non_lin_effects_mul_cst(self) -> List[List[List[float]]]:

        return self.__effects_non_lin_mul_cst
    # ==================================================================
    @property
    def _ind_effects(self) -> List[List[List[AbstractEffect]]]:

        return self.__effects_ind
    # ==================================================================
    @property
    def _ind_effects_mul_cst(self) -> List[List[List[float]]]:

        return self.__effects_ind_mul_cst
    # ==================================================================
    @property
    def _delay_effects(self) -> List[AbstractEffect]:

        return self.__effects_delay
    # ==================================================================
    @property
    def _delay_effects_noise_cst(self) -> List[float]:

        return self.__effects_delay_mul_cst
    # ==================================================================
    @property
    def _noise_effects(self) -> List[List[Callable]]:

        return self.__effects_noise
    # ==================================================================
    @property
    def _noise_effects_mul_cst(self) -> List[List[float]]:

        return self.__effects_noise_mul_cst
    # ==================================================================
    # Open, Close, Set, Update =========================================
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        """This function is called once before a stepper began the
        computation. Pass the time, wavelength and center wavelength
        to all the effects in the equation.
        """
        # Counters and flags -------------------------------------------
        self._call_counter = 0
        # Sub-equations open -------------------------------------------
        for i in range(len(self.__eqs)):
            for eq in self.__eqs[i]:
                eq.open(domain, fields[i])
        self._id_tracker.initialize(list(fields))
        # Attributes definitions ---------------------------------------
        self._center_omega = np.array([])
        self._abs_omega = np.zeros((0, domain.samples))
        self._rep_freq = np.array([])
        for field_list in fields:
            for field in field_list:
                self._center_omega = np.hstack((self._center_omega,
                                                field.center_omega))
                self._abs_omega = np.vstack((self._abs_omega,
                                             FFT.fftshift(field.omega)))
                self._rep_freq = np.hstack((self._rep_freq,
                                            field.rep_freq))
        self._omega = FFT.fftshift(domain.omega)
        self._domega = domain.domega
        self._time = domain.time
        self._dtime = domain.dtime
        effects = ["lin", "non_lin", "ind"]
        for effect in effects:
            effect_list = getattr(self, "_{}_effects".format(effect))
            for i in range(len(effect_list)):
                for j in range(len(effect_list[i])):
                    for k in range(len(effect_list[i][j])):
                        effect_list[i][j][k].omega = self._omega
                        effect_list[i][j][k].time = self._time
                        effect_list[i][j][k].domega = self._domega
                        effect_list[i][j][k].dtime = self._dtime
                        effect_list[i][j][k].rep_freq = self._rep_freq
        # Init variables -----------------------------------------------
        if (not self.STEP_UPDATE): # otherwise do it at each step in set()
            self._update_variables()
        # Delays -------------------------------------------------------
        self._init_delays = np.zeros(0)
        for field_list in fields:
            for field in field_list:
                self._init_delays = np.hstack((self._init_delays,
                                               field.delays))
        self._step_delay = np.zeros(len(self._center_omega))
        if (self.is_master()):
            self._total_delays = np.zeros((len(self._center_omega), 0))
        self._synchroniser.initialize(self._init_delays, self._rep_freq,
                                      self._id_tracker, self._dtime)
        # Noise --------------------------------------------------------
        self._noise_omega = domain.noise_omega
        self._noise_domega = domain.noise_domega
        self._noise_samples = domain.noise_samples
    # ==================================================================
    def close(self, domain: Domain, *fields: List[Field]) -> None:
        """This function is called once after a stepper ended the
        computation.
        """
        # Sub-equations close ------------------------------------------
        for i in range(len(self.__eqs)):
            for eq in self.__eqs[i]:
                eq.close(domain, fields[i])
        # Delays -------------------------------------------------------
        if (self.is_master() and fields):
            ind: int = 0
            # total_delays contains the total delay at each step
            delays: np.ndarray = self._total_delays[:,-1]
            for field_list in fields:
                for field in field_list:
                    field.add_delay(delays[ind:ind+len(field)])
                    ind += len(field)
        # Clear attributes ---------------------------------------------
        self.clear()
    # ==================================================================
    def set(self, waves: np.ndarray, noises: np.ndarray, z: float, h: float
            ) -> None:
        """This function is called before each step of the computation.
        """
        self._call_counter += 1
        # Sub-equations set --------------------------------------------
        for i in range(len(self.__eqs)):
            for eq in self.__eqs[i]:
                eq.set(self.id_tracker.waves_in_eq_id(waves, i),
                       self.id_tracker.fields_in_eq_id(noises, i), z, h)
        # Update variables ---------------------------------------------
        if (self.STEP_UPDATE):
            self._update_variables()
        # Delays -------------------------------------------------------
        # Initialize self._step_delay and self._total_delays arrays
        self._step_delay = np.zeros_like(self._step_delay)
        if (self.is_master()):
            if (not self._total_delays.size):
                self._total_delays = np.zeros((len(self._center_omega), 0))
            to_add: np.ndarray = np.zeros((len(self._center_omega), 1))
            self._total_delays = np.hstack((self._total_delays, to_add))
        # Perform change of variable T = t - \beta_1 z to ignore GV
        if (self.__effects_delay):  # if GV like coefficients
            for i in range(len(waves)):
                for effect in self.__effects_delay:
                    delay_factors = effect.delay_factors(i)
                    for delay_factor in delay_factors:
                        self._step_delay[i] += delay_factor * h
        # Noise --------------------------------------------------------
        if (self._NOISE and self.is_master()):
            if (not self._noise_func_per_field):
                self._initiate_noise_managament()
    # ==================================================================
    def update(self, waves: np.ndarray, noises: np.ndarray, z: float, h: float
               ) -> None:
        """This function is called after each step of the computation.
        """
        # Sub-equations update -----------------------------------------
        for i in range(len(self.__eqs)):
            for eq in self.__eqs[i]:
                eq.update(self.id_tracker.waves_in_eq_id(waves, i),
                          self.id_tracker.fields_in_eq_id(noises, i), z, h)
        # Delays -------------------------------------------------------
        if (self.is_master()):
            # Total delays = last delays of self + current delays of
            # self + current delays of all sub equations
            if (self._total_delays.shape[1] > 1):
                self._total_delays[:,-1] = self._total_delays[:,-2]
            self._total_delays[:,-1] += self._step_delay
            # Adding up delays per step of all sub equations for step z/h
            ind: int = 0
            len_eq: int = 0
            for i in range(len(self.__eqs)):
                for eq in self.__eqs[i]:
                    len_eq = self.id_tracker.nbr_waves_in_eq(i)
                    self._total_delays[ind:ind+len_eq,-1] += eq.step_delay
                    ind += len_eq
    # ==================================================================
    def pre_pass(self) -> None:
        """This function is called before a pass through the component,
        i.e. before the call to the step method in the stepper.
        """
        self.reset_delays()
    # ==================================================================
    def post_pass(self) -> None:
        """This function is called after a pass through the component,
        i.e. after the call to the step method in the stepper.
        """

        return None
    # ==================================================================
    @overload
    def calc_noise(self, noises: np.ndarray, z: float, h: float
                 )-> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def calc_noise(self, noises: np.ndarray, z: float, h: float, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def calc_noise(self, *args):
        """Compute noise."""
        if (len(args) == 4):
            noises, z, h, ind = args
            if (self._NOISE):
                noises_ = np.zeros_like(noises[ind])
                eq_id = self.id_tracker.eq_id_of_field_id(ind)
                noises_ = self._noise_func_per_field[eq_id](noises, z, h, ind)

                return noises_
            else:

                return noises[ind]
        elif (len(args) == 3):
            # For now, dummy identity function for equations that do not
            # implement any noise, might want to find another solution
            noises, z, h = args
            noises_ = np.zeros_like(noises)
            for i in range(len(noises)):
                eq_id = self.id_tracker.eq_id_of_field_id(i)
                noises_[i] = self._noise_func_per_field[eq_id](noises, z, h, i)

            return noises_
        else:

            raise NotImplementedError()
    # ==================================================================
    def _initiate_noise_managament(self) -> None:
        """Initiate noise manangement. Link each noise array to the
        appropriate noise effects.
        """
        crt_noise_effects: List[Callable]
        crt_noise_effects_mul_cst: List[float]
        for i in range(self._nbr_eqs):
            crt_noise_effects = self._noise_effects[i]
            crt_noise_effects_mul_cst = self._noise_effects_mul_cst[i]
            for eq in self.__eqs[i]:
                for j in range(len(eq._noise_effects)):
                    crt_noise_effects.extend(eq._noise_effects[j])
                    crt_noise_effects_mul_cst.extend(
                        eq._noise_effects_mul_cst[j])
            if (crt_noise_effects):
                effects_to_add = []
                for k in range(len(crt_noise_effects)):
                    effects_to_add.append(CallableLittExpr(
                        [crt_noise_effects[k], crt_noise_effects_mul_cst[k]],
                        ['*']))
                crt_ops = ['+' for _ in range(len(crt_noise_effects)-1)]
                noise_fct = CallableLittExpr(effects_to_add, crt_ops)
            else:
                noise_fct = CallableLittExpr([lambda noises, z, h, ind:
                                              noises[ind]])
            self._noise_func_per_field.append(noise_fct)
    # ==================================================================
    def clear(self) -> None:
        self._call_counter = 0
        self._rep_freq = np.array([])
        self._abs_omega = np.array([])
        self._center_omega = np.array([])
        self._omega = np.array([])
        self._step_delay = np.array([])
        self._total_delays = np.array([])
        self._init_delays = np.array([])
        self._dtime = 0.
        self._domega = 0.
        self._noise_omega = np.array([])
        self._noise_domega = 0.
        self._noise_samples = 0
        self._noise_func_per_field = []
        self.__effects_delay_mul_cst = []
        self.__effects_noise = [[] for i in range(self._nbr_eqs)]
        self.__effects_noise_mul_cst = [[] for i in range(self._nbr_eqs)]
    # ==================================================================
    # Effects variables management =====================================
    # ==================================================================
    def _clear_var_lin(self):
        return None
    def _clear_var_non_lin(self):
        return None
    def _clear_var_ind(self):
        return None
    # ==================================================================
    @staticmethod
    def exp(wave: np.ndarray, h: float) -> np.ndarray:
        """Exponential function.

        Parameters
        ----------
        wave :
            The wave vector
        h :
            The spatial step

        Returns
        -------
        :
            Exponential of the wave multiplied by the spatial step

        Notes
        -----

        .. math:: \exp(h\hat{D})

        """

        return np.exp(h*wave, dtype=cst.NPFT)
    # ==================================================================
    # Operators ========================================================
    # ==================================================================
    def _expr(self, op_or_term: str, effect_type: str, waves: np.ndarray,
              id: int, corr_wave: np.ndarray) -> np.ndarray:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type).
        """

        return (self._expr_sub(op_or_term, effect_type, waves, id, corr_wave)
                +self._expr_main(op_or_term, effect_type, waves, id, corr_wave)
                )
    # ==================================================================
    def _expr_main(self, op_or_term: str, effect_type: str,
                   waves: np.ndarray, id: int, corr_wave: np.ndarray
                   ) -> np.ndarray:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type) for effect lists in self."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        delays = self.total_delays
        eq_id = self.id_tracker.eq_id_of_wave_id(id)
        rel_wave_id = self.id_tracker.rel_wave_id(id)
        # Operators from current coupled equation ----------------------
        effect_eq_id = getattr(self, "_{}_effects".format(effect_type))[eq_id]
        effect_eq_id_mul_cst = getattr(self, "_{}_effects_mul_cst"
                                             .format(effect_type))[eq_id]
        if (self.SHARE_WAVES):
            for i in range(len(effect_eq_id)):
                crt_len_eq = self.id_tracker.nbr_waves_in_eq(i)
                if (crt_len_eq):    # If waves in considered eq.
                    # Pass waves_in_eq_id(waves, i) + rest of waves
                    # Operators and terms expect crt_wave to be first
                    crt_waves = np.vstack((
                        self.id_tracker.waves_in_eq_id(waves, i),
                        self.id_tracker.waves_out_eq_id(waves, i)))
                    # Get new position of waves[id] in crt_waves
                    if (eq_id == i):
                        crt_wave_id = rel_wave_id
                    elif (eq_id > i):
                        crt_wave_id = id
                    else:
                        crt_wave_id = crt_len_eq + id
                    # Propagate to all effects
                    for j, effect in enumerate(effect_eq_id[i]):
                        res += (effect_eq_id_mul_cst[i][j]
                                * getattr(effect, op_or_term)(crt_waves,
                                                              crt_wave_id,
                                                              corr_wave))
        else:
            for i in range(len(effect_eq_id)):
                crt_len_eq = self.id_tracker.nbr_waves_in_eq(i)
                if (crt_len_eq):    # If waves in considered eq.
                    # Pass waves_in_eq_id(waves, i) + the crt wave as last elem
                    if (eq_id == i):
                        crt_waves = self.id_tracker.waves_in_eq_id(waves, i)
                    else:
                        crt_waves = self.id_tracker.waves_in_eq_id(waves, i)
                        crt_waves = np.vstack((crt_waves, waves[id]))
                    # Get new position of waves[id] in crt_waves
                    if (eq_id == i):
                        crt_wave_id = rel_wave_id
                    else:
                        crt_wave_id = -1
                    # Propagate to all effects
                    for j, effect in enumerate(effect_eq_id[i]):
                        res += (effect_eq_id_mul_cst[i][j]
                                * getattr(effect, op_or_term)(crt_waves,
                                                              crt_wave_id,
                                                              corr_wave))
        # Clearing
        getattr(self, "_clear_var_{}".format(effect_type))

        return res
    # ==================================================================
    def _expr_sub(self, op_or_term: str, effect_type: str,
                  waves: np.ndarray, id: int, corr_wave: np.ndarray
                  ) -> np.ndarray:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type) for equations in self.__eqs."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        delays = self.total_delays
        eq_id = self.id_tracker.eq_id_of_wave_id(id)
        rel_wave_id = self.id_tracker.rel_wave_id(id)
        if (self.SHARE_WAVES):
            crt_waves = np.vstack((
                self.id_tracker.waves_in_eq_id(waves, eq_id),
                self.id_tracker.waves_out_eq_id(waves, eq_id)))
        else:
            crt_waves = self.id_tracker.waves_in_eq_id(waves, eq_id)
        # Operators from sub-equations ---------------------------------
        for i, eq in enumerate(self.__eqs[eq_id]):
            res += (self.__eqs_mul_cst[eq_id][i]
                    * getattr(eq, "{}_{}".format(op_or_term, effect_type)
                              )(crt_waves, rel_wave_id, corr_wave))

        return res
    # ==================================================================
    @sync_waves_decorator
    def op_lin(self, waves: np.ndarray, id: int,
               corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear operator of the equation."""

        return self._expr("op", "lin", waves, id, corr_wave)
    # ==================================================================
    @sync_waves_decorator
    def op_non_lin(self, waves: np.ndarray, id: int,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        """Non linear operator of the equation."""

        return self._expr("op", "non_lin", waves, id, corr_wave)
    # ==================================================================
    @sync_waves_decorator
    def op_ind(self, waves: np.ndarray, id: int,
               corr_wave: Optional[np.ndarray] = None
               ) -> np.ndarray:
        """General operator of the equation."""

        return self._expr("op", "ind", waves, id, corr_wave)
    # ==================================================================
    def exp_op_lin(self, waves: np.ndarray, id: int, h: float,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:

        return self.exp(self.op_lin(waves, id, corr_wave), h)
    # ==================================================================
    def exp_op_non_lin(self, waves: np.ndarray, id: int, h: float,
                       corr_wave: Optional[np.ndarray] = None
                       ) -> np.ndarray:

        return self.exp(self.op_non_lin(waves, id, corr_wave), h)
    # ==================================================================
    def exp_op_ind(self, waves: np.ndarray, id: int, h: float,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:

        return self.exp(self.op_ind(waves, id, corr_wave), h)
    # ==================================================================
    # Terms ============================================================
    # ==================================================================
    @sync_waves_decorator
    def term_lin(self, waves: np.ndarray, id: int, z: float,
                 corr_wave: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """Linear term of the equation."""

        return self._expr("term", "lin", waves, id, corr_wave)
    # ==================================================================
    @sync_waves_decorator
    def term_non_lin(self, waves: np.ndarray, id: int, z: float,
                     corr_wave: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        """Non linear term of the equation."""
        if (corr_wave is None):
            corr_wave = waves[id]

        return self._expr("term", "non_lin", waves, id, corr_wave)
    # ==================================================================
    @sync_waves_decorator
    def term_ind(self, waves: np.ndarray, id: int, z: float,
                 corr_wave: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """General term of the equation."""

        return self._expr("term", "ind", waves, id, corr_wave)
    # ==================================================================
    def exp_term_lin(self, waves: np.ndarray, id: int, z: float, h: float,
                     corr_wave: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        if (corr_wave is None):
            corr_wave = waves[id]

        return FFT.ifft_mult_fft(corr_wave,
                                 self.exp_op_lin(waves, id, h, corr_wave))
    # ==================================================================
    def exp_term_non_lin(self, waves: np.ndarray, id: int, z: float,
                         h: float, corr_wave: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        if (corr_wave is None):
            corr_wave = waves[id]

        return  self.exp_op_non_lin(waves, id, h, corr_wave) * corr_wave
    # ==================================================================
    def exp_term_ind(self, waves: np.ndarray, id: int, z: float,
                     h: float, corr_wave: Optional[np.ndarray] = None
                     ) -> np.ndarray:

        return np.zeros(waves[id].shape, dtype=cst.NPFT)
