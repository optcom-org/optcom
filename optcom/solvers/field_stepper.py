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
import math
import time
import warnings
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_field_equation import AbstractFieldEquation
from optcom.equations.boundary_conditions.abstract_boundary_conditions import\
    AbstractBoundaryConditions
from optcom.equations.convergence_checker.abstract_convergence_checker import\
    AbstractConvergenceChecker
from optcom.field import Field
from optcom.solvers.abstract_solver import AbstractSolver
from optcom.solvers.abstract_solver import SOLVER_CALLABLE_TYPE
from optcom.utils.storage import Storage


# Typing variables
STEP_METHOD_TYPE = Callable[[np.ndarray, np.ndarray, List[AbstractSolver],
                             List[Optional[AbstractSolver]],
                             List[SOLVER_CALLABLE_TYPE], int, bool, bool],
                            Tuple[np.ndarray, np.ndarray]]

STEPPER_METHOD_TYPE =  Callable[[np.ndarray, np.ndarray, List[AbstractSolver],
                                 List[Optional[AbstractSolver]],
                                 List[SOLVER_CALLABLE_TYPE], int,
                                 STEP_METHOD_TYPE],
                                Tuple[np.ndarray, np.ndarray]]


# Step methods
FIXED: str = "fixed"
ADAPTATIVE: str = "adaptative"
STEP_METHODS: List[str] = [FIXED, ADAPTATIVE]
# Solvers order method
ALTERNATING: str = "alternating"
FOLLOWING: str = "following"
SOLVER_ORDERS: List[str] = [ALTERNATING, FOLLOWING]
# Stepper methods
FORWARD: str = "forward"
BACKWARD: str = 'backward'
SHOOTING: str  = "shooting"
STEPPER_METHODS: List[str] = [FORWARD, BACKWARD, SHOOTING]


# Exceptions
class FieldStepperError(Exception):
    pass

class EquationTypeError(FieldStepperError):
    pass

class MissingInfoError(FieldStepperError):
    pass

class MemoryError(FieldStepperError):
    pass

class FieldStepperWarning(UserWarning):
    pass

class MemorySpaceWarning(FieldStepperWarning):
    pass


# Decorators
def step_method_decorator(step_method):
    """Decorator to execute some memory and AbstractFieldEquation
    routine before computation of one pass through a step method.
    """
    def func_wrapper(self, waves, noises, solvers, noise_solvers, eqs, steps,
                     forward, bidir):
        # Memory management --------------------------------------------
        if ((bidir or self.save_all) and not self._channels.size):
            self._enough_space, self._save_step = self._memory_management(
                waves, noises, steps, bidir)
            if (bidir and (not self._enough_space or self._save_step != 1)):
                raise MemoryError("There is not enough memory to compute "
                    "bidirectional style computation.")
        # AbstractFieldEquation management -----------------------------
        for eq in eqs:
            if (isinstance(eq, AbstractFieldEquation)):
                eq.pre_pass()

        res = step_method(self, waves, noises, solvers, noise_solvers, eqs,
                          steps, forward, bidir)

        for eq in eqs:
            if (isinstance(eq, AbstractFieldEquation)):
                eq.post_pass()

        return res

    return func_wrapper


class FieldStepper(object):
    """Compute equations by managing step size and calling Solver.

    Attributes
    ----------
    save_all : int
        If True, will save all channels of all fields at each space
        step during the equation resolution. The number of step
        recorded depends on the :attr:`memory_storage` attribute
        of :class:`layout.Layout`.

    """

    def __init__(self, solvers: List[AbstractSolver],
                 noise_solvers: List[Optional[AbstractSolver]] = [None],
                 length: float = 1.0, steps: List[int] = [100],
                 step_method: List[str] = [FIXED],
                 solver_order: str = FOLLOWING,
                 stepper_method: List[str] = [FORWARD],
                 solver_sequence: Optional[List[int]] = None,
                 boundary_cond: Optional[AbstractBoundaryConditions] = None,
                 conv_checker: Optional[AbstractConvergenceChecker] = None,
                 save_all: bool = False) -> None:
        r"""
        Parameters
        ----------
        solvers :
            The solvers used to solve the equations.
        nlse_solvers :
            The solvers used to solve the noise equations.
        length :
            The length over which the equation is considered.
            :math:`[km]`
        steps :
            The number of steps used for the computation.
        step_method :
            The method used to update the step size.
        solver_order :
            The order type in which the methods will be computed. (will
            be ignored if solver_sequence is provided)
        stepper_method :
            The method used to converge to solution.
        solver_sequence :
            The sequence in which the methods will be computed.
        boundary_cond : AbstractBoundaryConditions
            The boundaries conditions.
        conv_checker : AbstractConvergenceChecker
            The convergence checker.
        save_all :
            If True, will save all channels of all fields at each space
            step during the equation resolution. The number of step
            recorded depends on the :attr:`memory_storage` attribute
            of :class:`layout.Layout`.

        """
        # Attr types check ---------------------------------------------
        util.check_attr_type(length, 'length', int, float)
        util.check_attr_type(steps, 'steps', int, list)
        util.check_attr_type(step_method, 'step_method', str, list)
        util.check_attr_type(solver_order, 'solver_order', str)
        util.check_attr_type(stepper_method, 'stepper_method', str, list)
        util.check_attr_type(boundary_cond, 'boundary_cond',
                             AbstractBoundaryConditions, None)
        util.check_attr_type(conv_checker, 'conv_checker',
                             AbstractConvergenceChecker, None)
        util.check_attr_type(save_all, 'save_all', bool)
        # Attr ---------------------------------------------------------
        self._solvers: List[AbstractSolver] = util.make_list(solvers)
        self._nbr_solvers: int = len(self._solvers)
        self._noise_solvers: List[Optional[AbstractSolver]]
        self._noise_solvers = util.make_list(noise_solvers, self._nbr_solvers,
                                             None)
        self._eqs: List[SOLVER_CALLABLE_TYPE]
        self._eqs = [self._solvers[i].f for i in range(self._nbr_solvers)]
        self._steps: List[int] = util.make_list(steps, self._nbr_solvers)
        self._length: float = length
        # Stepper method
        stepper_method = util.make_list(stepper_method, self._nbr_solvers)
        self._stepper_method_str: List[str] = []
        self._stepper_method: List[STEPPER_METHOD_TYPE] = []
        for i in range(len(stepper_method)):
            self._stepper_method_str.append(
                util.check_attr_value(stepper_method[i].lower(),
                                      STEPPER_METHODS, FORWARD))
            self._stepper_method.append(getattr(self, "_{}_method"
                .format(self._stepper_method_str[i].lower())))
        # Step method
        step_method = util.make_list(step_method, self._nbr_solvers)
        self._step_method_str: List[str] = []
        self._step_method: List[STEP_METHOD_TYPE] = []
        for i in range(len(step_method)):
            self._step_method_str.append(
                util.check_attr_value(step_method[i].lower(),
                                      STEP_METHODS, FIXED))
            self._step_method.append(getattr(self, "_{}_step"
                .format(self._step_method_str[i].lower())))
        # Solver order
        solver_order = util.check_attr_value(solver_order.lower(),
                                             SOLVER_ORDERS, FOLLOWING)
        if (solver_sequence is None):
            if (solver_order.lower() == ALTERNATING):
                solver_sequence = [0 for i in range(self._nbr_solvers)]
            if (solver_order.lower() == FOLLOWING):
                solver_sequence = [i for i in range(self._nbr_solvers)]
        else:
            solver_sequence = solver_sequence
        self._solver_seq_ids: List[List[int]] =\
            self._get_seq_ids(solver_sequence)
        self.save_all: bool = save_all
        self._avail_memory: float = 0.
        self._save_step: int = 0
        self._enough_space: bool = False
        self._storage: Storage = Storage()
        self._channels: np.ndarray = np.array([])
        self._noises: np.ndarray = np.array([])
        self._space: np.ndarray = np.array([])
        self._conv_checker: Optional[AbstractConvergenceChecker] = conv_checker
        self._boundary_cond: Optional[AbstractBoundaryConditions] = boundary_cond
    # ==================================================================
    @property
    def equations(self) -> List[SOLVER_CALLABLE_TYPE]:

        return self._eqs
    # ==================================================================
    @property
    def solvers(self) -> List[AbstractSolver]:

        return self._solvers
    # ==================================================================
    @property
    def noise_solvers(self) -> List[Optional[AbstractSolver]]:

        return self._noise_solvers
    # ==================================================================
    @property
    def storage(self) -> Storage:

        return self._storage
    # ==================================================================
    @property
    def boundary_cond(self) -> Optional[AbstractBoundaryConditions]:

        return self._boundary_cond
    # ==================================================================
    @property
    def conv_checker(self) -> Optional[AbstractConvergenceChecker]:

        return self._conv_checker
    # ==================================================================
    def __call__(self, domain: Domain, *fields: List[Field]) -> List[Field]:
        """
        Parameters
        ----------
        domain : Domain
            The domain of the layout.
        fields :
            The fields to compute.

        Returns
        -------
        :
            The fields after the computation.

        """
        # Encoding -----------------------------------------------------
        # Make a unique array with all channels of all fields
        output_fields: List[Field] = []
        for i in range(len(fields)):
            output_fields.extend(fields[i])
        waves: np.ndarray = output_fields[0][:]
        noises: np.ndarray = np.array([output_fields[0].noise])
        for i in range(1, len(output_fields)):
            waves = np.vstack((waves, output_fields[i][:]))
            noises = np.vstack((noises, output_fields[i].noise))
        # Memory and saving management ---------------------------------
        self._avail_memory = domain.memory * 1e9  # Gb -> b
        self._channels = np.array([], dtype=cst.NPFT)
        self._noises = np.array([], dtype=float)
        self._storage = Storage()
        # Passing domain to conv checker and boundary cond if needed
        if (self._boundary_cond is not None):
            self._boundary_cond.initialize(domain)
        if (self._conv_checker is not None):
            self._conv_checker.initialize(domain)
        # Computing ----------------------------------------------------
        start: float = time.time()
        # N.B.: Make sure to not call open or close several times on the
        # same equations if multiple resolution methods apply to it.
        for eq in util.unique(self._eqs):
            if (isinstance(eq, AbstractFieldEquation)):
                eq.open(domain, *fields)
        for ids in self._solver_seq_ids:
            solvers = [self._solvers[id] for id in ids]
            noise_solvers = [self._noise_solvers[id] for id in ids]
            eqs = [self._eqs[id] for id in ids]
            waves, noises = self._stepper_method[ids[0]](waves, noises,
                solvers, noise_solvers, eqs, self._steps[ids[0]],
                self._step_method[ids[0]])
        # Saving computation parameters and waves ----------------------
        # Need to be done before closing to get variable values
        if (self.save_all):
            self._save_to_storage(domain, output_fields)
        # Closing ------------------------------------------------------
        for eq in util.unique(self._eqs):
            if (isinstance(eq, AbstractFieldEquation)):
                eq.close(domain, *fields)
        # Print time and solver info -----------------------------------
        self._print_computation_state(time.time() - start)
        # Decoding -----------------------------------------------------
        ind: int = 0
        for i, field in enumerate(output_fields):
            field.noise = noises[i]
            for j in range(len(field)):
                field[j] = waves[ind]
                ind += 1

        return output_fields
    # ==================================================================
    def _save_to_storage(self, domain: Domain, output_fields: List[Field]
                        ) -> None:
        """Save the waves, noises, space steps and time delays to the
        storage.
        """
        if (self._channels.size):
            time_ = np.ones((self._channels.shape)) * domain.time
            ind = 0
            rep_freq = np.array([], dtype=float)
            center_omega = np.array([], dtype=float)
            for i, field in enumerate(output_fields):
                rep_freq = np.append(rep_freq, field.rep_freq)
                center_omega = np.append(center_omega, field.center_omega)
                for j in range(len(field)):
                    time_[ind] += field.delays[j]
                    ind += 1
            delays = np.array([])
            for eq in util.unique(self._eqs):
                if (isinstance(eq, AbstractFieldEquation)):
                    if (not delays.size):
                        delays = eq.delays
                    else:
                        delays += eq.delays
            if (delays.size):
                # N.B.: [1:] bcs first is initial step and waves
                for i in range(len(delays)):
                    time_[i][1:] += delays[i].reshape((-1,1))
            self._storage.append(self._channels, self._noises, self._space,
                                 time_, center_omega, rep_freq)
            self._channels = np.array([])
            self._noises = np.array([])
            self._space = np.array([])
    # ==================================================================
    def _get_seq_ids(self, solver_sequence):
        """Sort the order of computation and check not feasable
        cases. If not feasable cases encountered, choose by default the
        first value recorded.
        """
        seq_ids = []
        for i in range(max(solver_sequence)+1):
            seq_ids.append([])
            for j in range(self._nbr_solvers):
                if (solver_sequence[j] == i):
                    seq_ids[-1].append(j)
                    # All the stepper methods should be the same
                    if (self._stepper_method[seq_ids[-1][-1]]
                            != self._stepper_method[seq_ids[-1][0]]):
                        util.warning_terminal("All step methods should be "
                            "the same for alternating solvers computation, "
                            "set {} stepper method for solver {}"
                            .format(self._stepper_method_str[seq_ids[-1][0]],
                                    seq_ids[-1][-1]))
                    # All the step methods should be the same
                    if (self._step_method[seq_ids[-1][-1]]
                            != self._step_method[seq_ids[-1][0]]):
                        util.warning_terminal("All step methods should be "
                            "the same for alternating solvers computation, "
                            "set {} step method for solver {}"
                            .format(self._step_method_str[seq_ids[-1][0]],
                                    seq_ids[-1][-1]))
                    # All the step numbers should be the same
                    if (self._steps[seq_ids[-1][-1]]
                            != self._steps[seq_ids[-1][0]]):
                        util.warning_terminal("All steps numbers should be "
                            "the same for alternating solvers computation, "
                            "set {} steps number for solvers {}"
                            .format(self._steps[seq_ids[-1][0]],
                                    seq_ids[-1][-1]))

        return seq_ids
    # ==================================================================
    def _print_computation_state(self, elapsed_time: float) -> None:
        str_to_print = ("Solved method(s) {} ({} steps) "
                        .format(self._solvers[0].name, self._steps[0]))
        for i in range(1, len(self._solvers)):
            str_to_print += ("and {} ({} steps) "
                             .format(self._solvers[i].name, self._steps[i]))
        str_to_print += "for {} km length in {} s".format(self._length,
                                                          str(elapsed_time))
        util.print_terminal(str_to_print, '')
    # ==================================================================
    def _memory_management(self, waves, noises, steps: int, bidir: bool):
        """Create channels, noise and space steps record container
        depending on memory size.
        """
        # Saving management --------------------------------------------
        if (bidir and self.save_all):
            waves_shape = (waves.shape[0]*2, waves.shape[1])
            waves_byte_size = 2*waves.itemsize*waves.size
            noises_shape = (noises.shape[0]*2, noises.shape[1])
            noises_byte_size = 2*noises.itemsize*noises.size
        else:
            waves_shape = (waves.shape[0], waves.shape[1])
            waves_byte_size = waves.itemsize*waves.size
            noises_shape = (noises.shape[0], noises.shape[1])
            noises_byte_size = noises.itemsize*noises.size
        save_step: int = 0
        enough_space: bool
        #memory_per_step: float = ((waves.itemsize*waves.size)
        #                          + )
        max_nbr_steps = int(self._avail_memory
                            / (waves_byte_size+noises_byte_size))
        if (max_nbr_steps):
            enough_space = True
            # N.B. -1 when modulo is zero
            save_step = max(1, int(steps/(max_nbr_steps-1))-1)
            nbr_steps = (steps // save_step)
            nbr_steps = (nbr_steps+1) if save_step != 1 else nbr_steps
        else:
            enough_space = False
            nbr_steps = 2
            warning_message: str = ("Not enough memory space available "
                "for storage.")
            warnings.warn(warning_message, MemorySpaceWarning)
        if (enough_space):
            size_waves_storage = (waves_shape[0], nbr_steps+1, waves_shape[1])
            self._channels = np.zeros(size_waves_storage, dtype=cst.NPFT)
            self._noises = np.zeros((noises_shape[0] , nbr_steps+1,
                                     noises_shape[1]))
            self._space = np.zeros(nbr_steps+1)

        return enough_space, save_step
    # ==================================================================
    def _call_solvers(self, solvers: List[AbstractSolver],
                      noise_solvers: List[Optional[AbstractSolver]],
                      eqs: List[SOLVER_CALLABLE_TYPE], waves: np.ndarray,
                      noises: np.ndarray, z: float, h: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
        # Avoid to call set and update of same equation multiple times
        call_eq_set_method = util.list_repetition_bool_map(eqs, True)
        call_eq_update_method = util.list_repetition_bool_map(eqs, False)
        for id in range(len(solvers)):
            eq: SOLVER_CALLABLE_TYPE = eqs[id]
            if (call_eq_set_method[id]
                    and isinstance(eq, AbstractFieldEquation)):
                eq.set(waves, noises, z, h)
            waves = solvers[id](waves, z, h)
            noise_solver = noise_solvers[id]
            if (noise_solver is not None):
                noises = noise_solver(noises, z, h)
            if (call_eq_update_method[id]
                    and isinstance(eq, AbstractFieldEquation)):
                eq.update(waves, noises, z, h)

        return waves, noises
    # ==================================================================
    @step_method_decorator
    def _fixed_step(self, waves: np.ndarray, noises: np.ndarray,
                    solvers: List[AbstractSolver],
                    noise_solvers: List[Optional[AbstractSolver]],
                    eqs: List[SOLVER_CALLABLE_TYPE], steps: int,
                    forward: bool = True, bidir: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        # Computation --------------------------------------------------
        zs, h = np.linspace(0.0, self._length, steps, False, True)
        zs += h

        if (forward):
            iter_method = 1
            stop = len(zs)
            start =  0
        else:
            iter_method = -1
            stop = -1
            start = len(zs)-1

        if (not bidir):
            if (self.save_all): # First step saving
                self._channels[:,0] = waves
                self._noises[:,0] = noises
                self._space[0] = 0.0
            for i in range(start, stop, iter_method):
                waves, noises = self._call_solvers(solvers, noise_solvers, eqs,
                                                   waves, noises, zs[i], h)
                if (self.save_all and self._enough_space
                        and (not i%self._save_step)):
                    self._channels[:,int(i/self._save_step)+1] = waves
                    self._noises[:,int(i/self._save_step)+1] = noises
                    self._space[int(i/self._save_step)+1] = zs[i]
        else:
            end_ind_waves = len(waves)
            end_ind_noises = len(noises)
            # Save the first step in storage if required ---------------
            if (self.save_all):
                if (forward):
                    self._channels[:end_ind_waves,0] = waves
                    self._noises[:end_ind_noises,0] = noises
                else:
                    self._channels[end_ind_waves:,-1] = waves
                    self._noises[end_ind_noises:,-1] = noises
                self._space[0] = 0.0
            else:
                if (forward):
                    self._channels[:,0] = waves
                    self._noises[:,0] = noises
                else:
                    self._channels[:,-1] = waves
                    self._noises[:,-1] = noises
            # Loop over space steps ------------------------------------
            for i in range(start, stop, iter_method):
                # Prepare waves to send to solver ----------------------
                waves_to_stack: np.ndarray = np.array([])
                noises_to_stack: np.ndarray = np.array([])
                if (self.save_all):
                    if (forward):
                        waves_to_stack = self._channels[end_ind_waves:,i+1]
                        noises_to_stack = self._noises[end_ind_noises:,i+1]
                    else:
                        waves_to_stack = self._channels[:end_ind_waves,i]
                        noises_to_stack = self._noises[:end_ind_noises,i]
                else:
                    if (forward):
                        waves_to_stack = self._channels[:,i+1]
                        noises_to_stack = self._noises[:,i+1]
                    else:
                        waves_to_stack = self._channels[:,i]
                        noises_to_stack = self._noises[:,i]
                waves_tot = np.vstack((waves, waves_to_stack))
                noises_tot = np.vstack((noises, noises_to_stack))
                # Compute one step by calling all the solvers ----------
                res = self._call_solvers(solvers, noise_solvers, eqs,
                                         waves_tot, noises_tot, zs[i], h)
                waves = res[0][:end_ind_waves]
                noises = res[1][:end_ind_noises]
                # Save current step in storage if required --------------
                if (self.save_all):
                    if (forward):
                        self._channels[:end_ind_waves,i+1] = waves
                        self._noises[:end_ind_noises,i+1] = noises
                    else:
                        self._channels[end_ind_waves:,i] = waves
                        self._noises[end_ind_noises:,i] = noises
                    self._space[i+1] = zs[i]
                else:   # Save current info to reuse it on the other direction
                    if (forward):
                        self._channels[:,i+1] = waves
                        self._noises[:,i+1] = noises
                    else:
                        self._channels[:,i] = waves
                        self._noises[:,i] = noises

        return waves, noises
    # ==================================================================
    @step_method_decorator
    def _adaptative_step(self, waves: np.ndarray, noises: np.ndarray,
                         solvers: List[AbstractSolver],
                         noise_solvers: List[Optional[AbstractSolver]],
                         eqs: List[SOLVER_CALLABLE_TYPE], steps: int,
                         forward: bool = True, bidir: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:

        return np.zeros(waves.shape, dtype=cst.NPFT)
    # ==================================================================
    def _forward_method(self, waves: np.ndarray, noises: np.ndarray,
                        solvers: List[AbstractSolver],
                        noise_solvers: List[Optional[AbstractSolver]],
                        eqs: List[SOLVER_CALLABLE_TYPE], steps: int,
                        step_method: STEP_METHOD_TYPE, bidir: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:

        return step_method(waves, noises, solvers, noise_solvers, eqs, steps,
                           True, bidir)
    # ==================================================================
    def _backward_method(self, waves: np.ndarray, noises: np.ndarray,
                         solvers: List[AbstractSolver],
                         noise_solvers: List[Optional[AbstractSolver]],
                         eqs: List[SOLVER_CALLABLE_TYPE], steps: int,
                         step_method: STEP_METHOD_TYPE, bidir: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:

        return step_method(waves, noises, solvers, noise_solvers, eqs, steps,
                           False, bidir)
    # ==================================================================
    def _shooting_method(self, waves: np.ndarray, noises: np.ndarray,
                         solvers: List[AbstractSolver],
                         noise_solvers: List[Optional[AbstractSolver]],
                         eqs: List[SOLVER_CALLABLE_TYPE], steps: int,
                         step_method: STEP_METHOD_TYPE
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the shooting algorithm."""
        if ((self._boundary_cond is not None)
                and (self._conv_checker is not None)):
            # Aliases for cleaner code
            forward = lambda waves, noises: self._forward_method(waves,
                noises, solvers, noise_solvers, eqs, steps, step_method, True)
            backward = lambda waves, noises: self._backward_method(waves,
                noises, solvers, noise_solvers, eqs, steps, step_method, True)
            apply_cond = lambda waves, noises, fiber_end:\
                self._boundary_cond.apply_cond(waves, noises, fiber_end)
            get_input = lambda waves, noises, fiber_end:\
                self._boundary_cond.get_input(waves, noises, fiber_end)
            get_output = lambda waves, waves_, noises, noises_:\
                self._boundary_cond.get_output(waves, waves_, noises, noises_)
            has_converged = lambda waves, waves_, noises, noises_:\
                self._conv_checker.has_converged(np.vstack((waves, waves_)),
                                                 np.vstack((noises, noises_)))
            converged: bool = False
            first_iter: bool = True
            waves_b: np.ndarray  # Backward propagating waves
            waves_f: np.ndarray  # Forward propagating waves
            noises_b: np.ndarray # Backward propagating noises
            noises_f: np.ndarray # Forward propagating noises
            # Start loop till convergence criterion is met
            while (not converged):
                if (first_iter):
                    waves_b, noises_b = get_input(waves, noises, True)
                    first_iter = False
                else:
                    waves_b, noises_b = apply_cond(waves_f, noises_f, True)
                waves_b, noises_b = backward(waves_b, noises_b)
                waves_f, noises_f = apply_cond(waves_b, noises_b, False)
                waves_f, noises_f = forward(waves_f, noises_f)
                converged = has_converged(waves_f, waves_b, noises_f, noises_b)
                util.print_terminal("Shooting method is running and got "
                                    "residual = {}"
                                    .format(self._conv_checker.residual), '')
            waves, noises = get_output(waves_f, waves_b, noises_f, noises_b)
            # Save channels and noises to storage if required
            if (self.save_all and self._channels.size):
                channels_f = self._channels[:(len(self._channels)//2)]
                channels_b = self._channels[(len(self._channels)//2):]
                noises_f = self._noises[:(len(self._noises)//2)]
                noises_b = self._noises[(len(self._noises)//2):]
                self._channels, self._noises = get_output(
                    channels_f, channels_b, noises_f, noises_b)
        else:

            raise MissingInfoError("Boundary condition and convergence "
                "checker must be provided for running shooting method.")

        return waves, noises


if __name__ == "__main__":

    from optcom.components.gaussian import Gaussian
    #from optcom.equations.abstract_wave_equation import AbstractFieldEquation
    from optcom.equations.nlse import NLSE
    from optcom.solvers.abstract_solver import AbstractSolver
    from optcom.solvers.nlse_solver import NLSESolver


    nlse1: AbstractFieldEquation = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse2: AbstractFieldEquation = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse3: AbstractFieldEquation = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse4: AbstractFieldEquation = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse5: AbstractFieldEquation = NLSE(alpha=[0.0], beta=[0.0, 0.0])

    eqs: List[AbstractFieldEquation] = [nlse1, nlse2, nlse3, nlse4, nlse5]
    methods: List[str] = ['ssfm', 'ssfm_super_sym', 'ssfm_symmetric',
                         'ssfm_reduced', 'ssfm']
    solvers: List[AbstractSolver] = []
    for i in range(len(eqs)):
        solvers.append(NLSESolver(eqs[i], methods[i]))
    length: float = 1.0
    steps: List[int] = [10, 9, 8, 7, 6, 5]
    step_method: List[str] = [FIXED, ADAPTATIVE, FIXED]
    solver_sequence: List[int] = [0,0,2,2,1]
    solver_order: str = ALTERNATING
    stepper_method: List[str] = [FORWARD]

    stepper = FieldStepper(solvers=solvers, length=length, steps=steps,
                           step_method=step_method,
                           solver_sequence=solver_sequence,
                           solver_order=solver_order,
                           stepper_method=stepper_method)

    dummy_domain: Domain = Domain()
    ggsn: Gaussian = Gaussian(channels=4)
    dummy_ports, dummy_field = ggsn(dummy_domain)

    res = stepper(dummy_domain, dummy_field)
