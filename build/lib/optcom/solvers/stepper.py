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
import time
from typing import Any, Callable, List, Optional, TypeVar, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_equation import AbstractEquation
from optcom.field import Field
from optcom.solvers.solver import Solver


STEP_METHOD_TYPE = Callable[[Array[cst.NPFT], List[Solver], int, bool],
                            Array[cst.NPFT]]


# Step methods
FIXED = "fixed"
ADAPTATIVE = "adaptative"
# Solvers order method
ALTERNATING = "alternating"
FOLLOWING = "following"
# Stepper methods
FORWARD = "forward"
SHOOTING  = "shooting"


class Stepper(object):

    def __init__(self, eqs: List[AbstractEquation],
                 method: List[str] = [cst.DEFAULT_SOLVER],
                 length: float = 1.0, steps: List[int] = [100],
                 step_method: List[str] = [FIXED],
                 solver_order: str = FOLLOWING,
                 stepper_method: List[str] = [FORWARD],
                 solver_sequence: Optional[List[int]] = None,
                 error: float = 0.01, save_all: bool = False) -> None:
        r"""
        Parameters
        ----------
        eqs : list of AbstractEquation
            The equation to solve.
        method :
            The method used to solve the equation.
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
        error :
            The error for convergence criterion of stepper resolution.
        save_all :
            If True, will save all channels of all fields at each space
            step during the equation resolution. The number of step
            recorded depends on the :attr:`memory_storage` attribute
            of :class:`layout.Layout`. The recorded channels can be
            accessed with the attribute :attr:`storage`.

        """
        self._eqs: List[AbstractEquation] = util.make_list(eqs)
        self._nbr_solvers: int = len(self._eqs)
        self._methods: List[str] = util.make_list(method, self._nbr_solvers)
        self._steps: List[int] = util.make_list(steps, self._nbr_solvers)
        self._length: float = length
        self._solvers: List[Solver] = [Solver(self._eqs[i], self._methods[i])
                                       for i in range(self._nbr_solvers)]
        stepper_method = util.make_list(stepper_method, self._nbr_solvers)
        self._stepper_method_str = stepper_method
        self._stepper_method = [getattr(self, "_{}_method"
                                        .format(stepper_method[i].lower()))
                                for i in range(self._nbr_solvers)]
        step_method = util.make_list(step_method, self._nbr_solvers)
        self._step_method_str = step_method
        self._step_method = [getattr(self, "_{}_step"
                                     .format(step_method[i].lower()))
                             for i in range(self._nbr_solvers)]
        self._start_shooting_forward: bool = True
        if (solver_sequence is None):
            if (solver_order.lower() == ALTERNATING):
                solver_sequence = [0 for i in range(self._nbr_solvers)]
            if (solver_order.lower() == FOLLOWING):
                solver_sequence = [i for i in range(self._nbr_solvers)]
        else:
            solver_sequence = solver_sequence
        self._solver_seq_ids: List[List[int]] =\
            self._get_seq_ids(solver_sequence)
        self._error = error
        self._save_all = save_all
    # ==================================================================
    @property
    def equations(self):

        return self._eqs
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
        output_fields = []
        for i in range(len(fields)):
            output_fields.extend(fields[i])

        waves = output_fields[0][:]
        for i in range(1, len(output_fields)):
            waves = np.vstack((waves, output_fields[i][:]))
        # Memory and saving management ---------------------------------
        if (self._save_all):
            memory = domain.memory * 1e9  # Gb -> b
            self._max_nbr_steps = int(memory / (waves.itemsize*waves.size))
            self._storage = np.array([], dtype=cst.NPFT)
        # Computing ----------------------------------------------------
        start = time.time()
        # N.B.: Make sure to not call open or close several times on the
        # same equations if multiple resolution methods apply to it.
        for eq in util.unique(self._eqs):
            eq.open(domain, *fields)
        for ids in self._solver_seq_ids:
            solvers = [self._solvers[id] for id in ids]
            eqs = [self._eqs[id] for id in ids]
            waves = self._stepper_method[ids[0]](waves, solvers, eqs,
                                                 self._steps[ids[0]],
                                                 self._step_method[ids[0]])
        for eq in util.unique(self._eqs):
            eq.close(domain, *fields)
        # Print time and solver info -----------------------------------
        elapsed_time = time.time() - start
        self._print_computation_state(elapsed_time)
        # Decoding -----------------------------------------------------
        ind = 0
        for i, field in enumerate(output_fields):
            if (self._save_all):
                field.storage = self._storage[:,ind:ind+len(field),:]
            for j in range(len(field)):
                field[j] = waves[ind]
                ind += 1

        return output_fields
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
                        .format(self._methods[0], self._steps[0]))
        for i in range(1, len(self._methods)):
            str_to_print += "and {} ({} steps) ".format(self._methods[i],
                                                        self._steps[i])
        str_to_print += "for {} km length in {} s".format(self._length,
                                                          str(elapsed_time))
        util.print_terminal(str_to_print, '')
    # ==================================================================
    def _fixed_step(self, waves: Array[cst.NPFT], solvers: List[Solver],
                    steps: int, forward: bool = True,
                    start: Optional[int] = None) -> Array[cst.NPFT]:
        # Saving management --------------------------------------------
        enough_space: bool
        if (self._save_all):
            if (self._max_nbr_steps):
                enough_space = True
                # N.B. -1 when modulo is zero
                save_step = max(1, int(steps/(self._max_nbr_steps-1))-1)
                nbr_steps = (steps // save_step)
                nbr_steps = (nbr_steps+1) if save_step != 1 else nbr_steps
            else:
                enough_space = False
                nbr_steps = 2
                util.warning_terminal("Not enough space for storage.")
            size_storage = (nbr_steps,) + waves.shape
            self._storage = np.zeros(size_storage, dtype=cst.NPFT)
        # Computation --------------------------------------------------
        zs, h = np.linspace(0.0, self._length, steps+1, True, True)
        zs = zs[:-1]

        if (start is not None and self._save_all and enough_space):
            if (start > 0):
                for i in range(start):
                    self._storage[i] = waves
            if (start < -1):
                for i in range(1, abs(start)):
                    self._storage[-i] = waves

        if (forward):
            iter_method = 1
            stop = len(zs)
            start =  0 if start is None else start
        else:
            iter_method = -1
            stop = -1
            start = len(zs)-1 if start is None else len(zs)+start

        for i in range(start, stop, iter_method):
            for id in range(len(solvers)):
                waves = solvers[id](waves, h, zs[i])
            if (self._save_all and enough_space and not i%save_step):
                self._storage[int(i/save_step)] = waves

        return waves
    # ==================================================================
    def _adaptative_step(self, waves: Array[cst.NPFT], solvers: List[Solver],
                         steps: int, forward: bool = True,
                         start: Optional[int] = None) -> Array[cst.NPFT]:

        return np.zeros(waves.shape, dtype=cst.NPFT)
    # ==================================================================
    def _forward_method(self, waves: Array[cst.NPFT], solvers: List[Solver],
                        eqs: List[AbstractEquation], steps: int,
                        step_method: STEP_METHOD_TYPE) -> Array[cst.NPFT]:
        waves = step_method(waves, solvers, steps, True)

        return waves
    # ==================================================================
    def start_shooting_forward(self):
        """Set parameters to start the shooting method with forward
        pass.
        """
        self._start_shooting_forward = True
    # ==================================================================
    def start_shooting_backward(self):
        """Set parameters to start the shooting method with backward
        pass.
        """
        self._start_shooting_forward = False
    # ==================================================================
    def _shooting_method(self, waves: Array[cst.NPFT], solvers: List[Solver],
                         eqs: List[AbstractEquation], steps: int,
                         step_method: STEP_METHOD_TYPE) -> Array[cst.NPFT]:

        # N.B.: take last equation to get criterion by defaults
        #       -> need to handle different cases that might pop up

        def apply_ic(eqs, waves_init, end):

            res = np.zeros_like(waves_init)
            for eq in eqs:
                res = eq.initial_condition(waves_init, end)

            return res

        error_bound = self._error
        # First iteration for co-prop or counter prop scheme
        if (self._start_shooting_forward):
            waves_f = apply_ic(eqs, waves, False)
            waves_f = step_method(waves_f, solvers, steps, True, 1)
        else:
            waves_b = apply_ic(eqs, waves, True)
            waves_b = step_method(waves_b, solvers, steps, False, -2)
            waves_f = apply_ic(eqs, waves, False)
            waves_f = step_method(waves_f, solvers, steps, True, 1)
        # Start loop till convergence criterion is met
        cached_criterion = 0.0
        error = error_bound * 1e10
        while (error > error_bound):
            waves_f_back_up = waves_f
            waves_b = apply_ic(eqs, waves_f, True)
            waves_b = step_method(waves_b, solvers, steps, False, -2)
            waves_f = apply_ic(eqs, waves_b, False)
            waves_f = step_method(waves_f, solvers, steps, True, 1)
            new_cached_criterion = eqs[-1].get_criterion(waves_f, waves_b)
            old_error = error
            error = abs(cached_criterion - new_cached_criterion)
            util.print_terminal("Shooting method is running and got error = {}"
                                .format(error), '')
            cached_criterion = new_cached_criterion
            if (old_error < error):
                util.warning_terminal("Shooting method stopped before "
                    "reaching error bound value as error is increasing.")
                error = 0.0
                waves_f = waves_f_back_up

        return waves_f


if __name__ == "__main__":

    from optcom.components.gaussian import Gaussian
    from optcom.equations.nlse import NLSE

    nlse1 = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse2 = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse3 = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse4 = NLSE(alpha=[0.0], beta=[0.0, 0.0])
    nlse5 = NLSE(alpha=[0.0], beta=[0.0, 0.0])

    eqs = [nlse1, nlse2, nlse3, nlse4, nlse5]
    method = ['ssfm', 'ssfm_super_sym', 'ssfm_symmetric', 'ssfm_reduced',
              'ssfm']
    length = 1.0
    steps = [10, 9, 8, 7, 6, 5]
    step_method = [FIXED, ADAPTATIVE, FIXED]
    solver_sequence = [0,0,2,2,1]
    solver_order = ALTERNATING
    stepper_method = [FORWARD]

    stepper = Stepper(eqs=eqs, method=method, length=length, steps=steps,
                      step_method=step_method, solver_sequence=solver_sequence,
                      solver_order=solver_order, stepper_method=stepper_method)

    dummy_domain = Domain()
    ggsn = Gaussian(channels=4)
    dummy_ports, dummy_field = ggsn(dummy_domain)

    res = stepper(dummy_domain, dummy_field)
