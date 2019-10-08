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

from typing import List, Tuple, Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.equations.abstract_equation import AbstractEquation
from optcom.field import Field
from optcom.utils.fft import FFT


class AbstractCoupledEquation(AbstractEquation):
    r"""Parent class of any coupled equation.

    Notes
    -----
    Considering :math:`l = 1, \ldots, M` equations with
    :math:`j = 1, \ldots, K_l` channels per equation, the
    AbstractCoupledEquation class implement the following
    generic equations:

    .. math::  \frac{\partial A_{lj}}{\partial z} =
               f_l(\{A_{l1}, \ldots, A_{lK_l}\})
               + \sum_{m \neq l}^{M} \hat{E}_{lm}^{l} A_{lj}
               + \sum_{m \neq l}^{M} \hat{E}_{lm}^{n} A_{lj}
               + \sum_{m=0}^M \hat{E}_{lm}^{a} \sum_{k=0}^{K_m} A_{mk}

    where :math:`f_l` is an equation related to equation :math:`l`,
    :math:`\hat{E}^l` are the linear effects (self._effects_lin),
    :math:`\hat{E}^n` are the non linear effects
    (self._effects_non_lin),
    and :math:`\hat{E}^l` are the general effects (self._effects_all).

    """

    def __init__(self, nbr_eqs: int):
        self._nbr_eqs: int = nbr_eqs
        self._eqs: List[List[AbstractEquation]] = [[] for i in range(nbr_eqs)]
        self._effects_lin: List[List[List[AbstractEffect]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self._effects_non_lin: List[List[List[AbstractEffect]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self._effects_all: List[List[List[AbstractEffect]]] =\
            [[[] for i in range(nbr_eqs)] for i in range(nbr_eqs)]
        self._op_type: str = ""
        self._omega_all: Array[float]
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        for i in range(len(self._eqs)):
            for eq in self._eqs[i]:
                eq.open(domain, fields[i])

        self._init_eq_ids(list(fields))

        self._center_omega = np.array([])
        self._omega_all = np.zeros((0, domain.samples))
        for field_list in fields:
            for field in field_list:
                self._center_omega = np.hstack((self._center_omega,
                                                field.center_omega))
                self._omega_all = np.vstack((self._omega_all, field.omega))

        self._omega = FFT.fftshift(domain.omega)

        effects = ["lin", "non_lin", "all"]
        for effect in effects:
            effect_list = getattr(self, "_effects_{}".format(effect))
            for i in range(len(effect_list)):
                for j in range(len(effect_list[i])):
                    for k in range(len(effect_list[i][j])):
                        effect_list[i][j][k].omega = self._omega
                        effect_list[i][j][k].time = domain.time
                        effect_list[i][j][k].center_omega =\
                            self._center_omega
    # ==================================================================
    def close(self, domain: Domain, *fields: List[Field]) -> None:
        """This function is called once after a Stepper ended the
        computation.
        """
        super().close(domain, *fields)
        self._omega_all = np.array([])
        for i in range(len(self._eqs)):
            for eq in self._eqs[i]:
                eq.close(domain, fields[i])
    # ==================================================================
    def set(self, waves: Array[cst.NPFT], h: float, x: float) -> None:
        """This function is called before each step of the computation.
        """
        for i in range(len(self._eqs)):
            for eq in self._eqs[i]:
                eq.set(self._in_eq_waves(waves, i), h, x)
    # ==================================================================
    def update(self, waves: Array[cst.NPFT], h: float, x: float) -> None:
        """This function is called after each step of the computation.
        """
        for i in range(len(self._eqs)):
            for eq in self._eqs[i]:
                eq.update(self._in_eq_waves(waves, i), h, x)
    # ==================================================================
    # Operators ========================================================
    # ==================================================================
    def _call(self, op_or_term: str, effect_type: str, waves: Array[cst.NPFT],
              id: int, corr_wave: Array[cst.NPFT]) -> Array[cst.NPFT]:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type) with specified operator type (self._op_type).
        """
        return (self._call_sub(op_or_term, effect_type, waves, id, corr_wave)
                +self._call_main(op_or_term, effect_type, waves, id, corr_wave)
                )
    # ==================================================================
    def _call_main(self, op_or_term: str, effect_type: str,
                   waves: Array[cst.NPFT], id: int, corr_wave: Array[cst.NPFT]
                   ) -> Array[cst.NPFT]:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type) for effect lists in self."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        eq_id = self._eq_id(id)
        rel_wave_id = self._rel_wave_id(id)
        # Operators from current coupled equation ----------------------
        effect_eq_id = getattr(self, "_effects_{}".format(effect_type))[eq_id]
        for i in range(len(effect_eq_id)):
            for effect in effect_eq_id[i]:
                if (eq_id == i):
                    temp_waves = self._in_eq_waves(waves, i)
                    temp_rel_wave_id = rel_wave_id
                else:
                    temp_waves = np.vstack((waves[id],
                                            self._in_eq_waves(waves, i)))
                    temp_rel_wave_id = 0
                res += getattr(effect, "{}{}".format(op_or_term, self._op_type)
                               )(temp_waves, temp_rel_wave_id, corr_wave)
        # Clearing
        getattr(self, "_clear_var_{}".format(effect_type))

        return res
    # ==================================================================
    def _call_sub(self, op_or_term: str, effect_type: str,
                  waves: Array[cst.NPFT], id: int, corr_wave: Array[cst.NPFT]
                  ) -> Array[cst.NPFT]:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type) for equations in self._eqs."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        eq_id = self._eq_id(id)
        rel_wave_id = self._rel_wave_id(id)
        # Operators from sub-equations ---------------------------------
        for eq in self._eqs[eq_id]:
            res += getattr(eq, "{}_{}".format(op_or_term, effect_type)
                           )(self._in_eq_waves(waves, eq_id),
                             rel_wave_id, corr_wave)
        return res
