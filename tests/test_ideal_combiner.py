import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

import optcom.config as cfg
import optcom.utils.constants as cst
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_combiner import IdealCombiner
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout

temporal_power = Field.temporal_power
spectral_power = Field.spectral_power
energy = Field.energy
fwhm = Field.fwhm


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_output_combiner_no_combine(nbr_channels, ratios):
    """Should fail if the output temporal power division does not correpsond to
    the dividing ratios.
    """
    # Environment creation
    gssns = []
    nbr_arms = len(ratios)
    base_power = 10.0
    for i in range(nbr_arms):
        gssns.append(Gaussian(channels=nbr_channels, save=True,
                     peak_power=[(i+1)*base_power
                                 for i in range(nbr_channels)]))
    combiner = IdealCombiner(arms=nbr_arms, ratios=ratios, save=True,
                             combine=False)
    lt = Layout()
    for i in range(nbr_arms):
        lt.add_link(gssns[i][0], combiner[i])
    lt.run_all()
    # Testing
    init_fields = []
    for i in range(0, nbr_arms):
        init_fields.extend(gssns[i][0].fields)
    output_fields = combiner[nbr_arms].fields
    assert len(output_fields) == nbr_arms
    assert len(output_fields) == len(init_fields)
    for i in range(len(output_fields)):
        for j in range(len(output_fields[i])):
            # Taking into account rounding errors
            power_1 = ratios[i]*temporal_power(init_fields[i][j])
            power_2 = temporal_power(output_fields[i][j])
            assert_array_almost_equal(power_1, power_2, 10)


@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_combine_output_diff_omega(nbr_channels, ratios):
    """Should fail if the different omega are added to each other.
    """
    # Environment creation
    back_up_flag_omega = cfg.get_field_op_matching_omega()
    back_up_flag_rep_freq = cfg.get_field_op_matching_rep_freq()
    cfg.set_field_op_matching_omega(True)
    cfg.set_field_op_matching_rep_freq(False)
    gssns = []
    nbr_arms = len(ratios)
    base_power = 10.0
    for i in range(nbr_arms):
        gssns.append(Gaussian(channels=nbr_channels, save=True,
                              peak_power=[(j+1)*base_power
                                          for j in range(nbr_channels)],
                              center_lambda=[(1500.+j)*(i+1)
                                             for j in range(nbr_channels)],
                              rep_freq=[1e-3]))
    combiner = IdealCombiner(arms=nbr_arms, ratios=ratios, save=True,
                             combine=True)
    lt = Layout()
    for i in range(nbr_arms):
        lt.add_link(gssns[i][0], combiner[i])
    lt.run_all()
    lt.reset()
    # Testing
    init_fields = []
    for i in range(0, nbr_arms):
        init_fields.extend(gssns[i][0].fields)
    output_fields = combiner[nbr_arms].fields
    assert len(output_fields) == 1
    assert len(output_fields[0]) == (nbr_channels*nbr_arms)
    # Reset
    cfg.set_field_op_matching_omega(back_up_flag_omega)
    cfg.set_field_op_matching_rep_freq(back_up_flag_rep_freq)


@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_combine_output_same_omega(nbr_channels, ratios):
    """Should fail if the different omega are not added to each other.
    """
    # Environment creation
    back_up_flag_omega = cfg.get_field_op_matching_omega()
    back_up_flag_rep_freq = cfg.get_field_op_matching_rep_freq()
    cfg.set_field_op_matching_omega(False)
    cfg.set_field_op_matching_rep_freq(True)
    gssns = []
    nbr_arms = len(ratios)
    base_power = 10.0
    for i in range(nbr_arms):
        gssns.append(Gaussian(channels=nbr_channels, save=True,
                              peak_power=[(j+1)*base_power
                                          for j in range(nbr_channels)],
                              center_lambda=[(1500.+j)*(i+1)
                                             for j in range(nbr_channels)],
                              rep_freq=[1e-3]))
    combiner = IdealCombiner(arms=nbr_arms, ratios=ratios, save=True,
                             combine=True)
    lt = Layout()
    for i in range(nbr_arms):
        lt.add_link(gssns[i][0], combiner[i])
    lt.run_all()
    lt.reset()
    # Testing
    init_fields = []
    for i in range(0, nbr_arms):
        init_fields.extend(gssns[i][0].fields)
    output_fields = combiner[nbr_arms].fields
    assert len(output_fields) == 1
    assert len(output_fields[0]) == nbr_channels
    # Reset
    cfg.set_field_op_matching_omega(back_up_flag_omega)
    cfg.set_field_op_matching_rep_freq(back_up_flag_rep_freq)


@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_combine_output_diff_rep_freq(nbr_channels, ratios):
    """Should fail if the different repetition frequencies are added to
    each other.
    """
    # Environment creation
    back_up_flag_omega = cfg.get_field_op_matching_omega()
    back_up_flag_rep_freq = cfg.get_field_op_matching_rep_freq()
    cfg.set_field_op_matching_omega(False)
    cfg.set_field_op_matching_rep_freq(True)
    gssns = []
    nbr_arms = len(ratios)
    base_power = 10.0
    for i in range(nbr_arms):
        gssns.append(Gaussian(channels=nbr_channels, save=True,
                              peak_power=[(j+1)*base_power
                                          for j in range(nbr_channels)],
                              center_lambda=[1500.
                                             for j in range(nbr_channels)],
                              rep_freq=[(1e-2+(j*1e-4))*(i+1)
                                        for j in range(nbr_channels)]))
    combiner = IdealCombiner(arms=nbr_arms, ratios=ratios, save=True,
                             combine=True)
    lt = Layout()
    for i in range(nbr_arms):
        lt.add_link(gssns[i][0], combiner[i])
    lt.run_all()
    lt.reset()
    # Testing
    init_fields = []
    for i in range(0, nbr_arms):
        init_fields.extend(gssns[i][0].fields)
    output_fields = combiner[nbr_arms].fields
    assert len(output_fields) == 1
    assert len(output_fields[0]) == (nbr_channels*nbr_arms)
    # Reset
    cfg.set_field_op_matching_omega(back_up_flag_omega)
    cfg.set_field_op_matching_rep_freq(back_up_flag_rep_freq)



@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_combine_output_same_rep_freq(nbr_channels, ratios):
    """Should fail if the different repetition frequencies are not added
    to each other.
    """
    # Environment creation
    back_up_flag_omega = cfg.get_field_op_matching_omega()
    back_up_flag_rep_freq = cfg.get_field_op_matching_rep_freq()
    cfg.set_field_op_matching_omega(True)
    cfg.set_field_op_matching_rep_freq(False)
    gssns = []
    nbr_arms = len(ratios)
    base_power = 10.0
    for i in range(nbr_arms):
        gssns.append(Gaussian(channels=nbr_channels, save=True,
                              peak_power=[(j+1)*base_power
                                          for j in range(nbr_channels)],
                              center_lambda=[1500.
                                             for j in range(nbr_channels)],
                              rep_freq=[(1e-2+(j*1e-4))*(i+1)
                                        for j in range(nbr_channels)]))
    combiner = IdealCombiner(arms=nbr_arms, ratios=ratios, save=True,
                             combine=True)
    lt = Layout()
    for i in range(nbr_arms):
        lt.add_link(gssns[i][0], combiner[i])
    lt.run_all()
    lt.reset()
    # Testing
    init_fields = []
    for i in range(0, nbr_arms):
        init_fields.extend(gssns[i][0].fields)
    output_fields = combiner[nbr_arms].fields
    assert len(output_fields) == 1
    assert len(output_fields[0]) == nbr_channels
    # Reset
    cfg.set_field_op_matching_omega(back_up_flag_omega)
    cfg.set_field_op_matching_rep_freq(back_up_flag_rep_freq)



@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_combine_output_diff_omega_and_rep_freq(nbr_channels, ratios):
    """Should fail if the different omega and repetition frequencies
    are added to each other.
    """
    # Environment creation
    back_up_flag_omega = cfg.get_field_op_matching_omega()
    back_up_flag_rep_freq = cfg.get_field_op_matching_rep_freq()
    cfg.set_field_op_matching_omega(True)
    cfg.set_field_op_matching_rep_freq(True)
    gssns = []
    nbr_arms = len(ratios)
    base_power = 10.0
    for i in range(nbr_arms):
        gssns.append(Gaussian(channels=nbr_channels, save=True,
                              peak_power=[(j+1)*base_power
                                          for j in range(nbr_channels)],
                              center_lambda=[(1500.+j)*(i+1)
                                             for j in range(nbr_channels)],
                              rep_freq=[(1e-2+(j*1e-4))*(i+1)
                                        for j in range(nbr_channels)]))
    combiner = IdealCombiner(arms=nbr_arms, ratios=ratios, save=True,
                             combine=True)
    lt = Layout()
    for i in range(nbr_arms):
        lt.add_link(gssns[i][0], combiner[i])
    lt.run_all()
    lt.reset()
    # Testing
    init_fields = []
    for i in range(0, nbr_arms):
        init_fields.extend(gssns[i][0].fields)
    output_fields = combiner[nbr_arms].fields
    assert len(output_fields) == 1
    assert len(output_fields[0]) == (nbr_channels*nbr_arms)
    # Reset
    cfg.set_field_op_matching_omega(back_up_flag_omega)
    cfg.set_field_op_matching_rep_freq(back_up_flag_rep_freq)
