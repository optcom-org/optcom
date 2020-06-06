import copy
import operator

import pytest
import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from optcom.domain import Domain
from optcom.field import Field
from optcom.field import OperatorError

# ----------------------------------------------------------------------
# Fixtures -------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.fixture
def operator_fixture():
    """Test if the result of operator is a Field and the channels have
    the right values. field_channel, center_omega and rep_freq are
    assumed to have the same length.
    """
    def fct(op, right_op, field_channel, type, center_omega, rep_freq, samples,
            *operands):
        res = []
        for operand in operands:
            new_field = Field(Domain(samples_per_bit=samples), type)
            for i, channel in enumerate(field_channel):
                new_field.append(channel, center_omega[i], rep_freq[i])
            if (right_op):  # Right operand operator
                res.append(op(operand, new_field))
            else:           # Left operand operator
                res.append(op(new_field, operand))

        return res

    return fct

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

scale = 2
length = 12
type_op_test = 1
center_omega_op_test = [1550.]
rep_freq_op_test = [1e3]
field_ = Field(Domain(samples_per_bit=length), type_op_test)
field_.append(scale*np.ones(length), center_omega_op_test[0],
              rep_freq_op_test[0])
operand_args = [int(scale), float(scale), complex(scale),
                scale*np.ones(length), field_]

@pytest.mark.field_op
@pytest.mark.parametrize("op, field_channel, op_res, operands",
    [(operator.__iadd__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)+(scale*np.ones(length))]), operand_args),
     (operator.__isub__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)-(scale*np.ones(length))]), operand_args),
     (operator.__imul__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)*(scale*np.ones(length))]), operand_args),
     (operator.__itruediv__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)/(scale*np.ones(length))]), operand_args),
     ])
def test_ioperator(operator_fixture, op, field_channel, op_res, operands):
    """Should fail if operations do not return the intended results.
    """
    res = operator_fixture(op, False, field_channel, type_op_test,
                           center_omega_op_test, rep_freq_op_test, length,
                           *operands)
    for field in res:
        assert (isinstance(field, Field))
        assert_array_equal(field[:], op_res)


@pytest.mark.field_op
@pytest.mark.parametrize("op, field_channel, op_res, operands",
    [(operator.__add__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)+(scale*np.ones(length))]), operand_args),
     (operator.__sub__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)-(scale*np.ones(length))]), operand_args),
     (operator.__mul__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)*(scale*np.ones(length))]), operand_args),
     (operator.__truediv__, [np.arange(1,length+1)],
      np.array([np.arange(1,length+1)/(scale*np.ones(length))]), operand_args),
     ])
def test_operator(operator_fixture, op, field_channel, op_res, operands):
    """Should fail if operations do not return the intended results.
    """
    res = operator_fixture(op, False, field_channel, type_op_test,
                           center_omega_op_test, rep_freq_op_test, length,
                           *operands)
    for field in res:
        assert (isinstance(field, Field))
        assert_array_equal(field[:], op_res)


@pytest.mark.field_op
@pytest.mark.parametrize("op, field_channel, op_res, operands",
    [(operator.__add__, [np.arange(1,length+1)],
      np.array([(scale*np.ones(length))+np.arange(1,length+1)]), operand_args),
     (operator.__sub__, [np.arange(1,length+1)],
      np.array([(scale*np.ones(length))-np.arange(1,length+1)]), operand_args),
     (operator.__mul__, [np.arange(1,length+1)],
      np.array([(scale*np.ones(length))*np.arange(1,length+1)]), operand_args),
     (operator.__truediv__, [np.arange(1,length+1)],
      np.array([(scale*np.ones(length))/np.arange(1,length+1)]), operand_args),
     ])
def test_roperator(operator_fixture, op, field_channel, op_res, operands):
    """Should fail if operations do not return the intended results.
    """
    res = operator_fixture(op, True, field_channel, type_op_test,
                           center_omega_op_test, rep_freq_op_test, length,
                           *operands)
    for field in res:
        assert (isinstance(field, Field))
        # assert_array_almost_equal to take into account rounding errors
        assert_array_almost_equal(field[:], op_res, 16)


@pytest.mark.field
@pytest.mark.parametrize("reset_channels, reset_noise, reset_delays",
    [(False, False, False), (False, True, True), (False, False, True),
     (True, False, True), (True, True, False), (False, True, False),
     (True, False, False), (True, True, True)
    ])
def test_copy_field(reset_channels, reset_noise, reset_delays):
    """Should fail if no valid copy is returned.
    """
    uni_length = 12
    domain = Domain(samples_per_bit=uni_length, noise_samples=uni_length)
    type = 1
    center_omega = 1550.
    rep_freq = 1e3
    delay = 10.
    field = Field(domain, type)
    field.noise = np.ones(uni_length)
    field.append(np.arange(uni_length), center_omega, rep_freq, delay)
    new_field = field.get_copy('', reset_channels, reset_noise,
                               reset_delays)
    equal_channels = np.array_equal(new_field.channels, field.channels)
    if (not reset_channels):
        assert equal_channels
    else:
        assert not equal_channels
    equal_noise = np.array_equal(new_field.noise, field.noise)
    if (not reset_noise):
        assert equal_noise
    else:
        assert not equal_noise
    equal_delays = np.array_equal(new_field.delays, field.delays)
    if (not reset_delays):
        assert equal_delays
    else:
        assert not equal_delays


@pytest.mark.field_op
@pytest.mark.parametrize("op, right_op",
    [(operator.__iadd__, False), (operator.__isub__, False),
     (operator.__imul__, False), (operator.__itruediv__, False),
     (operator.__add__, False), (operator.__sub__, False),
     (operator.__mul__, False), (operator.__truediv__, False),
     (operator.__add__, True), (operator.__sub__, True),
     (operator.__mul__, True), (operator.__truediv__, True)
     ])
def test_same_omega(operator_fixture, op, right_op):
    """Should not fail if the center omega are the same and in the same
    order."""
    length = 12
    field = Field(Domain(samples_per_bit=length), type_op_test)
    center_omegas = [1030., 1025., 1020.]
    rep_freqs = [1., 2., 3.]
    channels = [np.ones(length)*(i+1) for i in range(len(center_omegas))]
    for i in range(len(center_omegas)):
        field.append(channels[i], center_omegas[i], rep_freqs[i])
    operands = [field]
    res = operator_fixture(op, right_op, channels, type_op_test,
                           center_omegas, rep_freqs, length, *operands)


@pytest.mark.field_op
@pytest.mark.parametrize("op, right_op",
    [(operator.__iadd__, False), (operator.__isub__, False),
     (operator.__imul__, False), (operator.__itruediv__, False),
     (operator.__add__, False), (operator.__sub__, False),
     (operator.__mul__, False), (operator.__truediv__, False),
     (operator.__add__, True), (operator.__sub__, True),
     (operator.__mul__, True), (operator.__truediv__, True)
     ])
def test_same_unorder_omega(operator_fixture, op, right_op):
    """Should not fail if the center omega are the same but in different
    order."""
    length = 12
    field = Field(Domain(samples_per_bit=length), type_op_test)
    center_omegas = [1030., 1025., 1020.]
    rep_freqs = [1., 2., 3.]
    channels = [np.ones(length)*(i+1) for i in range(len(center_omegas))]
    for i in range(len(center_omegas)):
        field.append(channels[i], center_omegas[i], rep_freqs[i])
    operands = [field]
    center_omegas = [1025., 1020., 1030.]
    rep_freqs = [2., 3., 1.]
    res = operator_fixture(op, right_op, channels, type_op_test,
                           center_omegas, rep_freqs, length, *operands)


@pytest.mark.field_op
@pytest.mark.parametrize("op, right_op",
    [(operator.__iadd__, False), (operator.__isub__, False),
     (operator.__imul__, False), (operator.__itruediv__, False),
     (operator.__add__, False), (operator.__sub__, False),
     (operator.__mul__, False), (operator.__truediv__, False),
     (operator.__add__, True), (operator.__sub__, True),
     (operator.__mul__, True), (operator.__truediv__, True)
     ])
def test_same_unorder_omega(operator_fixture, op, right_op):
    """Should not fail if the center omega are the not same."""
    length = 12
    field = Field(Domain(samples_per_bit=length), type_op_test)
    center_omegas = [1030., 1025., 1020.]
    rep_freqs = [1., 2., 3.]
    channels = [np.ones(length)*(i+1) for i in range(len(center_omegas))]
    for i in range(len(center_omegas)):
        field.append(channels[i], center_omegas[i], rep_freqs[i])
    operands = [field]
    center_omegas = [1095., 1000., 1040.]
    rep_freqs = [2., 3., 1.]
    res = operator_fixture(op, right_op, channels, type_op_test,
                           center_omegas, rep_freqs, length, *operands)


@pytest.mark.field_op
@pytest.mark.parametrize("op, right_op",
    [(operator.__iadd__, False), (operator.__isub__, False),
     (operator.__imul__, False), (operator.__itruediv__, False),
     (operator.__add__, False), (operator.__sub__, False),
     (operator.__mul__, False), (operator.__truediv__, False),
     (operator.__add__, True), (operator.__sub__, True),
     (operator.__mul__, True), (operator.__truediv__, True)
     ])
def test_no_common_omegas(operator_fixture, op, right_op):
    """Should not perform math operators if the center omegas are
    different."""
    length = 12
    field = Field(Domain(samples_per_bit=length), type_op_test)
    center_omegas = [1030., 1025., 1020.]
    rep_freqs = [1., 2., 3.]
    channels = [np.ones(length)*(i+1) for i in range(len(center_omegas))]
    for i in range(len(center_omegas)):
        field.append(channels[i], center_omegas[i], rep_freqs[i])
    operands = [field]
    center_omegas = [1029., 1021.]
    channels = [np.ones(length)*(i+1) for i in range(len(center_omegas))]
    rep_freqs = [2., 3.]
    res = operator_fixture(op, right_op, channels, type_op_test, center_omegas,
                           rep_freqs, length, *operands)
    field_res = res[0]
    if (len(field_res) == len(field)):
        assert np.array_equal(field_res[:], field[:])
    else:
        assert np.array_equal(field_res[:], np.asarray(channels))
