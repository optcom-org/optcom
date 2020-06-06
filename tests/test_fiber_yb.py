import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

import optcom.utils.constants as cst
from optcom.components.cw import CW
from optcom.components.gaussian import Gaussian
from optcom.components.fiber import Fiber
from optcom.components.fiber_yb import FiberYb
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout


temporal_power = Field.temporal_power
spectral_power = Field.spectral_power
fwhm = Field.fwhm

# ----------------------------------------------------------------------
# Fixtures -------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.fixture
def split_noise_option_co_prop():
    """Basic layout structure with co-propagating pump and seed, return
    the noise of fields and reflected fields.

    Notes
    -----
    Test case::

    starter ___ fiber

    """
    def layout(split_noise_option, channels, peak_powers, center_lambdas):
        gssn = Gaussian(channels=channels, peak_power=peak_powers,
                        center_lambda=center_lambdas, save=True)
        pump = CW(channels=3, peak_power=[0.01], center_lambda=[976.])
        fiber = FiberYb(length=0.01, split_noise_option=split_noise_option,
                        steps=10, REFL_SEED=True, REFL_PUMP=True, BISEED=False,
                        BIPUMP=False, save=True, PROP_REFL=True,
                        PROP_PUMP=True, alpha=[0.05], max_nbr_iter=2,
                        NOISE=True)
        lt = Layout(Domain(samples_per_bit=64, noise_samples=10))
        lt.link_unidir((gssn[0], fiber[0]), (pump[0], fiber[2]))
        lt.run_all()

        noise_power_input = fiber[0].fields[0].noise
        noise_power_seed = fiber[1].fields[0].noise
        noise_power_refl_seed = fiber[0].fields[1].noise
        noise_power_pump = fiber[1].fields[1].noise
        noise_power_refl_pump = fiber[0].fields[2].noise

        return (noise_power_input, noise_power_seed, noise_power_refl_seed,
                noise_power_pump, noise_power_refl_pump)

    return layout


@pytest.fixture
def co_seed_co_pump_scheme():
    """Basic layout structure with co-propagating pump and seed. Return
    the storage.

    Notes
    -----
    Test case::

    Seed ____ Fiber Amp ____ X
            /           \
    Pump __/             \__ X

    """
    def layout(nbr_channels_seed, peak_powers_seed, center_lambdas_seed,
               nbr_channels_pump, peak_powers_pump, center_lambdas_pump,
               REFL_SEED, REFL_PUMP, NOISE):
        gssn = Gaussian(channels=nbr_channels_seed,
                        peak_power=peak_powers_seed,
                        center_lambda=center_lambdas_seed)
        pump = CW(channels=nbr_channels_pump,
                  peak_power=peak_powers_pump,
                  center_lambda=center_lambdas_pump)
        fiber = FiberYb(length=0.01, steps=10, REFL_SEED=REFL_SEED,
                        REFL_PUMP=REFL_PUMP, BISEED=False, BIPUMP=False,
                        save_all=True, alpha=[0.05], max_nbr_iter=2,
                        NOISE=NOISE)
        lt = Layout(Domain(samples_per_bit=64, noise_samples=10))
        lt.link_unidir((gssn[0], fiber[0]), (pump[0], fiber[2]))
        lt.run_all()

        return fiber.storages[0]

    return layout


@pytest.fixture
def co_seed_counter_pump_scheme():
    """Basic layout structure with co-propagating seed and counter-
    propagating pumps, split pump channels in two parts. Return the
    storage.

    Notes
    -----
    Test case::

    Seed ____ Fiber Amp ____ X
            /           \
    Pump __/             \__ Pump

    """
    def layout(nbr_channels_seed, peak_powers_seed, center_lambdas_seed,
               nbr_channels_pump, peak_powers_pump, center_lambdas_pump,
               REFL_SEED, REFL_PUMP, NOISE):
        gssn = Gaussian(channels=nbr_channels_seed,
                        peak_power=peak_powers_seed,
                        center_lambda=center_lambdas_seed)
        nbr_ch_pump = nbr_channels_pump // 2
        pump = CW(channels=nbr_ch_pump,
                  peak_power=peak_powers_pump[:nbr_ch_pump],
                  center_lambda=center_lambdas_pump[:nbr_ch_pump])
        pump_ = CW(channels=(nbr_channels_pump - nbr_ch_pump),
                   peak_power=peak_powers_pump[nbr_ch_pump:],
                   center_lambda=center_lambdas_pump[nbr_ch_pump:])
        fiber = FiberYb(length=0.01, steps=10, REFL_SEED=REFL_SEED,
                        REFL_PUMP=REFL_PUMP, BISEED=False, BIPUMP=True,
                        save_all=True, alpha=[0.05], max_nbr_iter=2,
                        NOISE=NOISE)
        lt = Layout(Domain(samples_per_bit=64, noise_samples=10))
        lt.link_unidir((gssn[0], fiber[0]), (pump[0], fiber[2]),
                       (pump_[0], fiber[3]))
        lt.run_all()

        return fiber.storages[0]

    return layout


@pytest.fixture
def counter_seed_co_pump_scheme():
    """Basic layout structure with co-propagating pump and counter-
    propagating seeds, split seed channels in two parts. Return the
    storage.

    Notes
    -----
    Test case::

    Seed ____ Fiber Amp ____ Seed
            /           \
    Pump __/             \__ X

    """
    def layout(nbr_channels_seed, peak_powers_seed, center_lambdas_seed,
               nbr_channels_pump, peak_powers_pump, center_lambdas_pump,
               REFL_SEED, REFL_PUMP, NOISE):
        nbr_ch_seed = nbr_channels_seed // 2
        gssn = Gaussian(channels=nbr_ch_seed,
                        peak_power=peak_powers_seed[:nbr_ch_seed],
                        center_lambda=center_lambdas_seed[:nbr_ch_seed])
        gssn_ = Gaussian(channels=(nbr_channels_seed - nbr_ch_seed),
                         peak_power=peak_powers_seed[nbr_ch_seed:],
                         center_lambda=center_lambdas_seed[nbr_ch_seed:])
        pump = CW(channels=nbr_channels_pump,
                  peak_power=peak_powers_pump,
                  center_lambda=center_lambdas_pump)
        fiber = FiberYb(length=0.01, steps=10, REFL_SEED=REFL_SEED,
                        REFL_PUMP=REFL_PUMP, BISEED=True, BIPUMP=False,
                        save_all=True, alpha=[0.05], max_nbr_iter=2,
                        NOISE=NOISE)
        lt = Layout(Domain(samples_per_bit=64, noise_samples=10))
        lt.link_unidir((gssn[0], fiber[0]), (pump[0], fiber[2]),
                       (gssn_[0], fiber[1]))
        lt.run_all()

        return fiber.storages[0]

    return layout


@pytest.fixture
def counter_seed_counter_pump_scheme():
    """Basic layout structure with counter-propagating seeds and pumps,
    split seed and pump channels in two parts. Return the storage.

    Notes
    -----
    Test case::

    Seed ____ Fiber Amp ____ Seed
            /           \
    Pump __/             \__ Pump

    """
    def layout(nbr_channels_seed, peak_powers_seed, center_lambdas_seed,
               nbr_channels_pump, peak_powers_pump, center_lambdas_pump,
               REFL_SEED, REFL_PUMP, NOISE):
        nbr_ch_seed = nbr_channels_seed // 2
        gssn = Gaussian(channels=nbr_ch_seed,
                        peak_power=peak_powers_seed[:nbr_ch_seed],
                        center_lambda=center_lambdas_seed[:nbr_ch_seed])
        gssn_ = Gaussian(channels=(nbr_channels_seed - nbr_ch_seed),
                         peak_power=peak_powers_seed[nbr_ch_seed:],
                         center_lambda=center_lambdas_seed[nbr_ch_seed:])
        nbr_ch_pump = nbr_channels_pump // 2
        pump = CW(channels=nbr_ch_pump,
                  peak_power=peak_powers_pump[:nbr_ch_pump],
                  center_lambda=center_lambdas_pump[:nbr_ch_pump])
        pump_ = CW(channels=(nbr_channels_pump - nbr_ch_pump),
                   peak_power=peak_powers_pump[nbr_ch_pump:],
                   center_lambda=center_lambdas_pump[nbr_ch_pump:])
        fiber = FiberYb(length=0.01, steps=10, REFL_SEED=REFL_SEED,
                        REFL_PUMP=REFL_PUMP, BISEED=True, BIPUMP=True,
                        save_all=True, alpha=[0.05], max_nbr_iter=2,
                        NOISE=NOISE)
        lt = Layout(Domain(samples_per_bit=64, noise_samples=10))
        lt.link_unidir((gssn[0], fiber[0]), (pump[0], fiber[2]),
                       (gssn_[0], fiber[1]), (pump_[0], fiber[3]))
        lt.run_all()

        return fiber.storages[0]

    return layout


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

nbr_channels = 3
peak_powers = [1.0, 1.0, 1.0]
center_lambdas = [1030., 1031., 1032.]

@pytest.mark.amp_noise
def test_split_noise_seed(split_noise_option_co_prop):
    """Should fail if the pump has non-zero noise."""
    # Environment creation
    outputs = split_noise_option_co_prop('seed_split', nbr_channels,
                                         peak_powers, center_lambdas)
    noise_power_input = outputs[0]
    noise_power_seed = outputs[1]
    noise_power_refl_seed = outputs[2]
    noise_power_pump = outputs[3]
    noise_power_refl_pump = outputs[4]
    # Testing
    assert not np.any(noise_power_input)
    assert np.any(noise_power_seed)
    assert np.any(noise_power_refl_seed)
    assert not np.any(noise_power_pump)
    assert not np.any(noise_power_refl_pump)


@pytest.mark.amp_noise
def test_split_noise_pump(split_noise_option_co_prop):
    """Should fail if the seed has non-zero noise."""
    # Environment creation
    outputs = split_noise_option_co_prop('pump_split', nbr_channels,
                                         peak_powers, center_lambdas)
    noise_power_input = outputs[0]
    noise_power_seed = outputs[1]
    noise_power_refl_seed = outputs[2]
    noise_power_pump = outputs[3]
    noise_power_refl_pump = outputs[4]
    # Testing
    assert not np.any(noise_power_input)
    assert not np.any(noise_power_seed)
    assert not np.any(noise_power_refl_seed)
    assert np.any(noise_power_pump)
    assert np.any(noise_power_refl_pump)


@pytest.mark.amp_noise
def test_split_noise_all(split_noise_option_co_prop):
    """Should fail if the pump or seed has non-zero noise."""
    # Environment creation
    outputs = split_noise_option_co_prop('all_split', nbr_channels,
                                         peak_powers, center_lambdas)
    noise_power_input = outputs[0]
    noise_power_seed = outputs[1]
    noise_power_refl_seed = outputs[2]
    noise_power_pump = outputs[3]
    noise_power_refl_pump = outputs[4]
    # Testing
    assert not np.any(noise_power_input)
    assert np.any(noise_power_seed)
    assert np.any(noise_power_refl_seed)
    assert np.any(noise_power_pump)
    assert np.any(noise_power_refl_pump)



@pytest.mark.amp_noise
def test_no_split_noise(split_noise_option_co_prop):
    """Should fail if the pump or seed has non-zero noise
    and all forward and all backward noise are resp. identical.
    """
    # Environment creation
    outputs = split_noise_option_co_prop('no_split', nbr_channels,
                                         peak_powers, center_lambdas)
    noise_power_input = outputs[0]
    noise_power_seed = outputs[1]
    noise_power_refl_seed = outputs[2]
    noise_power_pump = outputs[3]
    noise_power_refl_pump = outputs[4]
    # Testing
    assert not np.any(noise_power_input)
    assert np.any(noise_power_seed)
    assert np.any(noise_power_refl_seed)
    assert np.any(noise_power_pump)
    assert np.any(noise_power_refl_pump)
    assert np.array_equal(noise_power_seed, noise_power_pump)
    assert np.array_equal(noise_power_refl_seed, noise_power_refl_pump)



@pytest.mark.amp_noise
@pytest.mark.parametrize("split_noise_option",
    [('seed_split'), ('pump_split'), ('all_split'), ('no_split')
    ])
def test_split_noise_seed_consistency(split_noise_option_co_prop,
                                      split_noise_option):
    """Should fail if the sum of the noise of x channels ((x-1) null
    channels) is not the noise of one channel in the same conditions.
    """
    # Environment creation
    nbr_channels = 1
    peak_powers = [1.0]
    center_lambda = [1030.]
    outputs_1 = split_noise_option_co_prop(split_noise_option, nbr_channels,
                                           peak_powers, center_lambdas)
    noise_power_input_1 = outputs_1[0]
    noise_power_seed_1 = outputs_1[1]
    noise_power_refl_seed_1 = outputs_1[2]
    noise_power_pump_1 = outputs_1[3]
    noise_power_refl_pump_1 = outputs_1[4]
    nbr_channels = 3
    peak_powers = [0.0, 1.0, 0.0]
    center_lambda = [1025., 1030., 1040.]
    outputs_2 = split_noise_option_co_prop(split_noise_option, nbr_channels,
                                           peak_powers, center_lambdas)
    noise_power_input_2 = outputs_1[0]
    noise_power_seed_2 = outputs_2[1]
    noise_power_refl_seed_2 = outputs_2[2]
    noise_power_pump_2 = outputs_2[3]
    noise_power_refl_pump_2 = outputs_2[4]
    # Testing
    assert np.array_equal(noise_power_input_1, noise_power_input_2)
    assert np.array_equal(noise_power_seed_1, noise_power_seed_2)
    assert np.array_equal(noise_power_refl_seed_1, noise_power_refl_seed_2)
    assert np.array_equal(noise_power_pump_1, noise_power_pump_2)
    assert np.array_equal(noise_power_refl_pump_1, noise_power_refl_pump_2)




@pytest.mark.amp_scheme
def test_no_reflection(co_seed_co_pump_scheme,
                       co_seed_counter_pump_scheme,
                       counter_seed_co_pump_scheme,
                       counter_seed_counter_pump_scheme):
    """Should fail if reflection channels are non-zeros."""
    # Environment creation
    nbr_channels_seed = 5
    peak_powers_seed = [1.0, 0.7, 0.6, 0.8, 0.5]
    center_lambdas_seed = [1030., 1040., 1050., 1010., 1002.]
    nbr_channels_pump = 3
    peak_powers_pump = [0.01, 0.005, 0.02]
    center_lambdas_pump = [976., 940., 975.]
    storages = []
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, True))
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, False))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, True))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, False))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, True))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, False))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, True))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, False, False))
    # Testing
    total_nbr_ch = nbr_channels_seed + nbr_channels_pump
    for i in range(len(storages)):
        for j in range(nbr_channels_seed):
            assert not np.any(storages[i][total_nbr_ch+j])
        for j in range(nbr_channels_pump):
            assert not np.any(storages[i][total_nbr_ch+nbr_channels_seed+j])


@pytest.mark.amp_scheme
def test_no_seed_reflection(co_seed_co_pump_scheme,
                            co_seed_counter_pump_scheme,
                            counter_seed_co_pump_scheme,
                            counter_seed_counter_pump_scheme):
    """Should fail if reflection seed channels are non-zeros."""
    # Environment creation
    nbr_channels_seed = 5
    peak_powers_seed = [1.0, 0.7, 0.6, 0.8, 0.5]
    center_lambdas_seed = [1030., 1040., 1050., 1010., 1002.]
    nbr_channels_pump = 3
    peak_powers_pump = [0.01, 0.005, 0.02]
    center_lambdas_pump = [976., 940., 975.]
    storages = []
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, True))
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, False))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, True))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, False))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, True))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, False))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, True))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, False, True, False))
    # Testing
    total_nbr_ch = nbr_channels_seed + nbr_channels_pump
    for i in range(len(storages)):
        for j in range(nbr_channels_seed):
            assert not np.any(storages[i][total_nbr_ch+j])
        for j in range(nbr_channels_pump):
            assert np.any(storages[i][total_nbr_ch+nbr_channels_seed+j])


@pytest.mark.amp_scheme
def test_no_pump_reflection(co_seed_co_pump_scheme,
                            co_seed_counter_pump_scheme,
                            counter_seed_co_pump_scheme,
                            counter_seed_counter_pump_scheme):
    """Should fail if reflection pump channels are non-zeros."""
    # Environment creation
    nbr_channels_seed = 5
    peak_powers_seed = [1.0, 0.7, 0.6, 0.8, 0.5]
    center_lambdas_seed = [1030., 1040., 1050., 1010., 1002.]
    nbr_channels_pump = 3
    peak_powers_pump = [0.01, 0.005, 0.02]
    center_lambdas_pump = [976., 940., 975.]
    storages = []
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, True))
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, False))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, True))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, False))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, True))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, False))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, True))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, False, False))
    # Testing
    total_nbr_ch = nbr_channels_seed + nbr_channels_pump
    for i in range(len(storages)):
        for j in range(nbr_channels_seed):
            assert np.any(storages[i][total_nbr_ch+j])
        for j in range(nbr_channels_pump):
            assert not np.any(storages[i][total_nbr_ch+nbr_channels_seed+j])


@pytest.mark.amp_scheme
def test_no_seed_and_pump_reflection(co_seed_co_pump_scheme,
                                     co_seed_counter_pump_scheme,
                                     counter_seed_co_pump_scheme,
                                     counter_seed_counter_pump_scheme):
    """Should fail if reflection channels are zeros."""
    # Environment creation
    nbr_channels_seed = 5
    peak_powers_seed = [1.0, 0.7, 0.6, 0.8, 0.5]
    center_lambdas_seed = [1030., 1040., 1050., 1010., 1002.]
    nbr_channels_pump = 3
    peak_powers_pump = [0.01, 0.005, 0.02]
    center_lambdas_pump = [976., 940., 975.]
    storages = []
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, True))
    storages.append(co_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, False))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, True))
    storages.append(co_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, False))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, True))
    storages.append(counter_seed_co_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, False))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, True))
    storages.append(counter_seed_counter_pump_scheme(nbr_channels_seed,
                        peak_powers_seed, center_lambdas_seed,
                        nbr_channels_pump, peak_powers_pump,
                        center_lambdas_pump, True, True, False))
    # Testing
    total_nbr_ch = nbr_channels_seed + nbr_channels_pump
    for i in range(len(storages)):
        for j in range(nbr_channels_seed):
            assert np.any(storages[i][total_nbr_ch+j])
        for j in range(nbr_channels_pump):
            assert np.any(storages[i][total_nbr_ch+nbr_channels_seed+j])
