import numpy as np

from slicing_sim.dataset_gen import (
    build_covert_use_mask,
    generate_sample,
    generate_dataset,
)
from slicing_sim.ofdm_grid import (
    OFDMGridConfig,
    NetworkSlicingSimulator,
)


N_SYMBOLS = 200
N_SUBCARRIERS = 64
N_COVERT_BITS = 32


def test_covert_use_mask_shape_count_and_target_validity():
    """The covert mask must contain exactly 32 valid eMBB positions."""

    cfg = OFDMGridConfig(
        n_subcarriers=N_SUBCARRIERS,
        n_symbols=N_SYMBOLS,
        snr_db=10,
        seed=12345,
    )

    simulator = NetworkSlicingSimulator(cfg)
    allocations = simulator.allocate_slices()

    target_mask = allocations["eMBB"].subcarrier_mask

    covert_mask = build_covert_use_mask(
        target_mask=target_mask,
        n_covert_bits=N_COVERT_BITS,
        seed=12445,
    )

    assert covert_mask.shape == (N_SYMBOLS, N_SUBCARRIERS)
    assert covert_mask.dtype == bool

    # Exactly one selected position per covert bit.
    assert np.count_nonzero(covert_mask) == N_COVERT_BITS

    # Every covert position must belong to eMBB.
    assert np.all(~covert_mask | target_mask)


def test_covert_use_mask_is_reproducible():
    """Same target mask + same seed must produce the same covert mask."""

    cfg = OFDMGridConfig(
        n_subcarriers=N_SUBCARRIERS,
        n_symbols=N_SYMBOLS,
        snr_db=10,
        seed=12345,
    )

    simulator = NetworkSlicingSimulator(cfg)
    allocations = simulator.allocate_slices()

    target_mask = allocations["eMBB"].subcarrier_mask

    mask_a = build_covert_use_mask(
        target_mask=target_mask,
        n_covert_bits=N_COVERT_BITS,
        seed=12445,
    )

    mask_b = build_covert_use_mask(
        target_mask=target_mask,
        n_covert_bits=N_COVERT_BITS,
        seed=12445,
    )

    assert np.array_equal(mask_a, mask_b)


def test_covert_use_mask_changes_with_seed():
    """Different seeds should normally produce different selections."""

    cfg = OFDMGridConfig(
        n_subcarriers=N_SUBCARRIERS,
        n_symbols=N_SYMBOLS,
        snr_db=10,
        seed=12345,
    )

    simulator = NetworkSlicingSimulator(cfg)
    allocations = simulator.allocate_slices()

    target_mask = allocations["eMBB"].subcarrier_mask

    mask_a = build_covert_use_mask(
        target_mask=target_mask,
        n_covert_bits=N_COVERT_BITS,
        seed=12445,
    )

    mask_b = build_covert_use_mask(
        target_mask=target_mask,
        n_covert_bits=N_COVERT_BITS,
        seed=99999,
    )

    assert not np.array_equal(mask_a, mask_b)


def test_covert_use_mask_is_temporally_distributed():
    """
    The 32 covert uses must not collapse into one OFDM symbol.

    The implementation divides the 200 symbols into 32 temporal
    regions and selects one valid eMBB position per region.
    """

    cfg = OFDMGridConfig(
        n_subcarriers=N_SUBCARRIERS,
        n_symbols=N_SYMBOLS,
        snr_db=10,
        seed=12345,
    )

    simulator = NetworkSlicingSimulator(cfg)
    allocations = simulator.allocate_slices()

    target_mask = allocations["eMBB"].subcarrier_mask

    covert_mask = build_covert_use_mask(
        target_mask=target_mask,
        n_covert_bits=N_COVERT_BITS,
        seed=12445,
    )

    selected_symbols = np.where(np.any(covert_mask, axis=1))[0]

    # Exactly 32 different OFDM symbols should contain the covert uses.
    assert len(selected_symbols) == N_COVERT_BITS

    # No symbol should contain more than one covert use.
    per_symbol = np.sum(covert_mask, axis=1)
    assert np.max(per_symbol) == 1


def test_generate_sample_shapes_and_attack_effects():
    """One generated sample must have the expected shape and attacks
    must actually perturb the clean grid."""

    clean, nonadaptive, adaptive = generate_sample(
        snr_db=10,
        seed=12345,
    )

    expected_shape = (N_SYMBOLS, N_SUBCARRIERS)

    assert clean.shape == expected_shape
    assert nonadaptive.shape == expected_shape
    assert adaptive.shape == expected_shape

    assert clean.dtype == np.float64
    assert nonadaptive.dtype == np.float64
    assert adaptive.dtype == np.float64

    assert np.isfinite(clean).all()
    assert np.isfinite(nonadaptive).all()
    assert np.isfinite(adaptive).all()

    # Neither attacker should return an unchanged grid.
    assert not np.array_equal(clean, nonadaptive)
    assert not np.array_equal(clean, adaptive)


def test_generate_sample_is_reproducible():
    """Same SNR + seed must reproduce all three grids exactly."""

    sample_a = generate_sample(
        snr_db=10,
        seed=12345,
    )

    sample_b = generate_sample(
        snr_db=10,
        seed=12345,
    )

    for a, b in zip(sample_a, sample_b):
        assert np.array_equal(a, b)


def test_generate_dataset_shapes_labels_and_snr():
    """
    Use a tiny dataset so pytest stays fast.

    2 SNR values × 2 samples × 3 classes = 12 samples.
    """

    snr_values = [0, 10]
    samples_per_snr = 2

    X, y, snr_arr = generate_dataset(
        snr_values=snr_values,
        samples_per_snr=samples_per_snr,
        base_seed=12345,
    )

    expected_samples = 2 * 2 * 3

    assert X.shape == (
        expected_samples,
        N_SYMBOLS,
        N_SUBCARRIERS,
    )

    assert y.shape == (expected_samples,)
    assert snr_arr.shape == (expected_samples,)

    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert snr_arr.dtype == np.float32

    assert np.isfinite(X).all()

    # Exactly three classes.
    assert set(np.unique(y)) == {0, 1, 2}

    # Two samples per class at each SNR.
    for snr in snr_values:
        for label in [0, 1, 2]:
            count = np.sum(
                (snr_arr == snr) & (y == label)
            )
            assert count == samples_per_snr


def test_generate_dataset_is_reproducible():
    """Same dataset parameters and seed must produce identical arrays."""

    kwargs = dict(
        snr_values=[0, 10],
        samples_per_snr=2,
        base_seed=12345,
    )

    X1, y1, snr1 = generate_dataset(**kwargs)
    X2, y2, snr2 = generate_dataset(**kwargs)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(snr1, snr2)


def test_generate_dataset_changes_with_seed():
    """Changing the base seed should change the generated dataset."""

    kwargs = dict(
        snr_values=[0, 10],
        samples_per_snr=2,
    )

    X1, y1, snr1 = generate_dataset(
        **kwargs,
        base_seed=12345,
    )

    X2, y2, snr2 = generate_dataset(
        **kwargs,
        base_seed=54321,
    )

    # Labels and SNR metadata are intentionally identical because
    # the experimental design is the same.
    assert np.array_equal(y1, y2)
    assert np.array_equal(snr1, snr2)

    # Actual simulated grids must differ.
    assert not np.array_equal(X1, X2)
