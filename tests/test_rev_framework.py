# test_rev_framework.py
import numpy as np
import pytest
import importlib

# Try to import your module as 'aic' (your filename), fallback to 'rev_framework'
try:
    m = importlib.import_module("aic")
except ModuleNotFoundError:
    m = importlib.import_module("rev_framework")


def test_partial_trace_identity():
    """Reduced state stays a valid density matrix with trace 1."""
    rho = np.eye(4) / 4  # maximally mixed on 2 qubits
    rA = m._partial_trace(rho, keep_indices=[0], dims=[2, 2])
    assert rA.shape == (2, 2)
    assert np.isclose(np.trace(rA).real, 1.0, atol=1e-12)


def test_phi_product_vs_ghz():
    """Φq is ~0 for product states and ~1 bit for GHZ-like state (4 qubits)."""
    iiqt = m.IntegratedInformationQuantumTheory()

    # Product state: |0000>
    psi0 = np.zeros(16); psi0[0] = 1.0
    rho0 = np.outer(psi0, psi0.conj())
    out0 = iiqt.calculate_consciousness_level(rho0)
    assert np.isclose(out0["consciousness_level"], 0.0, atol=1e-9)

    # GHZ state: (|0000> + |1111>)/sqrt(2)
    ghz = np.zeros(16); ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    rhog = np.outer(ghz, ghz.conj())
    outg = iiqt.calculate_consciousness_level(rhog)
    assert outg["consciousness_level"] >= 0.95  # small slack ok


def test_phi_requires_power_of_two_dimension():
    """Non-2^n dimension should raise a helpful error."""
    iiqt = m.IntegratedInformationQuantumTheory()
    bad = np.eye(6) / 6
    with pytest.raises(ValueError):
        iiqt.calculate_consciousness_level(bad)


def test_binding_increases_at_criticality():
    """Binding (pairwise MI) should be >= at criticality vs subcritical."""
    np.random.seed(123)
    cbm = m.ConsciousnessBindingMechanism(n_neurons=120, quiet=True)
    metrics, _ = cbm.unified_experience_from_distributed_processing(steps=40)
    assert metrics["critical_binding"] >= metrics["subcritical_binding"] - 1e-6


def test_phase_transition_metrics_present():
    """Phase transition demo returns sane metrics for all regimes."""
    np.random.seed(123)
    eq = m.EmergentQuantumness(n_neurons=100, quiet=True)
    results, _ = eq.demonstrate_phase_transition(steps=40)
    for regime in ("Subcritical", "Critical", "Supercritical"):
        assert regime in results
        se = results[regime]["superposition_entropy"]
        mi = results[regime]["entanglement_mutual_info"]
        assert se >= 0.0 and mi >= 0.0


def test_predictor_stable_experience_hashing():
    """Experience prediction should be stable across repeated calls."""
    pred = m.UnprecedentedPredictions()
    v = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    a = pred.predict_subjective_experience(v)
    b = pred.predict_subjective_experience(v)
    assert a == b
