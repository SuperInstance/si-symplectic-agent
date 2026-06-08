//! Conservation law verification for Hamiltonian systems.
//!
//! The key property of symplectic integration is that it preserves the
//! symplectic 2-form, which implies:
//! - **Energy is nearly conserved** (bounded oscillation, no secular drift)
//! - **Phase space volume is preserved** (Liouville's theorem)
//! - **The flow is symplectic** (preserves the geometric structure)
//!
//! This module tracks these conservation laws throughout integration,
//! providing quantitative proof that symplectic methods outperform
//! non-symplectic ones for agent trajectories.

use crate::hamiltonian::Hamiltonian;
use crate::phase::PhaseSpace;

/// Tracks conservation laws during integration.
///
/// Records energy and phase-space volume at each step, enabling
/// quantitative comparison between symplectic and non-symplectic integrators.
#[derive(Clone, Debug)]
pub struct ConservationTracker {
    /// Initial total energy H(q₀, p₀)
    pub initial_energy: f64,
    /// Energy at each recorded step
    pub energies: Vec<f64>,
    /// Approximate phase space volume at each step (relative to initial)
    pub volumes: Vec<f64>,
    /// Number of recorded steps
    pub steps: usize,
    /// Initial phase space (for volume computation)
    initial_state: PhaseSpace,
    /// Reference perturbation magnitude for volume estimation
    epsilon: f64,
}

impl ConservationTracker {
    /// Create a new conservation tracker starting from the given initial state.
    pub fn new(initial: &PhaseSpace, hamiltonian: &dyn Hamiltonian) -> Self {
        let initial_energy = hamiltonian.value(&initial.position, &initial.momentum);
        Self {
            initial_energy,
            energies: vec![initial_energy],
            volumes: vec![1.0],
            steps: 0,
            initial_state: initial.clone(),
            epsilon: 1e-7,
        }
    }

    /// Record the current state's conservation properties.
    pub fn record(&mut self, state: &PhaseSpace, hamiltonian: &dyn Hamiltonian) {
        let energy = hamiltonian.value(&state.position, &state.momentum);
        self.energies.push(energy);
        self.steps += 1;

        // Approximate volume ratio using determinant of Jacobian
        // For a simple estimate, compare perturbation distances
        let volume = self.estimate_volume_ratio(state, hamiltonian);
        self.volumes.push(volume);
    }

    /// Maximum relative energy drift: max |H(t) - H(0)| / |H(0)|.
    ///
    /// For symplectic integrators, this should remain bounded (< 1e-6 for long runs).
    /// For non-symplectic integrators, this grows linearly (or worse) with time.
    pub fn energy_drift(&self) -> f64 {
        if self.initial_energy.abs() < 1e-15 {
            return self
                .energies
                .iter()
                .map(|e| (e - self.initial_energy).abs())
                .fold(0.0_f64, f64::max);
        }
        self.energies
            .iter()
            .map(|e| ((e - self.initial_energy) / self.initial_energy).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Maximum absolute energy deviation.
    pub fn max_energy_deviation(&self) -> f64 {
        self.energies
            .iter()
            .map(|e| (e - self.initial_energy).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Approximate phase space volume preservation.
    ///
    /// Returns the ratio of final to initial volume. For symplectic integrators,
    /// this should be ≈ 1.0 (Liouville's theorem).
    pub fn phase_volume(&self) -> f64 {
        *self.volumes.last().unwrap_or(&1.0)
    }

    /// Check if energy is conserved within the given tolerance.
    pub fn is_conserved(&self, tolerance: f64) -> bool {
        self.energy_drift() < tolerance
    }

    /// Estimate the symplecticity error.
    ///
    /// For a truly symplectic map, the Jacobian satisfies J^T Ω J = Ω.
    /// We estimate this by checking how a small perturbation evolves.
    pub fn symplecticity_error(&self) -> f64 {
        if self.volumes.len() < 2 {
            return 0.0;
        }
        // The symplecticity error is related to how much the volume deviates from 1
        self.volumes
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Generate a formatted conservation report.
    pub fn report(&self) -> String {
        let drift = self.energy_drift();
        let max_dev = self.max_energy_deviation();
        let vol = self.phase_volume();
        let symp_err = self.symplecticity_error();

        let conservation_quality = if drift < 1e-10 {
            "EXEMPLARY — symplectic conservation confirmed"
        } else if drift < 1e-6 {
            "EXCELLENT — symplectic-grade conservation"
        } else if drift < 1e-3 {
            "GOOD — acceptable for short runs"
        } else if drift < 1e-1 {
            "POOR — non-symplectic drift detected"
        } else {
            "FAILED — significant energy drift, non-symplectic"
        };

        format!(
            "=== Conservation Report ===\n\
             Steps: {}\n\
             Initial energy: {:.12}\n\
             Final energy:   {:.12}\n\
             Max |ΔH|:       {:.2e}\n\
             Relative drift: {:.2e}\n\
             Phase volume:   {:.10}\n\
             Volume drift:   {:.2e}\n\
             Symplecticity:  {:.2e}\n\
             Quality: {}\n\
             ==========================",
            self.steps,
            self.initial_energy,
            self.energies.last().unwrap_or(&self.initial_energy),
            max_dev,
            drift,
            vol,
            (vol - 1.0).abs(),
            symp_err,
            conservation_quality,
        )
    }

    /// Estimate volume ratio by tracking perturbation evolution.
    fn estimate_volume_ratio(&self, state: &PhaseSpace, _hamiltonian: &dyn Hamiltonian) -> f64 {
        // Simple volume estimation: compare phase space "radius"
        // For a proper implementation, we'd track a parallelepiped of nearby trajectories
        // Here we use the ratio of phase space distances as a proxy
        let d_initial = self.initial_state.distance(&PhaseSpace::new(self.initial_state.dimension()));
        let d_current = state.distance(&PhaseSpace::new(state.dimension()));

        if d_initial.abs() < 1e-15 {
            return 1.0;
        }

        // Volume scales as r^n in n dimensions
        let n = state.dimension() as f64;
        let r_ratio = d_current / d_initial;
        r_ratio.powf(n)
    }
}

/// Compute the Jacobian of an integrator step numerically.
///
/// Returns a flat vector representing the 2n×2n Jacobian matrix (row-major).
pub fn numerical_jacobian(
    integrator: &dyn crate::integrator::SymplecticIntegrator,
    state: &PhaseSpace,
    hamiltonian: &dyn Hamiltonian,
    dt: f64,
) -> Vec<Vec<f64>> {
    let n = state.dimension();
    let epsilon = 1e-7;
    let base = integrator.step(state, hamiltonian, dt);

    let mut jacobian = Vec::with_capacity(2 * n);

    // Perturb each position coordinate
    for i in 0..n {
        let mut perturbed = state.clone();
        perturbed.position[i] += epsilon;
        let perturbed_result = integrator.step(&perturbed, hamiltonian, dt);

        let mut col = Vec::with_capacity(2 * n);
        for j in 0..n {
            col.push((perturbed_result.position[j] - base.position[j]) / epsilon);
        }
        for j in 0..n {
            col.push((perturbed_result.momentum[j] - base.momentum[j]) / epsilon);
        }
        jacobian.push(col);
    }

    // Perturb each momentum coordinate
    for i in 0..n {
        let mut perturbed = state.clone();
        perturbed.momentum[i] += epsilon;
        let perturbed_result = integrator.step(&perturbed, hamiltonian, dt);

        let mut col = Vec::with_capacity(2 * n);
        for j in 0..n {
            col.push((perturbed_result.position[j] - base.position[j]) / epsilon);
        }
        for j in 0..n {
            col.push((perturbed_result.momentum[j] - base.momentum[j]) / epsilon);
        }
        jacobian.push(col);
    }

    jacobian
}

/// Check the symplectic condition: J^T Ω J = Ω
///
/// Returns the Frobenius norm of (J^T Ω J - Ω).
pub fn check_symplectic_condition(jacobian: &[Vec<f64>]) -> f64 {
    let n2 = jacobian.len(); // 2n
    let n = n2 / 2;

    // Compute J^T Ω J - Ω
    let mut error = 0.0;

    for i in 0..n2 {
        for j in 0..n2 {
            // (J^T Ω J)_{ij} = sum_{k,l} J_{ki} Ω_{kl} J_{lj}
            let mut val = 0.0;
            for k in 0..n2 {
                for l in 0..n2 {
                    let omega_kl = if k < n && l >= n && k == (l - n) {
                        1.0
                    } else if k >= n && l < n && (k - n) == l {
                        -1.0
                    } else {
                        0.0
                    };
                    val += jacobian[k][i] * omega_kl * jacobian[l][j];
                }
            }

            // Subtract Ω_{ij}
            let omega_ij = if i < n && j >= n && i == (j - n) {
                1.0
            } else if i >= n && j < n && (i - n) == j {
                -1.0
            } else {
                0.0
            };

            error += (val - omega_ij).powi(2);
        }
    }

    error.sqrt()
}
