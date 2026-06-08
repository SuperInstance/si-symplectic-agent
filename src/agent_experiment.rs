//! Agent trajectory experiments.
//!
//! This module provides high-level experiments demonstrating that agent trajectories
//! under symplectic integration conserve energy and reach objectives, while
//! non-symplectic integration leads to drift and instability.

use crate::conservation::{check_symplectic_condition, numerical_jacobian, ConservationTracker};
use crate::hamiltonian::{DoubleWellHamiltonian, Hamiltonian};
use crate::integrator::{RK4Baseline, StormerVerlet, SymplecticEuler, SymplecticIntegrator};
use crate::phase::PhaseSpace;

/// Result of an agent trajectory experiment.
#[derive(Clone, Debug)]
pub struct AgentResult {
    /// Full trajectory through phase space
    pub trajectory: Vec<PhaseSpace>,
    /// Conservation tracker with energy/volume data
    pub conservation: ConservationTracker,
    /// Final energy value
    pub final_energy: f64,
    /// Total energy drift (relative)
    pub energy_drift: f64,
    /// Path length through phase space
    pub path_length: f64,
    /// Name of the integrator used
    pub integrator_name: &'static str,
}

impl AgentResult {
    /// Final position in task space.
    pub fn final_position(&self) -> &[f64] {
        &self.trajectory.last().unwrap().position
    }

    /// Final momentum.
    pub fn final_momentum(&self) -> &[f64] {
        &self.trajectory.last().unwrap().momentum
    }

    /// Distance from initial to final position.
    pub fn displacement(&self) -> f64 {
        self.trajectory
            .first()
            .unwrap()
            .position_distance(self.trajectory.last().unwrap())
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        format!(
            "=== Agent Result ({}) ===\n\
             Steps: {}\n\
             Path length: {:.6}\n\
             Displacement: {:.6}\n\
             Final position: {:?}\n\
             Final momentum: {:?}\n\
             Energy drift: {:.2e}\n\
             \n{}",
            self.integrator_name,
            self.trajectory.len() - 1,
            self.path_length,
            self.displacement(),
            self.final_position(),
            self.final_momentum(),
            self.energy_drift,
            self.conservation.report(),
        )
    }
}

/// An experiment combining a Hamiltonian, integrator, and initial state.
pub struct AgentExperiment {
    /// The Hamiltonian system (defines agent dynamics)
    pub hamiltonian: Box<dyn Hamiltonian>,
    /// The integrator (symplectic or baseline)
    pub integrator: Box<dyn SymplecticIntegrator>,
    /// Initial phase space state
    pub initial: PhaseSpace,
}

impl AgentExperiment {
    /// Create a new experiment.
    pub fn new(
        hamiltonian: Box<dyn Hamiltonian>,
        integrator: Box<dyn SymplecticIntegrator>,
        initial: PhaseSpace,
    ) -> Self {
        Self {
            hamiltonian,
            integrator,
            initial,
        }
    }

    /// Run the experiment, integrating for the given number of steps.
    pub fn run(self, dt: f64, steps: usize) -> AgentResult {
        let mut conservation = ConservationTracker::new(&self.initial, self.hamiltonian.as_ref());

        let trajectory = self
            .integrator
            .integrate(&self.initial, self.hamiltonian.as_ref(), dt, steps);

        // Record conservation for all intermediate states
        for state in trajectory.iter().skip(1) {
            conservation.record(state, self.hamiltonian.as_ref());
        }

        let path_length: f64 = trajectory
            .windows(2)
            .map(|w| w[0].distance(&w[1]))
            .sum();

        let final_energy = trajectory
            .last()
            .unwrap()
            .energy(self.hamiltonian.as_ref());

        AgentResult {
            trajectory,
            final_energy,
            energy_drift: conservation.energy_drift(),
            conservation,
            path_length,
            integrator_name: self.integrator.name(),
        }
    }
}

/// Compare all integrators on the same Hamiltonian system.
///
/// Returns results for SymplecticEuler, StormerVerlet, and RK4Baseline.
/// The key comparison: symplectic integrators should conserve energy,
/// RK4 should drift.
pub fn compare_integrators<H: Hamiltonian + Clone + 'static>(
    hamiltonian: &H,
    initial: &PhaseSpace,
    dt: f64,
    steps: usize,
) -> Vec<(&'static str, AgentResult)> {
    let integrators: Vec<Box<dyn SymplecticIntegrator>> = vec![
        Box::new(SymplecticEuler),
        Box::new(StormerVerlet),
        Box::new(RK4Baseline),
    ];

    integrators
        .into_iter()
        .map(|integrator| {
            let name = integrator.name();
            let ham_clone: Box<dyn Hamiltonian> = Box::new(hamiltonian.clone());
            let experiment = AgentExperiment::new(ham_clone, integrator, initial.clone());
            let result = experiment.run(dt, steps);
            (name, result)
        })
        .collect()
}

/// Find the attractor (minimum energy state) by gradient descent in position space.
///
/// This represents the agent's goal — the minimum of the potential V(q).
pub fn find_attractor(
    hamiltonian: &dyn Hamiltonian,
    initial: &PhaseSpace,
) -> Vec<f64> {
    let mut q = initial.position.clone();
    let lr = 0.01;
    let max_iter = 10_000;
    let tolerance = 1e-10;

    for _ in 0..max_iter {
        let grad = hamiltonian.gradient_q(&q, &vec![0.0; q.len()]);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

        if grad_norm < tolerance {
            break;
        }

        for i in 0..q.len() {
            q[i] -= lr * grad[i];
        }
    }

    q
}

/// Compute a bifurcation diagram for the double-well Hamiltonian.
///
/// Varies the barrier parameter and finds the equilibrium positions.
/// Shows the bistable nature of the system.
pub fn bifurcation_diagram(
    _hamiltonian: &DoubleWellHamiltonian,
    param_range: (f64, f64),
    num_points: usize,
) -> Vec<(f64, Vec<f64>)> {
    let mut result = Vec::new();
    let (a_start, a_end) = param_range;

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let a = a_start + t * (a_end - a_start);

        // For V(q) = -a*q² + q⁴, critical points at q = 0 and q² = a/2
        // q = 0 is a maximum (for a > 0), q = ±sqrt(a/2) are minima
        let mut equilibria = vec![0.0_f64];

        if a > 0.0 {
            let q_min = (a / 2.0).sqrt();
            equilibria.push(q_min);
            equilibria.push(-q_min);
        }

        result.push((a, equilibria));
    }

    result
}

/// Compute the symplectic condition error for an integrator at a given state.
///
/// Returns ||J^T Ω J - Ω||_F where J is the integrator's Jacobian.
/// Should be ≈ 0 for symplectic integrators.
pub fn verify_symplecticity(
    integrator: &dyn SymplecticIntegrator,
    state: &PhaseSpace,
    hamiltonian: &dyn Hamiltonian,
    dt: f64,
) -> f64 {
    let jacobian = numerical_jacobian(integrator, state, hamiltonian, dt);
    check_symplectic_condition(&jacobian)
}
