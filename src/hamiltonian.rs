//! Hamiltonian systems for agent trajectories.
//!
//! A Hamiltonian H(q, p) defines the total energy of the system. The equations of motion
//! are Hamilton's equations:
//!   dq/dt =  ∂H/∂p   (position evolves along gradient of H w.r.t. momentum)
//!   dp/dt = -∂H/∂q   (momentum evolves against gradient of H w.r.t. position)
//!
//! For agents:
//! - H = kinetic + potential = ||p||²/2m + V(q)
//! - V(q) is the task objective (minimize V = reach goal)
//! - p is cognitive momentum

use crate::phase::PhaseSpace;

/// A Hamiltonian system defining agent dynamics.
///
/// The Hamiltonian H(q, p) must provide:
/// - Total energy value
/// - Gradient with respect to position q (∂H/∂q)
/// - Gradient with respect to momentum p (∂H/∂p)
pub trait Hamiltonian {
    /// Compute H(q, p) — total energy.
    fn value(&self, q: &[f64], p: &[f64]) -> f64;

    /// Compute ∂H/∂q — gradient with respect to position.
    fn gradient_q(&self, q: &[f64], p: &[f64]) -> Vec<f64>;

    /// Compute ∂H/∂p — gradient with respect to momentum.
    fn gradient_p(&self, q: &[f64], p: &[f64]) -> Vec<f64>;

    /// Compute the Hamiltonian vector field at a given phase space point.
    /// Returns (dq/dt, dp/dt) = (∂H/∂p, -∂H/∂q).
    fn vector_field(&self, state: &PhaseSpace) -> (Vec<f64>, Vec<f64>) {
        let dq = self.gradient_p(&state.position, &state.momentum);
        let dp = self
            .gradient_q(&state.position, &state.momentum)
            .iter()
            .map(|x| -x)
            .collect();
        (dq, dp)
    }
}

/// Quadratic (harmonic oscillator) Hamiltonian: H = p²/2m + k*q²/2
///
/// This is the simplest non-trivial Hamiltonian with known exact solutions.
/// The agent oscillates around the origin — a stable equilibrium.
#[derive(Clone, Debug)]
pub struct QuadraticHamiltonian {
    /// Mass parameter m
    pub mass: f64,
    /// Stiffness parameter k
    pub stiffness: f64,
}

impl QuadraticHamiltonian {
    pub fn new(mass: f64, stiffness: f64) -> Self {
        Self { mass, stiffness }
    }
}

impl Hamiltonian for QuadraticHamiltonian {
    fn value(&self, q: &[f64], p: &[f64]) -> f64 {
        let kinetic: f64 = p.iter().map(|pi| pi * pi).sum::<f64>() / (2.0 * self.mass);
        let potential: f64 = self.stiffness * q.iter().map(|qi| qi * qi).sum::<f64>() / 2.0;
        kinetic + potential
    }

    fn gradient_q(&self, q: &[f64], _p: &[f64]) -> Vec<f64> {
        q.iter().map(|qi| self.stiffness * qi).collect()
    }

    fn gradient_p(&self, _q: &[f64], p: &[f64]) -> Vec<f64> {
        p.iter().map(|pi| pi / self.mass).collect()
    }
}

/// Double-well Hamiltonian: H = p²/2m - a*||q||² + ||q||⁴
///
/// Creates a bistable potential with two minima. The agent must "choose"
/// which well to settle into — modeling decision-making in task completion.
#[derive(Clone, Debug)]
pub struct DoubleWellHamiltonian {
    /// Barrier height parameter a
    pub barrier: f64,
    /// Mass parameter m
    pub mass: f64,
}

impl DoubleWellHamiltonian {
    pub fn new(barrier: f64, mass: f64) -> Self {
        Self { barrier, mass }
    }
}

impl Hamiltonian for DoubleWellHamiltonian {
    fn value(&self, q: &[f64], p: &[f64]) -> f64 {
        let kinetic: f64 = p.iter().map(|pi| pi * pi).sum::<f64>() / (2.0 * self.mass);
        let q_norm_sq: f64 = q.iter().map(|qi| qi * qi).sum();
        let potential = -self.barrier * q_norm_sq + q_norm_sq * q_norm_sq;
        kinetic + potential
    }

    fn gradient_q(&self, q: &[f64], _p: &[f64]) -> Vec<f64> {
        let q_norm_sq: f64 = q.iter().map(|qi| qi * qi).sum();
        q.iter()
            .map(|qi| -2.0 * self.barrier * qi + 4.0 * q_norm_sq * qi)
            .collect()
    }

    fn gradient_p(&self, _q: &[f64], p: &[f64]) -> Vec<f64> {
        p.iter().map(|pi| pi / self.mass).collect()
    }
}

/// Agent Hamiltonian: H = ||p||²/2m + V(q)
///
/// The task objective V(q) becomes the potential energy.
/// Minimizing V = reaching the goal. The agent's cognitive momentum p
/// carries it through the task space, and the Hamiltonian flow ensures
/// it follows energy-conserving trajectories toward the objective.
///
/// This is the direct bridge from Cyberloop's Riemannian control to symplectic integration.
#[derive(Clone)]
pub struct AgentHamiltonian {
    /// Task objective V(q) — minimize this to reach goal
    pub objective: fn(&[f64]) -> f64,
    /// Gradient of the objective ∇V(q)
    pub gradient: fn(&[f64]) -> Vec<f64>,
    /// Mass parameter — controls momentum scaling
    pub mass: f64,
}

impl AgentHamiltonian {
    pub fn new(
        objective: fn(&[f64]) -> f64,
        gradient: fn(&[f64]) -> Vec<f64>,
        mass: f64,
    ) -> Self {
        Self {
            objective,
            gradient,
            mass,
        }
    }
}

impl Hamiltonian for AgentHamiltonian {
    fn value(&self, q: &[f64], p: &[f64]) -> f64 {
        let kinetic: f64 = p.iter().map(|pi| pi * pi).sum::<f64>() / (2.0 * self.mass);
        let potential = (self.objective)(q);
        kinetic + potential
    }

    fn gradient_q(&self, q: &[f64], _p: &[f64]) -> Vec<f64> {
        (self.gradient)(q)
    }

    fn gradient_p(&self, _q: &[f64], p: &[f64]) -> Vec<f64> {
        p.iter().map(|pi| pi / self.mass).collect()
    }
}
