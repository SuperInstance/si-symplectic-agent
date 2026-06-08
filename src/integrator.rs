//! Symplectic integrators for Hamiltonian systems.
//!
//! Symplectic integrators preserve the symplectic 2-form ω = dq ∧ dp, which means:
//! - Phase space volume is preserved (Liouville's theorem)
//! - Energy is nearly conserved (bounded oscillation, no drift)
//! - The integrator itself is a symplectic map
//!
//! This is crucial for agent trajectories: symplectic integration guarantees
//! the agent cannot drift off objective because energy cannot grow unboundedly.
//!
//! We also include a non-symplectic RK4 baseline to demonstrate the difference.

use crate::hamiltonian::Hamiltonian;
use crate::phase::PhaseSpace;

/// A symplectic (or comparative) integrator for Hamiltonian systems.
pub trait SymplecticIntegrator {
    /// Perform a single integration step.
    fn step(&self, state: &PhaseSpace, hamiltonian: &dyn Hamiltonian, dt: f64) -> PhaseSpace;

    /// Integrate for multiple steps, returning the full trajectory.
    fn integrate(
        &self,
        initial: &PhaseSpace,
        hamiltonian: &dyn Hamiltonian,
        dt: f64,
        steps: usize,
    ) -> Vec<PhaseSpace> {
        let mut trajectory = Vec::with_capacity(steps + 1);
        trajectory.push(initial.clone());
        let mut state = initial.clone();
        for _ in 0..steps {
            state = self.step(&state, hamiltonian, dt);
            trajectory.push(state.clone());
        }
        trajectory
    }

    /// Name of this integrator for reporting.
    fn name(&self) -> &'static str;
}

/// Symplectic Euler integrator (first-order).
///
/// This is the simplest symplectic method. It's explicit and first-order accurate.
///
/// Algorithm:
///   q' = q + dt · ∂H/∂p(q, p)
///   p' = p - dt · ∂H/∂q(q', p)
///
/// Notice q' is used in the momentum update — this "semi-implicit" structure
/// is what makes it symplectic.
#[derive(Clone, Debug)]
pub struct SymplecticEuler;

impl SymplecticIntegrator for SymplecticEuler {
    fn step(&self, state: &PhaseSpace, hamiltonian: &dyn Hamiltonian, dt: f64) -> PhaseSpace {
        // Update position first using current momentum
        let grad_p = hamiltonian.gradient_p(&state.position, &state.momentum);
        let new_q: Vec<f64> = state
            .position
            .iter()
            .zip(grad_p.iter())
            .map(|(qi, dpi)| qi + dt * dpi)
            .collect();

        // Update momentum using NEW position (semi-implicit)
        let grad_q = hamiltonian.gradient_q(&new_q, &state.momentum);
        let new_p: Vec<f64> = state
            .momentum
            .iter()
            .zip(grad_q.iter())
            .map(|(pi, dqi)| pi - dt * dqi)
            .collect();

        PhaseSpace {
            position: new_q,
            momentum: new_p,
        }
    }

    fn name(&self) -> &'static str {
        "SymplecticEuler"
    }
}

/// Störmer-Verlet (leapfrog) integrator (second-order symplectic).
///
/// This is the gold standard for Hamiltonian integration. Second-order accurate
/// and symplectic — it preserves the qualitative features of the Hamiltonian flow
/// much better than higher-order non-symplectic methods.
///
/// Algorithm (velocity Verlet form):
///   p_{1/2} = p - (dt/2) · ∂H/∂q(q, p)
///   q'      = q + dt · ∂H/∂p(q, p_{1/2})
///   p'      = p_{1/2} - (dt/2) · ∂H/∂q(q', p_{1/2})
#[derive(Clone, Debug)]
pub struct StormerVerlet;

impl SymplecticIntegrator for StormerVerlet {
    fn step(&self, state: &PhaseSpace, hamiltonian: &dyn Hamiltonian, dt: f64) -> PhaseSpace {
        // Half-step momentum
        let grad_q = hamiltonian.gradient_q(&state.position, &state.momentum);
        let p_half: Vec<f64> = state
            .momentum
            .iter()
            .zip(grad_q.iter())
            .map(|(pi, dqi)| pi - 0.5 * dt * dqi)
            .collect();

        // Full-step position using half-step momentum
        let grad_p = hamiltonian.gradient_p(&state.position, &p_half);
        let new_q: Vec<f64> = state
            .position
            .iter()
            .zip(grad_p.iter())
            .map(|(qi, dpi)| qi + dt * dpi)
            .collect();

        // Half-step momentum using new position
        let grad_q_new = hamiltonian.gradient_q(&new_q, &p_half);
        let new_p: Vec<f64> = p_half
            .iter()
            .zip(grad_q_new.iter())
            .map(|(pi, dqi)| pi - 0.5 * dt * dqi)
            .collect();

        PhaseSpace {
            position: new_q,
            momentum: new_p,
        }
    }

    fn name(&self) -> &'static str {
        "StormerVerlet"
    }
}

/// 4th-order Runge-Kutta integrator (non-symplectic baseline).
///
/// RK4 is higher-order accurate but NOT symplectic. Over long integrations,
/// it will exhibit energy drift. This serves as a comparison to demonstrate
/// why symplectic methods are essential for agent trajectories.
#[derive(Clone, Debug)]
pub struct RK4Baseline;

impl SymplecticIntegrator for RK4Baseline {
    fn step(&self, state: &PhaseSpace, hamiltonian: &dyn Hamiltonian, dt: f64) -> PhaseSpace {
        let n = state.dimension();

        // k1
        let (dq1, dp1) = hamiltonian.vector_field(state);
        let s1 = PhaseSpace {
            position: state
                .position
                .iter()
                .zip(dq1.iter())
                .map(|(qi, d)| qi + 0.5 * dt * d)
                .collect(),
            momentum: state
                .momentum
                .iter()
                .zip(dp1.iter())
                .map(|(pi, d)| pi + 0.5 * dt * d)
                .collect(),
        };

        // k2
        let (dq2, dp2) = hamiltonian.vector_field(&s1);
        let s2 = PhaseSpace {
            position: state
                .position
                .iter()
                .zip(dq2.iter())
                .map(|(qi, d)| qi + 0.5 * dt * d)
                .collect(),
            momentum: state
                .momentum
                .iter()
                .zip(dp2.iter())
                .map(|(pi, d)| pi + 0.5 * dt * d)
                .collect(),
        };

        // k3
        let (dq3, dp3) = hamiltonian.vector_field(&s2);
        let s3 = PhaseSpace {
            position: state
                .position
                .iter()
                .zip(dq3.iter())
                .map(|(qi, d)| qi + dt * d)
                .collect(),
            momentum: state
                .momentum
                .iter()
                .zip(dp3.iter())
                .map(|(pi, d)| pi + dt * d)
                .collect(),
        };

        // k4
        let (dq4, dp4) = hamiltonian.vector_field(&s3);

        // Combine
        let new_q: Vec<f64> = (0..n)
            .map(|i| {
                state.position[i]
                    + (dt / 6.0) * (dq1[i] + 2.0 * dq2[i] + 2.0 * dq3[i] + dq4[i])
            })
            .collect();
        let new_p: Vec<f64> = (0..n)
            .map(|i| {
                state.momentum[i]
                    + (dt / 6.0) * (dp1[i] + 2.0 * dp2[i] + 2.0 * dp3[i] + dp4[i])
            })
            .collect();

        PhaseSpace {
            position: new_q,
            momentum: new_p,
        }
    }

    fn name(&self) -> &'static str {
        "RK4Baseline"
    }
}
