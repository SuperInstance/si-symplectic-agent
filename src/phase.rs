//! Phase space representation for agent trajectories.
//!
//! In Hamiltonian mechanics, the state of a system is described by canonical coordinates
//! (q, p) — position and momentum. For agents:
//! - **q (position)**: current state in the task/parameter space
//! - **p (momentum)**: cognitive momentum — the agent's tendency to continue in its current direction
//!
//! Together they form phase space, and symplectic integration preserves the structure of this space.

/// Canonical phase space coordinates (q, p) for an agent trajectory.
///
/// Position q represents the agent's current state in task space.
/// Momentum p represents the agent's "cognitive momentum" — its tendency
/// to continue moving in its current direction.
#[derive(Clone, Debug)]
pub struct PhaseSpace {
    /// Position coordinates q (agent state in task space)
    pub position: Vec<f64>,
    /// Momentum coordinates p (cognitive momentum)
    pub momentum: Vec<f64>,
}

impl PhaseSpace {
    /// Create a zero-initialized phase space of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            position: vec![0.0; dimension],
            momentum: vec![0.0; dimension],
        }
    }

    /// Dimension of the phase space (number of position coordinates).
    pub fn dimension(&self) -> usize {
        self.position.len()
    }

    /// Compute total energy H(q, p) under the given Hamiltonian.
    pub fn energy(&self, hamiltonian: &dyn crate::hamiltonian::Hamiltonian) -> f64 {
        hamiltonian.value(&self.position, &self.momentum)
    }

    /// Euclidean distance between two phase space points.
    pub fn distance(&self, other: &PhaseSpace) -> f64 {
        let d_q: f64 = self
            .position
            .iter()
            .zip(other.position.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let d_p: f64 = self
            .momentum
            .iter()
            .zip(other.momentum.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (d_q + d_p).sqrt()
    }

    /// Kinetic energy: ||p||² / 2
    pub fn kinetic_energy(&self) -> f64 {
        self.momentum.iter().map(|p| p * p).sum::<f64>() / 2.0
    }

    /// Potential energy V(q) under the given potential function.
    pub fn potential_energy(&self, potential: fn(&[f64]) -> f64) -> f64 {
        potential(&self.position)
    }

    /// Position-space Euclidean distance.
    pub fn position_distance(&self, other: &PhaseSpace) -> f64 {
        self.position
            .iter()
            .zip(other.position.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Momentum magnitude.
    pub fn momentum_norm(&self) -> f64 {
        self.momentum.iter().map(|p| p * p).sum::<f64>().sqrt()
    }
}

impl PartialEq for PhaseSpace {
    fn eq(&self, other: &Self) -> bool {
        self.position.len() == other.position.len()
            && self.momentum.len() == other.momentum.len()
            && self
                .position
                .iter()
                .zip(other.position.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
            && self
                .momentum
                .iter()
                .zip(other.momentum.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
    }
}
