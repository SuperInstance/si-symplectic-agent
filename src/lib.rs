//! # si-symplectic-agent
//!
//! Proof of concept: agent trajectories as Hamiltonian flows.
//!
//! This crate demonstrates that Cyberloop's Riemannian manifold control for agent
//! steps can be formalized using symplectic geometry, yielding energy-conserving,
//! phase-space-volume-preserving trajectories that provably cannot drift off objective.
//!
//! ## Key Insight
//!
//! Cyberloop v3.0 uses "Riemannian manifold control" for agent steps.
//! Symplectic geometry ⊂ Riemannian geometry with extra structure.
//! A symplectic manifold preserves phase space volume (Liouville's theorem).
//! Hamiltonian flows conserve energy.
//! **So agent steps as Hamiltonian flows = energy-conserving, volume-preserving trajectories.**
//! The agent CANNOT drift off objective because energy is conserved.

pub mod phase;
pub mod hamiltonian;
pub mod integrator;
pub mod conservation;
pub mod agent_experiment;

pub use phase::PhaseSpace;
pub use hamiltonian::{Hamiltonian, QuadraticHamiltonian, DoubleWellHamiltonian, AgentHamiltonian};
pub use integrator::{SymplecticIntegrator, SymplecticEuler, StormerVerlet, RK4Baseline};
pub use conservation::ConservationTracker;
pub use agent_experiment::{AgentExperiment, AgentResult};

#[cfg(test)]
mod tests;
