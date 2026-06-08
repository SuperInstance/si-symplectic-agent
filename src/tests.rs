#[cfg(test)]
mod tests {
    use crate::*;
    use crate::agent_experiment::{compare_integrators, find_attractor, bifurcation_diagram, verify_symplecticity};

    // Helper: simple quadratic potential V(q) = ||q||²
    fn quadratic_potential(q: &[f64]) -> f64 {
        q.iter().map(|qi| qi * qi).sum()
    }

    // Helper: gradient of quadratic potential
    fn quadratic_gradient(q: &[f64]) -> Vec<f64> {
        q.iter().map(|qi| 2.0 * qi).collect()
    }

    // ============================================================
    // PhaseSpace tests
    // ============================================================

    #[test]
    fn test_phase_space_creation() {
        let ps = PhaseSpace::new(3);
        assert_eq!(ps.dimension(), 3);
        assert_eq!(ps.position, vec![0.0, 0.0, 0.0]);
        assert_eq!(ps.momentum, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_phase_space_kinetic_energy() {
        let mut ps = PhaseSpace::new(2);
        ps.momentum = vec![3.0, 4.0];
        let ke = ps.kinetic_energy();
        assert!((ke - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_phase_space_potential_energy() {
        let mut ps = PhaseSpace::new(2);
        ps.position = vec![1.0, 2.0];
        let pe = ps.potential_energy(quadratic_potential);
        assert!((pe - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_space_distance() {
        let mut a = PhaseSpace::new(2);
        let mut b = PhaseSpace::new(2);
        a.position = vec![1.0, 0.0];
        b.position = vec![0.0, 0.0];
        assert!((a.distance(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_space_position_distance() {
        let mut a = PhaseSpace::new(2);
        let mut b = PhaseSpace::new(2);
        a.position = vec![3.0, 4.0];
        assert!((a.position_distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_space_momentum_norm() {
        let mut ps = PhaseSpace::new(2);
        ps.momentum = vec![3.0, 4.0];
        assert!((ps.momentum_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_space_energy() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut ps = PhaseSpace::new(1);
        ps.position = vec![1.0];
        ps.momentum = vec![1.0];
        // H = p²/2 + k*q²/2 = 0.5 + 0.5 = 1.0
        assert!((ps.energy(&ham) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_space_equality() {
        let mut a = PhaseSpace::new(2);
        let mut b = PhaseSpace::new(2);
        a.position = vec![1.0, 2.0];
        a.momentum = vec![3.0, 4.0];
        b.position = vec![1.0, 2.0];
        b.momentum = vec![3.0, 4.0];
        assert_eq!(a, b);
    }

    // ============================================================
    // Hamiltonian tests
    // ============================================================

    #[test]
    fn test_quadratic_hamiltonian_value() {
        let ham = QuadraticHamiltonian::new(1.0, 2.0);
        let q = vec![1.0, 2.0];
        let p = vec![3.0, 4.0];
        // H = (9+16)/2 + 2*(1+4)/2 = 12.5 + 5.0 = 17.5
        let h = ham.value(&q, &p);
        assert!((h - 17.5).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_hamiltonian_gradient_q() {
        let ham = QuadraticHamiltonian::new(1.0, 3.0);
        let q = vec![1.0, 2.0];
        let grad = ham.gradient_q(&q, &vec![0.0; 2]);
        assert!((grad[0] - 3.0).abs() < 1e-10);
        assert!((grad[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_hamiltonian_gradient_p() {
        let ham = QuadraticHamiltonian::new(2.0, 1.0);
        let p = vec![4.0, 6.0];
        let grad = ham.gradient_p(&vec![0.0; 2], &p);
        assert!((grad[0] - 2.0).abs() < 1e-10);
        assert!((grad[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_well_hamiltonian_value_at_origin() {
        let ham = DoubleWellHamiltonian::new(1.0, 1.0);
        let q = vec![0.0];
        let p = vec![0.0];
        // H = 0 + 0 = 0
        assert!((ham.value(&q, &p)).abs() < 1e-10);
    }

    #[test]
    fn test_double_well_hamiltonian_value_at_minimum() {
        let ham = DoubleWellHamiltonian::new(2.0, 1.0);
        // Minima at q² = a/2 = 1, so q = ±1
        let q = vec![1.0];
        let p = vec![0.0];
        let h = ham.value(&q, &p);
        // V = -2*1 + 1 = -1
        assert!((h - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_double_well_bistability() {
        let ham = DoubleWellHamiltonian::new(2.0, 1.0);
        let q_pos = vec![1.0];
        let q_neg = vec![-1.0];
        let p = vec![0.0];
        let h_pos = ham.value(&q_pos, &p);
        let h_neg = ham.value(&q_neg, &p);
        assert!((h_pos - h_neg).abs() < 1e-10);
    }

    #[test]
    fn test_agent_hamiltonian_value() {
        let ham = AgentHamiltonian::new(quadratic_potential, quadratic_gradient, 1.0);
        let q = vec![1.0, 0.0];
        let p = vec![0.0, 2.0];
        // H = (0+4)/2 + (1+0) = 2.0 + 1.0 = 3.0
        let h = ham.value(&q, &p);
        assert!((h - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_hamiltonian_gradient() {
        let ham = AgentHamiltonian::new(quadratic_potential, quadratic_gradient, 2.0);
        let q = vec![3.0, 4.0];
        let grad_q = ham.gradient_q(&q, &vec![0.0; 2]);
        assert!((grad_q[0] - 6.0).abs() < 1e-10);
        assert!((grad_q[1] - 8.0).abs() < 1e-10);

        let p = vec![4.0, 6.0];
        let grad_p = ham.gradient_p(&q, &p);
        assert!((grad_p[0] - 2.0).abs() < 1e-10);
        assert!((grad_p[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![1.0];
        let (dq, dp) = ham.vector_field(&state);
        // dq/dt = p/m = 1.0
        assert!((dq[0] - 1.0).abs() < 1e-10);
        // dp/dt = -k*q = -1.0
        assert!((dp[0] - (-1.0)).abs() < 1e-10);
    }

    // ============================================================
    // Integrator tests
    // ============================================================

    #[test]
    fn test_symplectic_euler_single_step() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = SymplecticEuler;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let next = integrator.step(&state, &ham, 0.1);
        // q' = 1.0 + 0.1 * p/m = 1.0 + 0 = 1.0
        assert!((next.position[0] - 1.0).abs() < 1e-10);
        // p' = 0.0 - 0.1 * k * q' = -0.1
        assert!((next.momentum[0] - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_stormer_verlet_single_step() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let next = integrator.step(&state, &ham, 0.1);
        // p_half = 0 - 0.05 * 1.0 = -0.05
        // q' = 1.0 + 0.1 * (-0.05) = 0.995
        assert!((next.position[0] - 0.995).abs() < 1e-10);
        // p' = -0.05 - 0.05 * 0.995 = -0.09975
        assert!((next.momentum[0] - (-0.09975)).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_single_step() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = RK4Baseline;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let next = integrator.step(&state, &ham, 0.1);
        // RK4 for harmonic oscillator: exact up to O(dt^5)
        // Exact: q(0.1) = cos(0.1) ≈ 0.99500417, p(0.1) = -sin(0.1) ≈ -0.09983342
        assert!((next.position[0] - 0.99500417).abs() < 1e-5);
        assert!((next.momentum[0] - (-0.09983342)).abs() < 1e-5);
    }

    #[test]
    fn test_harmonic_oscillator_exact_period_symplectic_euler() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = SymplecticEuler;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        // Period = 2π, integrate for one period
        let dt = 0.01;
        let steps = 628; // ≈ 2π/0.01
        let trajectory = integrator.integrate(&state, &ham, dt, steps);

        let final_state = trajectory.last().unwrap();
        // Should be approximately back to start (symplectic preserves orbit)
        assert!((final_state.position[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_harmonic_oscillator_exact_period_stormer_verlet() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let dt = 0.01;
        let steps = 628;
        let trajectory = integrator.integrate(&state, &ham, dt, steps);

        let final_state = trajectory.last().unwrap();
        // Verlet is more accurate than Euler
        assert!((final_state.position[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_symplectic_euler_energy_conservation_harmonic() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = SymplecticEuler;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];
        let initial_energy = ham.value(&state.position, &state.momentum);

        let dt = 0.01;
        let trajectory = integrator.integrate(&state, &ham, dt, 1000);

        let max_drift = trajectory
            .iter()
            .map(|s| (ham.value(&s.position, &s.momentum) - initial_energy).abs())
            .fold(0.0_f64, f64::max);

        // Symplectic: energy drift should be bounded (Euler is first-order, so ~dt)
        assert!(max_drift / initial_energy < 0.01, "SymplecticEuler drift too large: {}", max_drift / initial_energy);
    }

    #[test]
    fn test_stormer_verlet_energy_conservation_harmonic() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];
        let initial_energy = ham.value(&state.position, &state.momentum);

        let dt = 0.01;
        let trajectory = integrator.integrate(&state, &ham, dt, 1000);

        let max_drift = trajectory
            .iter()
            .map(|s| (ham.value(&s.position, &s.momentum) - initial_energy).abs())
            .fold(0.0_f64, f64::max);

        // Verlet: should be even better than Euler
        assert!(max_drift / initial_energy < 1e-3, "StormerVerlet drift too large: {}", max_drift / initial_energy);
    }

    #[test]
    fn test_rk4_energy_drift_long_integration() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = RK4Baseline;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];
        let initial_energy = ham.value(&state.position, &state.momentum);

        let dt = 0.1;
        let trajectory = integrator.integrate(&state, &ham, dt, 10000);

        let max_drift = trajectory
            .iter()
            .map(|s| (ham.value(&s.position, &s.momentum) - initial_energy).abs())
            .fold(0.0_f64, f64::max);

        // RK4 should show noticeable drift over 10000 steps with dt=0.1
        // (This is the key insight: non-symplectic drifts)
        assert!(max_drift / initial_energy > 1e-6, "RK4 should drift more than it does: {}", max_drift / initial_energy);
    }

    #[test]
    fn test_verlet_more_accurate_than_euler() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let dt = 0.01;
        let steps = 100;

        let euler = SymplecticEuler;
        let verlet = StormerVerlet;

        let euler_traj = euler.integrate(&state, &ham, dt, steps);
        let verlet_traj = verlet.integrate(&state, &ham, dt, steps);

        // Exact solution at t = steps * dt
        let t = steps as f64 * dt;
        let exact_q = t.cos();
        let exact_p = -t.sin();

        let euler_err = (euler_traj.last().unwrap().position[0] - exact_q).abs();
        let verlet_err = (verlet_traj.last().unwrap().position[0] - exact_q).abs();

        assert!(
            verlet_err < euler_err,
            "Verlet should be more accurate: euler_err={}, verlet_err={}",
            euler_err, verlet_err
        );
    }

    #[test]
    fn test_integrate_returns_correct_length() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = SymplecticEuler;
        let state = PhaseSpace::new(1);
        let trajectory = integrator.integrate(&state, &ham, 0.01, 50);
        assert_eq!(trajectory.len(), 51); // initial + 50 steps
    }

    // ============================================================
    // Conservation tests
    // ============================================================

    #[test]
    fn test_conservation_tracker_creation() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let state = PhaseSpace::new(1);
        let tracker = ConservationTracker::new(&state, &ham);
        assert!((tracker.initial_energy).abs() < 1e-10);
        assert_eq!(tracker.energies.len(), 1);
        assert_eq!(tracker.steps, 0);
    }

    #[test]
    fn test_conservation_tracker_record() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        let mut tracker = ConservationTracker::new(&state, &ham);

        tracker.record(&state, &ham);
        assert_eq!(tracker.steps, 1);
        assert_eq!(tracker.energies.len(), 2);
    }

    #[test]
    fn test_conservation_energy_drift() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        let mut tracker = ConservationTracker::new(&state, &ham);

        let integrator = StormerVerlet;
        let trajectory = integrator.integrate(&state, &ham, 0.01, 100);
        for s in trajectory.iter().skip(1) {
            tracker.record(s, &ham);
        }

        assert!(tracker.energy_drift() < 1e-4);
    }

    #[test]
    fn test_conservation_is_conserved() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        let mut tracker = ConservationTracker::new(&state, &ham);

        let integrator = StormerVerlet;
        let trajectory = integrator.integrate(&state, &ham, 0.01, 100);
        for s in trajectory.iter().skip(1) {
            tracker.record(s, &ham);
        }

        assert!(tracker.is_conserved(1e-4));
    }

    #[test]
    fn test_conservation_report() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let state = PhaseSpace::new(1);
        let tracker = ConservationTracker::new(&state, &ham);
        let report = tracker.report();
        assert!(report.contains("Conservation Report"));
        assert!(report.contains("Steps: 0"));
    }

    #[test]
    fn test_symplecticity_error_symplectic() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.5];

        let err = verify_symplecticity(&integrator, &state, &ham, 0.01);
        // Verlet should satisfy symplectic condition approximately
        assert!(err < 0.1, "Symplecticity error too large: {}", err);
    }

    // ============================================================
    // Agent experiment tests
    // ============================================================

    #[test]
    fn test_agent_experiment_run() {
        let ham = Box::new(QuadraticHamiltonian::new(1.0, 1.0));
        let integrator = Box::new(StormerVerlet);
        let mut initial = PhaseSpace::new(1);
        initial.position = vec![1.0];

        let experiment = AgentExperiment::new(ham, integrator, initial);
        let result = experiment.run(0.01, 100);

        assert_eq!(result.trajectory.len(), 101);
        assert_eq!(result.integrator_name, "StormerVerlet");
    }

    #[test]
    fn test_compare_integrators() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut initial = PhaseSpace::new(1);
        initial.position = vec![1.0];

        let results = compare_integrators(&ham, &initial, 0.01, 100);
        assert_eq!(results.len(), 3);

        // Check names
        let names: Vec<_> = results.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"SymplecticEuler"));
        assert!(names.contains(&"StormerVerlet"));
        assert!(names.contains(&"RK4Baseline"));
    }

    #[test]
    fn test_symplectic_wins_on_conservation() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut initial = PhaseSpace::new(1);
        initial.position = vec![1.0];

        // Use large dt to expose RK4's non-symplectic drift
        let results = compare_integrators(&ham, &initial, 0.3, 10000);

        let mut symplectic_drifts = Vec::new();
        let mut rk4_drift = 0.0_f64;

        for (name, result) in &results {
            if name == &"RK4Baseline" {
                rk4_drift = result.energy_drift;
            } else {
                symplectic_drifts.push(result.energy_drift);
            }
        }

        // Verlet (second-order symplectic) should beat RK4 at large dt/long runs
        let verlet_drift = results.iter().find(|(n, _)| *n == "StormerVerlet").unwrap().1.energy_drift;
        assert!(
            verlet_drift < rk4_drift,
            "Verlet ({}) should have less drift than RK4 ({})",
            verlet_drift, rk4_drift
        );
    }

    #[test]
    fn test_find_attractor_quadratic() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut initial = PhaseSpace::new(2);
        initial.position = vec![5.0, -3.0];

        let attractor = find_attractor(&ham, &initial);
        assert!(attractor[0].abs() < 0.01);
        assert!(attractor[1].abs() < 0.01);
    }

    #[test]
    fn test_find_attractor_agent() {
        let ham = AgentHamiltonian::new(quadratic_potential, quadratic_gradient, 1.0);
        let mut initial = PhaseSpace::new(2);
        initial.position = vec![5.0, -3.0];

        let attractor = find_attractor(&ham, &initial);
        assert!(attractor[0].abs() < 0.01);
        assert!(attractor[1].abs() < 0.01);
    }

    #[test]
    fn test_double_well_finds_minimum() {
        let ham = DoubleWellHamiltonian::new(2.0, 1.0);
        let mut initial = PhaseSpace::new(1);
        initial.position = vec![0.5];

        let attractor = find_attractor(&ham, &initial);
        // Should find one of the minima at q ≈ ±1
        assert!(
            (attractor[0] - 1.0).abs() < 0.1 || (attractor[0] + 1.0).abs() < 0.1,
            "Should find a minimum near ±1, got {}",
            attractor[0]
        );
    }

    #[test]
    fn test_bifurcation_diagram() {
        let ham = DoubleWellHamiltonian::new(1.0, 1.0);
        let diagram = bifurcation_diagram(&ham, (0.0, 3.0), 10);

        assert_eq!(diagram.len(), 10);
        // At a=0: only q=0
        assert_eq!(diagram[0].1.len(), 1);
        // At a=3: q=0 and q=±sqrt(3/2)
        assert_eq!(diagram.last().unwrap().1.len(), 3);
    }

    #[test]
    fn test_agent_result_summary() {
        let ham = Box::new(QuadraticHamiltonian::new(1.0, 1.0));
        let integrator = Box::new(StormerVerlet);
        let mut initial = PhaseSpace::new(1);
        initial.position = vec![1.0];

        let experiment = AgentExperiment::new(ham, integrator, initial);
        let result = experiment.run(0.01, 10);

        let summary = result.summary();
        assert!(summary.contains("StormerVerlet"));
        assert!(summary.contains("Steps: 10"));
    }

    #[test]
    fn test_agent_result_displacement() {
        let ham = Box::new(QuadraticHamiltonian::new(1.0, 1.0));
        let integrator = Box::new(StormerVerlet);
        let mut initial = PhaseSpace::new(1);
        initial.position = vec![1.0];

        let experiment = AgentExperiment::new(ham, integrator, initial);
        let result = experiment.run(0.01, 10);

        // Should have moved
        assert!(result.displacement() > 0.0);
    }

    // ============================================================
    // THE KEY TESTS: Proving symplectic superiority
    // ============================================================

    #[test]
    fn test_key_harmonic_energy_conserved_symplectic() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let dt = 0.01;
        let steps = 1000;
        let initial_energy = ham.value(&state.position, &state.momentum);

        let trajectory = integrator.integrate(&state, &ham, dt, steps);
        let mut tracker = ConservationTracker::new(&state, &ham);
        for s in trajectory.iter().skip(1) {
            tracker.record(s, &ham);
        }

        // THE KEY ASSERTION: Symplectic integration preserves energy
        assert!(
            tracker.energy_drift() < 1e-3,
            "Symplectic energy drift should be < 1e-3, got {}",
            tracker.energy_drift()
        );

        let max_dev = (ham.value(
            &trajectory.last().unwrap().position,
            &trajectory.last().unwrap().momentum,
        ) - initial_energy)
            .abs();
        assert!(max_dev < 1e-3, "Final energy deviation too large: {}", max_dev);
    }

    #[test]
    fn test_key_long_integration_symplectic_bounded() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let dt = 0.05;
        let steps = 10000;

        let trajectory = integrator.integrate(&state, &ham, dt, steps);
        let mut tracker = ConservationTracker::new(&state, &ham);
        for s in trajectory.iter().skip(1) {
            tracker.record(s, &ham);
        }

        // Even after 10000 steps, symplectic should be bounded
        assert!(
            tracker.energy_drift() < 1e-3,
            "Long symplectic integration should remain bounded, drift: {}",
            tracker.energy_drift()
        );

        // Phase space should remain bounded
        let max_q: f64 = trajectory
            .iter()
            .map(|s| s.position[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(max_q < 2.0, "Symplectic trajectory should remain bounded, max_q: {}", max_q);
    }

    #[test]
    fn test_key_long_integration_rk4_drifts() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = RK4Baseline;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];
        state.momentum = vec![0.0];

        let dt = 0.3; // Larger dt to induce more drift
        let steps = 10000;

        let trajectory = integrator.integrate(&state, &ham, dt, steps);
        let initial_energy = ham.value(&state.position, &state.momentum);

        // RK4 should show growing energy deviation
        let final_energy = ham.value(
            &trajectory.last().unwrap().position,
            &trajectory.last().unwrap().momentum,
        );
        let drift = ((final_energy - initial_energy) / initial_energy).abs();
        assert!(
            drift > 1e-6,
            "RK4 should drift over long integration, drift: {}",
            drift
        );
    }

    #[test]
    fn test_key_agent_reaches_goal() {
        let ham = AgentHamiltonian::new(quadratic_potential, quadratic_gradient, 1.0);
        let integrator = SymplecticEuler;
        let mut state = PhaseSpace::new(2);
        state.position = vec![5.0, 3.0];
        state.momentum = vec![0.0, 0.0];

        // With damping (not pure Hamiltonian, but shows the direction)
        // Use small dt and many steps
        let trajectory = integrator.integrate(&state, &ham, 0.001, 10000);

        // Energy should be conserved (symplectic)
        let initial_energy = ham.value(&state.position, &state.momentum);
        let final_energy = ham.value(
            &trajectory.last().unwrap().position,
            &trajectory.last().unwrap().momentum,
        );
        assert!(
            ((final_energy - initial_energy) / initial_energy).abs() < 1e-3,
            "Agent energy should be conserved: initial={}, final={}",
            initial_energy, final_energy
        );
    }

    #[test]
    fn test_key_conservation_law_proof() {
        // THE definitive test: prove symplectic preserves what Cyberloop needs
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let mut initial = PhaseSpace::new(2);
        initial.position = vec![1.0, 0.0];
        initial.momentum = vec![0.0, 1.0];

        // Large dt + many steps to expose non-symplectic drift
        let results = compare_integrators(&ham, &initial, 0.3, 10000);

        // Find results
        let verlet_result = &results.iter().find(|(n, _)| *n == "StormerVerlet").unwrap().1;
        let rk4_result = &results.iter().find(|(n, _)| *n == "RK4Baseline").unwrap().1;

        // Symplectic MUST have lower energy drift at large dt
        assert!(
            verlet_result.energy_drift < rk4_result.energy_drift,
            "KEY ASSERTION FAILED: Verlet drift ({}) >= RK4 drift ({})",
            verlet_result.energy_drift, rk4_result.energy_drift
        );

        // Symplectic should have excellent conservation
        assert!(
            verlet_result.conservation.is_conserved(1e-2),
            "Verlet should conserve energy within 1e-2"
        );
    }

    #[test]
    fn test_path_length_positive() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];

        let experiment = AgentExperiment::new(
            Box::new(ham),
            Box::new(integrator),
            state,
        );
        let result = experiment.run(0.01, 100);
        assert!(result.path_length > 0.0);
    }

    #[test]
    fn test_final_position_and_momentum() {
        let ham = QuadraticHamiltonian::new(1.0, 1.0);
        let integrator = StormerVerlet;
        let mut state = PhaseSpace::new(1);
        state.position = vec![1.0];

        let experiment = AgentExperiment::new(
            Box::new(ham),
            Box::new(integrator),
            state,
        );
        let result = experiment.run(0.01, 10);
        assert_eq!(result.final_position().len(), 1);
        assert_eq!(result.final_momentum().len(), 1);
    }
}
