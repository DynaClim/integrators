//! Specialised numerical integrator originally written by Siddharth Bhatnagar (in python).
//!
//!
use crate::integrator::{Error, Integrator, Stats, System};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct Rkdp45 {
    x_initial: f64,
    x_final: f64,
    y_initial: Vec<f64>,
    y_final: Vec<f64>,
    /// First timestep, do not change as far as possible: 1e3 - 8.76 hours time step, 1e4 - 0.876 hrs, 31623 - 16.22 m, 1e5 - 5.256 mins
    step_size: f64,
    /// Maximum number of loop iterations in the integrator.
    max_iterations: usize,
    /// Frequency of processing output from integration (calling solout function).
    /// (1 = all output, 2 is every second iteration, 10 is every tenth, etc)
    output_frequency: usize,
    /// Error tolerance for truncation error. Good convergence from 1e-14 tolerance onwards
    tolerance: f64,
    /// Parameters for adaptive stepsize
    fac_min: f64,
    fac_max: f64,
    step_size_error_factor: f64,
    stats: Stats,
}

impl Default for Rkdp45 {
    fn default() -> Self {
        Self {
            x_initial: 0.,
            x_final: 0.,
            y_initial: vec![],
            y_final: vec![],
            // Sid's magic number initial timestep: SECONDS_IN_YEAR / 31623
            step_size: 997.931_885_020_396_5,
            max_iterations: 0,
            output_frequency: 1,
            tolerance: 1.0e-14,
            fac_min: 0.5,
            fac_max: 2.0,
            step_size_error_factor: 0.9,
            stats: Stats::default(),
        }
    }
}

impl Integrator for Rkdp45 {
    fn initialise(&mut self, x_initial: f64, x_final: f64, y_initial: &[f64]) -> Result<(), Error> {
        self.initialise(x_initial, x_final, y_initial);
        self.reconfigure();
        Ok(())
    }

    fn integrate<S: System>(&mut self, system: &mut S) -> Result<Stats, Error> {
        self.integrate(system)
    }

    fn y_final(&self) -> Vec<f64> {
        self.y_final.clone()
    }
}

impl Rkdp45 {
    pub fn new() -> Self {
        Self::default()
    }

    fn initialise(&mut self, x_initial: f64, x_final: f64, y_initial: &[f64]) {
        self.x_initial = x_initial;
        self.x_final = x_final;
        self.y_initial = y_initial.to_vec();
        self.reconfigure();
    }

    fn reconfigure(&mut self) {
        // If the maximum iterations are not specified, set it dynamically based on the starting conditions.
        // This cannot be a default value as it relies on other fields (x_final, x_initial, step_size).
        if self.max_iterations == 0 {
            self.max_iterations = ((self.x_final - self.x_initial) / self.step_size) as usize;
        }
    }

    fn integrate<S>(&mut self, system: &mut S) -> Result<Stats, Error>
    where
        S: System,
    {
        // Dormand-Prince method with adaptive stepsize control

        // Intermediate internal representations. Useful for resuming snapshots.
        let mut x = self.x_initial;
        let mut y = self.y_initial.clone();

        let system_size = self.y_initial.len();
        // Intermediate values.
        let mut buf = vec![0.0; system_size];
        let mut k1 = vec![0.0; system_size];
        let mut k2 = vec![0.0; system_size];
        let mut k3 = vec![0.0; system_size];
        let mut k4 = vec![0.0; system_size];
        let mut k5 = vec![0.0; system_size];
        let mut k6 = vec![0.0; system_size];
        let mut k7 = vec![0.0; system_size];
        let mut u4 = vec![0.0; system_size];
        let mut u5 = vec![0.0; system_size];
        let mut u_err: f64;
        let mut step_size_new: f64;

        while self.stats.accepted_steps < self.max_iterations {
            system.derive(x, &y, &mut k1)?;

            for (i, b) in buf.iter_mut().enumerate() {
                *b = y[i] + self.step_size * (1. / 5. * k1[i]);
            }
            system.derive(x, &buf, &mut k2)?;

            for (i, b) in buf.iter_mut().enumerate() {
                *b = y[i] + self.step_size * (3. / 40. * k1[i] + 9. / 40. * k2[i]);
            }
            system.derive(x, &buf, &mut k3)?;

            for (i, b) in buf.iter_mut().enumerate() {
                *b = y[i]
                    + self.step_size * (44. / 45. * k1[i] - 56. / 15. * k2[i] + 32. / 9. * k3[i]);
            }
            system.derive(x, &buf, &mut k4)?;

            for (i, b) in buf.iter_mut().enumerate() {
                *b = y[i]
                    + self.step_size
                        * (19372. / 6561. * k1[i] - 25360. / 2187. * k2[i]
                            + 64448. / 6561. * k3[i]
                            - 212. / 729. * k4[i]);
            }
            system.derive(x, &buf, &mut k5)?;

            for (i, b) in buf.iter_mut().enumerate() {
                *b = y[i]
                    + self.step_size
                        * (9017. / 3168. * k1[i] - 355. / 33. * k2[i]
                            + 46732. / 5247. * k3[i]
                            + 49. / 176. * k4[i]
                            - 5103. / 18656. * k5[i]);
            }
            system.derive(x, &buf, &mut k6)?;

            for (i, u) in u5.iter_mut().enumerate() {
                *u = y[i]
                    + self.step_size
                        * (35. / 384. * k1[i] + 500. / 1113. * k3[i] + 125. / 192. * k4[i]
                            - 2187. / 6784. * k5[i]
                            + 11. / 84. * k6[i]);
            }
            system.derive(x, &u5, &mut k7)?;

            self.stats.function_calls += 6;

            for (i, u) in u4.iter_mut().enumerate() {
                *u = y[i]
                    + self.step_size
                        * (5179. / 57600. * k1[i] + 7571. / 16695. * k3[i] + 393. / 640. * k4[i]
                            - 92097. / 339_200. * k5[i]
                            + 187. / 2100. * k6[i]
                            + 1. / 40. * k7[i]);
            }
            // Truncation error
            // From https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            // np.linalg.norm(a) == max(abs(a))
            u_err = u5
                .iter()
                .zip(u4.iter())
                .map(|(a, b)| abs!(a - b))
                .reduce(f64::max)
                .unwrap_or(0.0_f64);

            if u_err == 0.0 {
                step_size_new = self.fac_max * self.step_size;
            } else {
                step_size_new = self.step_size
                    * min!(
                        self.fac_max,
                        max!(
                            self.fac_min,
                            self.step_size_error_factor * (self.tolerance / u_err).powf(1. / 5.)
                        )
                    );
            }
            // If truncation error bigger than tolerance, try again.
            if u_err > self.tolerance {
                self.stats.rejected_steps += 1;
                self.step_size = step_size_new;
                continue;
            }

            // Final stepsize cannot overshoot x_final.
            self.step_size = min!(self.x_final - x, step_size_new);

            // Process the intermediate integration output at the provided rate
            if self.stats.accepted_steps % self.output_frequency == 0 {
                system.solout(x, &y)?;
                // TODO add snapshot check (set x_initial = x and y_initial = y for snapshots)
            }

            // Update the variables for next iteration.
            x += self.step_size;
            y.clone_from(&u5);

            // End the integration if x_final is reached.
            if self.x_final - x <= 1e-30 {
                // Process the final integration output.
                system.solout(x, &y)?;
                self.y_final.clone_from(&y);
                return Ok(self.stats);
            }
            self.stats.accepted_steps += 1;
        }
        // Integration incomplete: The maximum number of iterations was reached before x_final was reached.
        self.y_final.clone_from(&y);

        Err(Error::StepLimitReached {
            x,
            n_step: self.stats.accepted_steps,
        })
    }
}
