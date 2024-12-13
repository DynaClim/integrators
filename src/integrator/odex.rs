use crate::integrator::{Error, Integrator, Stats, System};
use itertools::izip;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

// Numerical solution of a system of first order ordinary diffferential equations y' = f(x, y).
// This is an extrapolation-algorithm (gbs), based on the explicit midpoint rule
// (with stepsize control, order selection and dense output).

// E. Hairer, S.P. Norsett & G. Wanner
// Solving Ordinary Diffferential Equations I. Nonstiff Problems. 2nd Edition.
// Springer series in computational mathematics, Springer-Verlag (1993)
// Section II.9

// Deviations from fortran reference implementation:
//  - Contex function not implemented.
//  - Dense components untested

// Errors of y[i] are kept below tolerance.relative[i] * abs!(y[i]) + tolerance.absolute[i]
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Tolerance {
    pub absolute: Vec<f64>,
    pub relative: Vec<f64>,
}

#[derive(Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum StepState {
    Basic,
    Other,
}

impl Tolerance {
    fn reconfigure(&mut self, system_size: usize) -> Result<(), Error> {
        Self::extend_or_error(&mut self.absolute, system_size)?;
        Self::extend_or_error(&mut self.relative, system_size)?;

        Ok(())
    }

    // If a single tolerance value is provided, extend the vector to apply it to all elements.
    fn extend_or_error(tolerance: &mut Vec<f64>, system_size: usize) -> Result<(), Error> {
        match tolerance.len().cmp(&system_size) {
            Ordering::Less => {
                if tolerance.len() == 1 {
                    *tolerance = vec![tolerance[0]; system_size];
                    return Ok(());
                }
            }
            Ordering::Equal => return Ok(()),
            Ordering::Greater => (),
        };

        Err(Error::InvalidParameter {
            param: format!("tolerance: {tolerance:?}"),
        })
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Default)]
// Specifies under which condition to write (output) the state of the integration (by calling `system.solout()`).
pub enum SolutionOutput {
    #[default]
    Dense,
    Regular,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct Odex {
    x_initial: f64,
    x_final: f64,
    y_initial: Vec<f64>,
    y_final: Vec<f64>,
    // WORK
    // Maximum step size. Default is x_final - x.
    max_step_size: f64, // HMAX
    // Step size for the j-th diagonal entry:
    // fac_min / self.step_size_selection_b <= HNEW(j) / HOLD <= 1 / fac_min
    // where: fac_min = self.step_size_selection_a ** (1 / (2 * j-1))
    step_size_selection_a: f64, // FAC1
    step_size_selection_b: f64, // FAC2
    // step size is decreased if self.work_per_unit_step[K - 1] <= self.work_per_unit_step[K] * self.step_order_selection_a
    // step size is increased if self.work_per_unit_step[K]    <= self.work_per_unit_step[K - 1 * self.step_order_selection_b
    step_order_selection_a: f64, // FAC3
    step_order_selection_b: f64, // FAC4
    // self.step_control_safety_b = 0.94  if  "HOPE FOR CONVERGENCE"
    // self.step_control_safety_b = 0.90  if  "NO HOPE FOR CONVERGENCE"
    step_control_safety_a: f64,      // SAFE1
    step_control_safety_b: f64,      // SAFE2
    step_size_reduction_factor: f64, // SAFE3

    // IWORK
    max_integration_steps: usize, // NMAX
    // Should be 0 or >= 3 according to original documentation.
    // Maximum number of columns in the extrapolation
    max_extrapolation_columns: usize, // KM
    // Selector for the step size sequence (even numbers):
    // 1 => { 2, 4,  6,  8, 10, 12, 14, 16, ...}
    // 2 => { 2, 4,  8, 12, 16, 20, 24, 28, ...}
    // 3 => { 2, 4,  6,  8, 12, 16, 24, 32, ...}
    // 4 => { 2, 6, 10, 14, 18, 22, 26, 30, ...}
    // 5 => { 4, 8, 12, 16, 20, 24, 28, 32, ...}
    step_size_sequence: usize, // NSEQU
    // Stability check is activated at most self.stability_check_freq times one line of the extapolation table.
    stability_check_freq: usize, // MSTAB
    // Stability check is activated only in the "lines" up to self.stability_check_lines of the extapolation table.
    stability_check_lines: usize, // JSTAB
    // Toggle error estimation in the dense output formula
    dense_output_error_estimator: bool, // IDERR
    // Determines the degree of interpolation formula. Should be in range 1 < self.interpolation_formula_degree < 6
    // MU = 2  *  KAPPA - self.interpolation_formula_degree  +  1
    interpolation_formula_degree: usize, // MUDIF
    // Number of components for which dense output is required.
    dense_component_count: usize, // NRDENS or NRD

    a: [f64; 30],
    t: Vec<Vec<f64>>,
    ysafe: Vec<Vec<f64>>,
    fsafe: Vec<Vec<f64>>,

    num_of_func_evals: Vec<usize>,
    step_num_seq_nj: Vec<usize>,

    work_per_unit_step: Vec<f64>,
    step_size_at_interpolation_level: Vec<f64>,
    scal: Vec<f64>,
    dz: Vec<f64>,
    dy: Vec<f64>, // Buffer passed to derivation function to calculate derivative of y.
    yh1: Vec<f64>,
    yh2: Vec<f64>,

    dense_output: bool,
    // Vectors of positional tolerance.
    tolerance: Tolerance,

    step_size: f64,
    err: f64,
    errold: f64,
    fac: f64,
    // Number of rejected steps due to error test, excluding the first step.
    stats: Stats, // NFCN, NACCPT, NREJCT, NSTEP
    ipt: usize,
    // Represents the direction of the integration (1.0 forward, -1.0 backward).
    sign: f64,
    // Number of quantities to integrate (i.e. the y.len())
    system_size: usize,

    k: usize,
    kappa: usize,
    state: StepState,
}

impl Default for Odex {
    fn default() -> Self {
        // Defaults from the original fortran77 odex implementation.
        let max_extrapolation_columns = 9;
        let fsafe_rows =
            2 * max_extrapolation_columns * max_extrapolation_columns + max_extrapolation_columns;
        Self {
            x_initial: 0.0,
            x_final: 0.0,
            y_initial: vec![],
            y_final: vec![],

            step_size: 10.0,
            max_step_size: 0.0,
            step_size_reduction_factor: 0.5,
            step_size_selection_a: 0.02,
            step_size_selection_b: 4.0,
            step_order_selection_a: 0.8,
            step_order_selection_b: 0.9,
            step_control_safety_a: 0.65,
            step_control_safety_b: 0.94,
            max_integration_steps: 10000,
            max_extrapolation_columns,
            step_size_sequence: 0,
            stability_check_freq: 1,
            stability_check_lines: 2,
            dense_output_error_estimator: true,
            interpolation_formula_degree: 4,
            dense_component_count: 0,
            dense_output: true,
            tolerance: Tolerance {
                absolute: vec![1E-9],
                relative: vec![1E-9],
            },

            a: [0.0; 30],
            t: vec![vec![0.0; max_extrapolation_columns]],
            ysafe: vec![vec![0.0; max_extrapolation_columns]],
            fsafe: vec![vec![0.0; fsafe_rows]],

            num_of_func_evals: vec![0; max_extrapolation_columns],
            work_per_unit_step: vec![0.0; max_extrapolation_columns],
            step_size_at_interpolation_level: vec![0.0; max_extrapolation_columns],

            scal: vec![0.0],
            dz: vec![0.0],
            dy: vec![0.0],
            yh1: vec![0.0],
            yh2: vec![0.0],

            fac: 0.,
            err: 0.,
            errold: 1.0e10,

            step_num_seq_nj: vec![],
            stats: Stats::default(),
            ipt: 0,
            sign: 1.0,
            system_size: 0,
            k: 0,
            kappa: 0,
            state: StepState::Basic,
        }
    }
}

impl Integrator for Odex {
    fn initialise(&mut self, x_initial: f64, x_final: f64, y_initial: &[f64]) -> Result<(), Error> {
        self.initialise(x_initial, x_final, y_initial);
        self.reconfigure()?;
        Ok(())
    }

    fn integrate<S: System>(&mut self, system: &mut S) -> Result<Stats, Error> {
        self.integrate(system)
    }

    fn y_final(&self) -> Vec<f64> {
        self.y_final.clone()
    }
}

impl Odex {
    fn initialise(&mut self, x_initial: f64, x_final: f64, y_initial: &[f64]) {
        self.system_size = y_initial.len();
        self.x_initial = x_initial;
        self.x_final = x_final;
        // Set the direction of the integration (positive is forwards, negative is backwards).
        self.sign = f64::signum(x_final - x_initial);
        self.y_initial = y_initial.to_vec();
        self.t = vec![vec![0.0; self.max_extrapolation_columns]; self.system_size];
        self.ysafe = vec![vec![0.0; self.max_extrapolation_columns]; self.system_size];
        let fsafe_rows = 2 * self.max_extrapolation_columns * self.max_extrapolation_columns
            + self.max_extrapolation_columns;
        self.fsafe = vec![vec![0.0; fsafe_rows]; self.system_size];
        self.scal = vec![0.0; self.system_size];
        self.dz = vec![0.0; self.system_size];
        self.dy = vec![0.0; self.system_size];
        self.yh1 = vec![0.0; self.system_size];
        self.yh2 = vec![0.0; self.system_size];
    }

    fn reconfigure(&mut self) -> Result<(), Error> {
        if self.max_extrapolation_columns < 3 {
            return Err(Error::InvalidParameter {
                param: format!(
                    "max_extrapolation_columns: {}",
                    self.max_extrapolation_columns
                ),
            });
        };

        if self.interpolation_formula_degree > 6 {
            return Err(Error::InvalidParameter {
                param: format!(
                    "interpolation_formula_degree: {}",
                    self.interpolation_formula_degree
                ),
            });
        };

        if self.dense_component_count > self.y_initial.len() {
            return Err(Error::InvalidParameter {
                param: format!("dense_component_count: {}", self.dense_component_count),
            });
        };

        if (self.step_size_reduction_factor <= f64::EPSILON)
            || (self.step_size_reduction_factor >= 1.0)
        {
            return Err(Error::InvalidParameter {
                param: format!(
                    "step_size_reduction_factor: {}",
                    self.step_size_reduction_factor
                ),
            });
        };

        if ((self.step_size_sequence == 2) || (self.step_size_sequence == 3)) && self.dense_output {
            return Err(Error::InvalidParameter {
                param: format!("step_size_sequence: {}", self.step_size_sequence),
            });
        };

        // Dense error estimation without dense output is not possible.
        if !self.dense_output && self.dense_output_error_estimator {
            return Err(Error::InvalidParameter {
                param: format!("dense_output_error_estimator: {}", self.step_size_sequence),
            });
        };

        self.tolerance.reconfigure(self.system_size)?;

        // Set the maximum step size to be the "duration" of the entire integration.
        if self.max_step_size == 0.0 {
            self.max_step_size = self.x_final - self.x_initial;
        };

        if self.step_size_sequence == 0 {
            if self.dense_output {
                self.step_size_sequence = 4;
            } else {
                self.step_size_sequence = 1;
            }
        }

        self.step_num_seq_nj = match self.step_size_sequence {
            // 2, 4, 6, 8, 10, 12, 14, 16, ... (eq 9.8')
            1 => (1..=self.max_extrapolation_columns)
                .map(|x| x * 2)
                .collect(),
            // 2, 4, 8, 12, 16, 20, 24, 28, ...
            2 => (1..=self.max_extrapolation_columns)
                .map(|x| x * 4 - 4)
                .collect(),
            // 2, 4, 6, 8, 12, 16, 24, 32, 48, ... (eq 9.7')
            3 => (1..=self.max_extrapolation_columns)
                .flat_map(|x| [2_usize.pow(x as u32 + 1), 2_usize.pow(x as u32) * 6])
                .collect(),
            // 2, 6, 10, 14, 18, 22, 26, 30, ... (eq 9.35)
            4 => (1..=self.max_extrapolation_columns)
                .map(|x| x * 4 - 2)
                .collect(),
            // 4, 8, 12, 16, 20, 24, 28, 32, ...
            5 => (1..=self.max_extrapolation_columns)
                .map(|x| x * 4)
                .collect(),
            _ => {
                return Err(Error::InvalidParameter {
                    param: format!(
                        "step_size_reduction_factor: {}",
                        self.step_size_reduction_factor
                    ),
                });
            }
        };

        Ok(())
    }

    fn update_ysafe(&mut self, start: usize, end: usize) -> f64 {
        let mut factor = 0.0;
        for l in (start..=end).rev() {
            factor = (self.step_num_seq_nj[end] as f64 / self.step_num_seq_nj[l - 1] as f64)
                .powi(2)
                - 1.0;
            for ysafe_val in self.ysafe.iter_mut().take(self.dense_component_count) {
                ysafe_val[l - 1] = ysafe_val[l] + (ysafe_val[l] - ysafe_val[l - 1]) / factor;
            }
        }

        factor
    }

    // Computes the j-th line of the extapolation table and estimates the optimal stepsize.
    fn midex<S>(&mut self, j: usize, x: f64, y: &[f64], system: &mut S) -> Result<bool, Error>
    where
        S: System,
    {
        let dont_reject_step = false;
        let reject_step = true;
        let hj = self.step_size / self.step_num_seq_nj[j] as f64;
        // Euler starting step
        self.yh1[..].copy_from_slice(y);
        self.yh2 = izip!(y, &self.dz).map(|(a, b)| a + hj * b).collect();
        // Explicit midpoint rule
        let m = self.step_num_seq_nj[j] - 1;
        let nj_mid = self.step_num_seq_nj[j] / 2;

        for k in 0..m {
            if self.dense_output && (k + 1 == nj_mid) {
                for y in self.ysafe.iter_mut().take(self.dense_component_count) {
                    y[j] = self.yh2[0];
                }
            }

            system.derive(x + hj * (k + 1) as f64, &self.yh2, &mut self.dy)?;

            if self.dense_output && (nj_mid.abs_diff(k + 1) <= 2 * j + 1) {
                for f in self.fsafe.iter_mut().take(self.dense_component_count) {
                    f[self.ipt] = self.dy[0];
                }
                self.ipt += 1;
            }

            let tmp = self.yh2.clone();
            self.yh2 = izip!(&self.yh1, &self.dy)
                .map(|(a, b)| a + 2.0 * hj * b)
                .collect();
            self.yh1 = tmp;

            if (k < self.stability_check_freq) && (j < self.stability_check_lines) {
                // Stability check
                let del1 = izip!(&self.dz, &self.scal)
                    .map(|(a, b)| (*a / *b).powi(2))
                    .sum();

                let del2: f64 = izip!(&self.dz, &self.scal, &self.dy)
                    .map(|(a, b, c)| ((*c - *a) / *b).powi(2))
                    .sum();

                let quot = del2 / max!(f64::EPSILON, del1);

                if quot > 4.0 {
                    self.stats.function_calls += 1;
                    self.step_size *= self.step_size_reduction_factor;
                    return Ok(reject_step);
                }
            }
        }

        // Final smoothing step
        system.derive(x + self.step_size, &self.yh2, &mut self.dy)?;

        if self.dense_output && (nj_mid <= (2 * j + 1)) {
            for f in self.fsafe.iter_mut().take(self.dense_component_count) {
                f[self.ipt] = self.dy[0];
            }
            self.ipt += 1;
        }

        for (i, (a, b, c)) in izip!(&self.yh1, &self.yh2, &self.dy).enumerate() {
            self.t[i][j] = (a + b + hj * c) / 2.0;
        }

        self.stats.function_calls += self.step_num_seq_nj[j];
        // Polynomial extapolation
        if j == 0 {
            return Ok(dont_reject_step);
        }

        //    self.fac = update_ysafe(T, self.step_num_seq_nj, 1, j, N);
        for l in (1..=j).rev() {
            self.fac =
                (self.step_num_seq_nj[j] as f64 / self.step_num_seq_nj[l - 1] as f64).powi(2) - 1.0;
            for x in &mut self.t {
                x[l - 1] = x[l] + (x[l] - x[l - 1]) / self.fac;
            }
        }

        // Scaling
        for ((i, scal), atol, rtol, y_val) in izip!(
            self.scal.iter_mut().enumerate().take(self.system_size),
            &self.tolerance.absolute,
            &self.tolerance.relative,
            y
        ) {
            *scal = atol + rtol * max!(abs!(y_val), abs!(self.t[i][0]));
        }

        self.err = (0..self.system_size)
            .map(|i| ((self.t[i][0] - self.t[i][1]) / self.scal[i]).powi(2))
            .sum();
        self.err = sqrt!(self.err / self.system_size as f64);

        if (self.err * f64::EPSILON < 1.) && ((j <= 1) || (self.err < self.errold)) {
            self.errold = max!(4.0 * self.err, 1.0);
            // Compute optimal step sizes
            let exponent = 1.0 / (2 * j + 1) as f64;
            let fac_min = self.step_size_selection_a.powf(exponent);
            self.fac = min!(
                self.step_size_selection_b / fac_min,
                max!(
                    fac_min,
                    (self.err / self.step_control_safety_a).powf(exponent)
                        / self.step_control_safety_b
                )
            );
            self.fac = 1.0 / self.fac;
            self.step_size_at_interpolation_level[j] =
                min!(abs!(self.step_size) * self.fac, self.max_step_size);
            // (eq 9.26)
            self.work_per_unit_step[j] =
                self.num_of_func_evals[j] as f64 / self.step_size_at_interpolation_level[j];
            return Ok(dont_reject_step);
        }

        self.step_size *= self.step_size_reduction_factor;

        Ok(reject_step)
    }

    // TODO untested function.
    fn interpolate(&mut self, y: &mut [f64], imit: usize) {
        // Computes the coefficients of the interpolation formula
        // Begin with hermite interpolation
        for i in 0..self.dense_component_count {
            let y0 = y[i];
            let y1 = y[2 * self.dense_component_count + i];
            let yp0 = y[self.dense_component_count + i];
            let yp1 = y[3 * self.dense_component_count + i];
            let ydiff = y1 - y0;
            let aspl = -yp1 + ydiff;
            let bspl = yp0 - ydiff;

            y[self.dense_component_count + i] = ydiff;
            y[2 * self.dense_component_count + i] = aspl;
            y[3 * self.dense_component_count + i] = bspl;

            // Compute the derivatives of hermite at midpoint
            let ph0 = (y0 + y1) * 0.5 + 0.125 * (aspl + bspl);
            let ph1 = ydiff + (aspl - bspl) * 0.25;
            let ph2 = -(yp0 - yp1);
            let ph3 = 6.0 * (bspl - aspl);

            // Compute the further coefficients
            self.a.fill(0.0);
            self.a[0] = (y[4 * self.dense_component_count + i] - ph0) * 16.0;

            if imit >= 1 {
                self.a[1] = 16.0 * (y[5 * self.dense_component_count + i] - ph1);
            }
            if imit >= 2 {
                self.a[2] = (y[self.dense_component_count * 6 + i] - ph2 + self.a[0]) * 16.0;
            }
            if imit >= 3 {
                self.a[3] = 16.0 * (y[7 * self.dense_component_count + i] - ph3 + 3.0 * self.a[1]);
            }
            if imit >= 4 {
                for j in 4..=imit {
                    self.step_size_selection_a = (j * (j - 1)) as f64 / 2.0;
                    self.step_size_selection_b = (j * (j - 1) * (j - 2) * (j - 3)) as f64;
                    self.a[j] = 16.0
                        * (y[(j + 4) * self.dense_component_count + i]
                            + self.step_size_selection_a * self.a[j - 2]
                            - self.step_size_selection_b * self.a[j - 4]);
                }
            }

            for j in 0..=imit {
                y[self.dense_component_count * (j + 4) + i] = self.a[j];
            }
        }
    }

    fn step_rejected(&mut self) {
        self.k = min!(self.k, self.kappa, self.max_extrapolation_columns - 1);

        // (eq 9.27 variant)
        if (self.k > 2)
            && (self.work_per_unit_step[self.k - 1]
                < self.work_per_unit_step[self.k] * self.step_order_selection_a)
        {
            self.k -= 1;
        }
        self.stats.rejected_steps += 1;
        self.step_size = self.sign * self.step_size_at_interpolation_level[self.k - 1];
    }

    fn integrate<S>(&mut self, system: &mut S) -> Result<Stats, Error>
    where
        S: System,
    {
        let mut x = self.x_initial;
        let mut y = self.y_initial.clone();
        let mut x_prev;

        // Setup for dense components. Only used if dense_output enabled.
        let dense_component_length = max!(
            1,
            (2 * self.max_extrapolation_columns + 5) * self.dense_component_count
        );
        let mut dense_components = vec![0.0; dense_component_length];
        let mut error_factors = vec![0.0; 2 * self.max_extrapolation_columns];
        let mut ipoint = vec![0; self.max_extrapolation_columns + 1];
        if self.dense_output {
            for i in 0..self.max_extrapolation_columns {
                let njadd = 4 * i + 2;
                ipoint[i + 1] = ipoint[i] + njadd;
                if self.step_num_seq_nj[i] > njadd {
                    ipoint[i + 1] += 1;
                }
            }

            for mu in 1..=2 * self.max_extrapolation_columns {
                let mufloat = mu as f64;
                let errx = sqrt!(mufloat / (mufloat + 4.0)) * 0.5;
                let mut prod = 1. / (mufloat + 4.0).powi(2);
                for j in 1..=mu {
                    prod *= errx / (j as f64);
                }
                error_factors[mu - 1] = prod;
            }
        }

        // Define num_of_func_evals for order selection
        self.num_of_func_evals[0] = 1 + self.step_num_seq_nj[0];
        for i in 1..self.max_extrapolation_columns {
            self.num_of_func_evals[i] = self.num_of_func_evals[i - 1] + self.step_num_seq_nj[i];
        }

        for (scal, atol, rtol, y_val) in izip!(
            &mut self.scal,
            &self.tolerance.absolute,
            &self.tolerance.relative,
            &y
        ) {
            *scal = atol + rtol * abs!(y_val);
        }

        // Initial preparations
        self.k = min!(
            self.max_extrapolation_columns - 1,
            f64::floor(-log10!(self.tolerance.relative[0]) * 0.6 + 1.5) as usize
        );
        self.k = max!(2, self.k);

        self.max_step_size = abs!(self.max_step_size);
        self.step_size = max!(abs!(self.step_size), 1.0e-4);
        self.step_size = self.sign
            * min!(
                self.step_size,
                self.max_step_size,
                abs!(self.x_final - x) / 2.0
            );

        // Output starting values to begin integration.
        system.solout(x, &y)?;

        let mut hoptde = self.sign * self.max_step_size;
        self.work_per_unit_step[0] = 0.;
        let mut rejected = false;
        let mut first_step = true;
        let mut final_step = false;
        // Is x_final reached in the next step?
        'integration_loop: while 0.1 * abs!(self.x_final - x) > abs!(x) * f64::EPSILON {
            self.state = StepState::Basic;
            self.step_size = self.sign
                * min!(
                    abs!(self.step_size),
                    abs!(self.x_final - x),
                    self.max_step_size,
                    abs!(hoptde)
                );
            // Check if this is expected to be the final step.
            if (x + 1.01 * self.step_size - self.x_final) * self.sign > 0.0 {
                self.step_size = self.x_final - x;
                final_step = true;
            }

            if first_step || !self.dense_output {
                system.derive(x, &y, &mut self.dz)?;
            }
            self.stats.function_calls += 1;

            if first_step || final_step {
                first_step = false;
                self.stats.complete_steps += 1;
                self.ipt = 0;
                for j in 0..self.k {
                    self.kappa = j + 1;
                    if self.midex(j, x, &y, system)? {
                        rejected = true;
                        continue 'integration_loop;
                    }

                    // Is the step accepted?
                    if (j > 0) && (self.err <= 1.) {
                        self.state = StepState::Other;
                        break;
                    }
                }
            }
            match self.state {
                StepState::Other => {}
                StepState::Basic => {
                    // Basic integration step
                    'basic_step: loop {
                        self.ipt = 0;
                        self.stats.complete_steps += 1;

                        if self.stats.complete_steps >= self.max_integration_steps {
                            self.y_final.clone_from(&y);
                            // Integration incomplete: The maximum number of iterations was reached before x_final was reached.
                            return Err(Error::StepLimitReached {
                                x,
                                n_step: self.stats.complete_steps,
                            });
                        }

                        self.kappa = self.k - 1;
                        for j in 0..self.kappa {
                            if self.midex(j, x, &y, system)? {
                                rejected = true;
                                continue 'integration_loop;
                            }
                        }

                        // Convergence monitor
                        if !((self.k == 2) || rejected) {
                            if self.err <= 1.0 {
                                break 'basic_step;
                            }
                            // (eq 9.29)
                            if self.err
                                > ((self.step_num_seq_nj[self.k] * self.step_num_seq_nj[self.k - 1])
                                    as f64
                                    / self.step_num_seq_nj[0].pow(2) as f64)
                                    .powi(2)
                            {
                                self.step_rejected();
                                rejected = true;
                                continue 'basic_step;
                            }
                        }

                        if self.midex(self.k - 1, x, &y, system)? {
                            rejected = true;
                            continue 'integration_loop;
                        }

                        self.kappa = self.k;

                        if self.err <= 1. {
                            break 'basic_step;
                        }

                        // Hope for convergence in line self.k + 1
                        // (eq 9.31)
                        if self.err
                            > (self.step_num_seq_nj[self.k] as f64 / self.step_num_seq_nj[0] as f64)
                                .powi(2)
                        {
                            self.step_rejected();
                            rejected = true;
                            continue 'basic_step;
                        }

                        self.kappa = self.k + 1;
                        if self.midex(self.kappa - 1, x, &y, system)? {
                            rejected = true;
                            continue 'integration_loop;
                        }

                        // (eq 9.32)
                        if self.err > 1.0 {
                            self.step_rejected();
                            rejected = true;
                            continue 'basic_step;
                        }

                        break 'basic_step;
                    } // integration_step end
                } //arm
            }; // match

            // Step is accepted
            x_prev = x;
            x += self.step_size;

            if self.dense_output {
                let kmit = self.dense_component_calculation(
                    x,
                    &y,
                    &ipoint,
                    &mut dense_components,
                    system,
                )?;

                self.interpolate(&mut dense_components, kmit);

                // Estimation of the interpolation error
                if self.dense_output_error_estimator && (kmit >= 1) {
                    let error_estimate =
                        self.compute_dense_error_estimate(&dense_components, &error_factors, kmit);
                    hoptde = self.step_size
                        / max!((error_estimate).powf(1.0 / (kmit + 4) as f64), 0.01_f64);

                    if error_estimate > 10. {
                        self.step_size = hoptde;
                        x = x_prev;
                        self.stats.rejected_steps += 1;
                        rejected = true;
                        continue 'integration_loop;
                    }
                }
            }

            self.dz[..].copy_from_slice(&self.yh2);

            // y[i] set to t[i][0]
            for (i, y_val) in y.iter_mut().enumerate() {
                *y_val = self.t[i][0];
            }

            // Output intermediate integration.
            self.stats.accepted_steps += 1;
            system.solout(x, &y)?;

            // Compute optimal order
            let k_new = self.compute_optimal_order(rejected);
            self.compute_new_stepsize(rejected, k_new);
            rejected = false;
        } // integration_loop

        self.y_final.clone_from(&y);
        Ok(self.stats)
    }

    fn compute_dense_error_estimate(
        &mut self,
        dense_components: &[f64],
        error_factors: &[f64],
        kmit: usize,
    ) -> f64 {
        let mut error_estimate = 0.;
        for i in 0..self.dense_component_count {
            // TODO scal indices were indexing uninitialised values of the array, dense_components(i), which returned 1 (first index in fortran).
            error_estimate += (dense_components[(kmit + 4) * self.dense_component_count + i]
                / self.scal[0])
                .powi(2);
        }
        error_estimate =
            sqrt!(error_estimate / self.dense_component_count as f64) * error_factors[kmit];
        error_estimate
    }

    fn compute_optimal_order(&mut self, rejected: bool) -> usize {
        let mut k_new;
        // (eq 9.30)
        if self.kappa == 2 {
            k_new = if rejected {
                2
            } else {
                min!(3, self.max_extrapolation_columns - 1)
            };
        } else if self.kappa <= self.k {
            k_new = self.kappa;
            if self.work_per_unit_step[self.kappa - 2]
                < self.work_per_unit_step[self.kappa - 1] * self.step_order_selection_a
            {
                k_new = self.kappa - 1;
            }
            if self.work_per_unit_step[self.kappa - 1]
                < self.work_per_unit_step[self.kappa - 2] * self.step_order_selection_b
            {
                k_new = min!(self.kappa + 1, self.max_extrapolation_columns - 1);
            }
        } else {
            k_new = self.kappa - 1;
            if (self.kappa > 3)
                && (self.work_per_unit_step[self.kappa - 3]
                    < self.work_per_unit_step[self.kappa - 2] * self.step_order_selection_a)
            {
                k_new = self.kappa - 2;
            }
            if self.work_per_unit_step[self.kappa - 1]
                < self.work_per_unit_step[k_new - 1] * self.step_order_selection_b
            {
                k_new = min!(self.kappa, self.max_extrapolation_columns - 1);
            }
        }

        k_new
    }

    // Compute stepsize for next step
    fn compute_new_stepsize(&mut self, rejected: bool, k_new: usize) {
        // After a rejected step
        if rejected {
            self.k = min!(k_new, self.kappa);
            self.step_size = self.sign
                * min!(
                    abs!(self.step_size),
                    abs!(self.step_size_at_interpolation_level[self.k - 1])
                );
        } else {
            if k_new <= self.kappa {
                self.step_size = self.step_size_at_interpolation_level[k_new - 1];
            } else if (self.kappa < self.k)
                && (self.work_per_unit_step[self.kappa - 1]
                    < self.work_per_unit_step[self.kappa - 2] * self.step_order_selection_b)
            {
                self.step_size = self.step_size_at_interpolation_level[self.kappa - 1]
                    * self.num_of_func_evals[k_new] as f64
                    / self.num_of_func_evals[self.kappa - 1] as f64;
            } else {
                self.step_size = self.step_size_at_interpolation_level[self.kappa - 1]
                    * self.num_of_func_evals[k_new - 1] as f64
                    / self.num_of_func_evals[self.kappa - 1] as f64;
            }

            self.k = k_new;
            self.step_size = self.sign * abs!(self.step_size);
        }
    }

    fn dense_component_calculation<S>(
        &mut self,
        x: f64,
        y: &[f64],
        ipoint: &[usize],
        dense_components: &mut [f64],
        system: &mut S,
    ) -> Result<usize, Error>
    where
        S: System,
    {
        // kmit is mu of the paper
        let kmit = 2 * self.kappa - self.interpolation_formula_degree + 1;
        for i in 0..self.dense_component_count {
            // TODO all these indices were indexing uninitialised values of the array, dense_components(i), which returned 1 (first index in fortran).
            dense_components[i] = y[0];
            dense_components[self.dense_component_count + i] = self.step_size * self.dz[0];
            dense_components[2 * self.dense_component_count + i] = self.t[0][0];
        }

        // Compute solution at mid-point
        for j in 1..self.kappa {
            _ = self.update_ysafe(1, j);
        }
        for i in 0..self.dense_component_count {
            dense_components[4 * self.dense_component_count + i] = self.ysafe[i][0];
        }
        // Compute first derivative at right end
        for (i, yh1_val) in self.yh1.iter_mut().enumerate() {
            *yh1_val = self.t[i][0];
        }

        system.derive(x, &self.yh1, &mut self.yh2)?;

        for i in 0..self.dense_component_count {
            // TODO self.yh2 indices were indexing uninitialised values of the array, dense_components(i), which returned 1 (first index in fortran).
            dense_components[3 * self.dense_component_count + i] = self.yh2[0] * self.step_size;
        }

        for kmi in 1..=kmit {
            // Compute the kmi-th derivative at mid-point
            let first = (kmi + 1) / 2 - 1;
            for kk in first..self.kappa {
                let factor_nj = (self.step_num_seq_nj[kk] as f64 / 2.0).powi(kmi as i32 - 1);
                self.ipt = ipoint[kk + 1] - 2 * (kk + 1) + kmi;
                for i in 0..self.dense_component_count {
                    self.ysafe[i][kk] = self.fsafe[i][self.ipt - 1] * factor_nj;
                }
            }

            for j in (first + 1)..self.kappa {
                _ = self.update_ysafe(first + 1, j);
            }

            let offset = (kmi + 4) * self.dense_component_count;
            for i in 0..self.dense_component_count {
                dense_components[offset + i] = self.ysafe[i][first] * self.step_size;
            }

            if kmi == kmit {
                break;
            }

            // Compute differences
            for kk in ((kmi + 2) / 2)..=self.kappa {
                let first = ipoint[kk];
                let mut last = ipoint[kk - 1] + kmi;
                if (kmi == 1) && (self.step_size_sequence == 4) {
                    last += 2;
                }
                for l in (last..first).rev().step_by(2) {
                    for f in self.fsafe.iter_mut().take(self.dense_component_count) {
                        f[l] -= f[l - 2];
                    }
                }

                if (kmi == 1) && (self.step_size_sequence == 4) {
                    let l = last - 2;
                    // TODO self.dz indices were indexing uninitialised values of the array, dense_components(i), which returned 1 (first index in fortran).
                    for f in self.fsafe.iter_mut().take(self.dense_component_count) {
                        f[l] -= self.dz[0];
                    }
                }
            }

            // Compute differences
            for kk in ((kmi + 2) / 2)..=self.kappa {
                let first = ipoint[kk] - 1;
                let last = ipoint[kk - 1] + kmi + 1;
                for l in (last..first).rev().step_by(2) {
                    for f in self.fsafe.iter_mut().take(self.dense_component_count) {
                        f[l] -= f[l - 2];
                    }
                }
            }
        }

        Ok(kmit)
    }
}
