mod odex;
mod rkdp45;
use odex::Odex;
use rkdp45::Rkdp45;

use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use thiserror::Error;

#[derive(Serialize, Deserialize, Debug)]
pub enum IntegratorType {
    Odex(Odex),
    Rkdp45(Rkdp45),
}

pub trait Integrator: Serialize + for<'a> Deserialize<'a> {
    // Set the initial and final conditions for the integration.
    fn initialise(&mut self, x_initial: f64, x_final: f64, y_initial: &[f64]) -> Result<(), Error>;
    // Begin the integration, with system containing derivation, solout, and associated data.
    fn integrate<S: System>(&mut self, system: &mut S) -> Result<Stats, Error>;
    // The final integrated quantities.
    fn y_final(&self) -> Vec<f64>;
}

// Allow calling the public integrator methods common to each variant on the `IntegratorType` enum directly.
impl Integrator for IntegratorType {
    fn initialise(&mut self, x_initial: f64, x_final: f64, y_initial: &[f64]) -> Result<(), Error> {
        match self {
            IntegratorType::Odex(integrator) => {
                integrator.initialise(x_initial, x_final, y_initial)?;
            }
            IntegratorType::Rkdp45(integrator) => {
                integrator.initialise(x_initial, x_final, y_initial)?;
            }
        }
        Ok(())
    }

    fn integrate<S: System>(&mut self, system: &mut S) -> Result<Stats, Error> {
        let stats = match self {
            IntegratorType::Odex(integrator) => integrator.integrate(system)?,
            IntegratorType::Rkdp45(integrator) => integrator.integrate(system)?,
        };
        Ok(stats)
    }

    fn y_final(&self) -> Vec<f64> {
        match self {
            IntegratorType::Odex(integrator) => integrator.y_final(),
            IntegratorType::Rkdp45(integrator) => integrator.y_final(),
        }
    }
}

/// Any type implementing the `System` trait must provide implementations for these function signatures,
/// which are called by the integrator for calculating derivatives and outputing intermediate solutions.
/// `Data` and `Output` are user defined types specifiying opaque input data and an output sink.
pub trait System {
    type Data;
    type Output;
    fn new(output: Self::Output, data: Self::Data) -> Self;
    // Calculates the derivates of y with respect to x
    fn derive(
        &mut self,
        x: f64,
        y: &[f64],
        dy: &mut [f64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    // Outputs each step of the solution
    fn solout(&mut self, x: f64, y: &[f64])
    -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Possible error conditions that may arise during integration.
#[derive(Debug, Error)]
pub enum Error {
    #[error("maximum number of steps reached ({n_step}) at x = {x}")]
    StepLimitReached { x: f64, n_step: usize },
    #[error("invalid {param}")]
    InvalidParameter { param: String },
    #[error("Error in `derive` or `solout` method")]
    External(#[from] Box<dyn std::error::Error + Send + Sync>),
}
/*
impl From<Box<dyn std::error::Error>> for Error {

}
*/

/// Contains some statistics of the integration.
#[derive(Debug, Default, Deserialize, Serialize, Copy, Clone)]
pub struct Stats {
    pub function_calls: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    // complete_steps should be accepted_steps + rejected_steps.
    pub complete_steps: usize,
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "function evaluations: {}, ", self.function_calls)?;
        write!(f, "accepted steps: {}, ", self.accepted_steps)?;
        write!(f, "rejected steps: {}, ", self.rejected_steps)?;
        write!(f, "complete steps: {}", self.complete_steps)?;

        Ok(())
    }
}
