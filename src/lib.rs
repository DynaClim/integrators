// Macros for simple math operations: {min, max, etc}
#[macro_use]
extern crate math_macros;
pub use math_macros::*;
pub mod integrator;
pub use integrator::{Error, Integrator, IntegratorType, Stats, System};
