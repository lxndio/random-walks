//! A collection of different steppers for use in the dynamic program.

use crate::dp::propdp::ProbabilityDynamicProgram;
use crate::dp::DynamicProgram;
use num::BigUint;

pub mod simple;

pub trait Stepper {
    fn step(&self, dp: &DynamicProgram, x: isize, y: isize, t: usize) -> BigUint;
    fn name(&self, short: bool) -> String;
}

pub trait ProbabilityStepper {
    fn step(&self, dp: &ProbabilityDynamicProgram, x: isize, y: isize) -> BigUint;
    fn name(&self, short: bool) -> String;
}
