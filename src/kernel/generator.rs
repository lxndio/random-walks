use crate::kernel::Kernel;
use thiserror::Error;

pub trait KernelGenerator {
    fn prepare(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError>;
    fn generate(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError>;
    fn generates_qty(&self) -> usize;
    fn name(&self) -> (String, String);
}

#[derive(Error, Debug)]
pub enum KernelGeneratorError {
    #[error("one kernel required, found none")]
    OneKernelRequired,
    #[error("multiple kernels required, not enough kernels were found")]
    NotEnoughKernels,
    #[error("kernel size must be odd")]
    SizeEven,
}
