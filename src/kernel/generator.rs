use thiserror::Error;

use crate::kernel::Kernel;
use crate::kernel::DirKernel;

pub trait KernelGenerator {
    fn prepare(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError>;
    fn generate(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError>;
    fn generates_qty(&self) -> usize;
    fn name(&self) -> (String, String);
}

pub trait DirKernelGenerator {
    fn prepare(&self, kernels: &mut Vec<DirKernel>) -> Result<(), DirKernelGeneratorError>;
    fn generate(&self, kernels: &mut Vec<DirKernel>) -> Result<(), DirKernelGeneratorError>;
    fn num_directions(&self) -> usize;
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


#[derive(Error, Debug)]
pub enum DirKernelGeneratorError {
    #[error("one kernel required, found none")]
    OneKernelRequired,
    #[error("multiple kernels required, not enough kernels were found")]
    NotEnoughKernels,
    #[error("kernel size must be odd")]
    SizeEven,
}
