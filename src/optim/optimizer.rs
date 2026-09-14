use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, parameters: &[Tensor]);
}