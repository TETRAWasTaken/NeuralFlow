use super::optimizer::Optimizer;
use crate::tensor::Tensor;

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step(&self, parameters: &[Tensor]) {
        for param in parameters {
            let mut p = param.0.borrow_mut();
            let len = p.data.len();
            for i in 0..len {
                p.data[i] -= self.lr * p.grad[i];
            }
        }
    }
}