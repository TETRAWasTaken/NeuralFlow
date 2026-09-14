use super::optimizer::Optimizer;
use crate::tensor::Tensor;
use rayon::prelude::*;

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

const PARALLEL_THRESHOLD: usize = 32_768;

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[Tensor]) {
        let lr = self.lr;
        for param in parameters {
            let mut p = param.0.borrow_mut();
            let crate::tensor::inner::TensorInner { data, grad, .. } = &mut *p;
            if data.len() > PARALLEL_THRESHOLD {
                data.par_iter_mut()
                    .zip(grad.par_iter())
                    .for_each(|(d, &g)| {
                        *d -= lr * g;
                    });
            } else {
                for i in 0..data.len() {
                    data[i] -= lr * grad[i];
                }
            }
        }
    }
}