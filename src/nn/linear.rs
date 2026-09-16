use super::module::Module;
use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::kaiming_normal((in_features, out_features)),
        }
    }

    pub fn random(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::random((in_features, out_features)),
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone()]
    }
}