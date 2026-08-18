use super::module::Module;
use crate::tensor::Tensor;

pub struct ReLu;
impl Module for ReLu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct LeakyReLu {
    pub alpha: f32,
}
impl Module for LeakyReLu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.alpha)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}