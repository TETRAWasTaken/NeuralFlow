use crate::tensor::Tensor;
use std::cell::Cell;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn set_training(&self, _mode: bool) {}

    fn train(&self) {
        self.set_training(true);
    }

    fn eval(&self) {
        self.set_training(false);
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: Cell<bool>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self {
            layers,
            training: Cell::new(true),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            layers: Vec::with_capacity(capacity),
            training: Cell::new(true),
        }
    }

    pub fn add(&mut self, module: impl Module + 'static) {
        self.layers.push(Box::new(module));
    }

    pub fn layers(&self) -> &[Box<dyn Module>] {
        &self.layers
    }

    pub fn is_training(&self) -> bool {
        self.training.get()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn set_training(&self, mode: bool) {
        self.training.set(mode);
        for layer in &self.layers {
            layer.set_training(mode);
        }
    }
}