use super::module::Module;
use crate::tensor::Tensor;
use std::cell::Cell;

pub struct Dropout {
    pub p: f32,
    pub is_training: Cell<bool>,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be between 0.0 and 1.0"
        );
        Self {
            p,
            is_training: Cell::new(true),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.is_training.get() || self.p == 0.0 {
            return input.clone();
        }

        let inner = input.0.borrow();
        let scale = 1.0 / (1.0 - self.p);
        let mut out_data = vec![0.0; inner.data.len()];
        let mut mask = vec![0.0; inner.data.len()];

        for i in 0..inner.data.len() {
            if rand::random::<f32>() < 1.0 - self.p {
                mask[i] = scale;
                out_data[i] = inner.data[i] * scale;
            }
        }

        let out = Tensor::new(out_data, inner.shape);
        drop(inner);

        if crate::tensor::is_grad_enabled() {
            out.0.borrow_mut().prev = vec![input.clone()];

            let input_clone = input.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut inp_grad = input_clone.0.borrow_mut();

                for i in 0..out_grad.len() {
                    inp_grad.grad[i] += out_grad[i] * mask[i];
                }
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn set_training(&self, mode: bool) {
        self.is_training.set(mode);
    }

}