use super::module::Module;
use crate::tensor::{is_grad_enabled, Tensor};

pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (batch_size, total_features) = {
            let inner = input.0.borrow();
            let batch = if let Some(s4) = inner.shape4d {
                s4.0
            } else {
                inner.shape.0
            };
            let total = inner.data.len();
            let feat = if batch > 0 { total / batch } else { 0 };
            (batch, feat)
        };

        let data = input.0.borrow().data.clone();
        let out = Tensor::new(data, (batch_size, total_features));

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![input.clone()];

            let input_clone = input.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut in_inner = input_clone.0.borrow_mut();
                let in_grad = &mut in_inner.grad;

                for (ig, &og) in in_grad.iter_mut().zip(out_grad.iter()) {
                    *ig += og;
                }
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
