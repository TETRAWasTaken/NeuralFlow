use super::optimizer::Optimizer;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: usize,

    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &[Tensor]) {
        self.t += 1;
        let t = self.t as f32;

        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;

        for param in parameters {
            let id = Rc::as_ptr(&param.0) as usize;

            let mut inner = param.0.borrow_mut();
            let len = inner.data.len();

            let m_buf = self.m.entry(id).or_insert_with(|| vec![0.0; len]);
            let v_buf = self.v.entry(id).or_insert_with(|| vec![0.0; len]);

            for i in 0..len {
                let g = inner.grad[i];

                m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
                v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;

                let m_hat = m_buf[i] / bc1;
                let v_hat = v_buf[i] / bc2;

                inner.data[i] -=
                    lr * m_hat / (v_hat.sqrt() + eps) + lr * self.weight_decay * inner.data[i];
            }
        }
    }
}
