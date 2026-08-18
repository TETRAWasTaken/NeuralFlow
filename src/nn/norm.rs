use super::module::Module;
use crate::tensor::Tensor;

pub struct LayerNorm {
    pub gamma: Tensor, 
    pub beta: Tensor, 
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        let gamma = Tensor::new(vec![1.0; normalized_shape], (1, normalized_shape));
        let beta = Tensor::new(vec![0.0; normalized_shape], (1, normalized_shape));
        Self {
            gamma,
            beta,
            eps: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (b, d) = input.0.borrow().shape;
        let x = input.0.borrow().data.clone();
        let gamma_data = self.gamma.0.borrow().data.clone();
        let beta_data = self.beta.0.borrow().data.clone();

        let mut out_data = vec![0.0; b * d];
        let mut x_hat = vec![0.0; b * d];
        let mut inv_std = vec![0.0; b];

        for i in 0..b {
            let row_start = i * d;
            let row = &x[row_start..row_start + d];
            let mean: f32 = row.iter().sum::<f32>() / (d as f32);
            let var: f32 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (d as f32);
            let istd = 1.0 / (var + self.eps).sqrt();
            inv_std[i] = istd;
            for j in 0..d {
                let idx = row_start + j;
                let x_hat_val = (x[idx] - mean) * istd;
                x_hat[idx] = x_hat_val;
                out_data[idx] = x_hat_val * gamma_data[j] + beta_data[j];
            }
        }

        let out = Tensor::new(out_data, (b, d));
        out.0.borrow_mut().prev = vec![input.clone(), self.gamma.clone(), self.beta.clone()];

        let input_clone = input.clone();
        let gamma_clone = self.gamma.clone();
        let beta_clone = self.beta.clone();
        let out_clone = out.clone();

        out.0.borrow_mut().backward = Some(Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();
            let g_data = gamma_clone.0.borrow().data.clone();

            let mut inp_grad = input_clone.0.borrow_mut();
            let mut gamma_grad = gamma_clone.0.borrow_mut();
            let mut beta_grad = beta_clone.0.borrow_mut();

            for i in 0..b {
                let row_start = i * d;
                let istd = inv_std[i];
                let mut sum_dout_gamma = 0.0;
                let mut sum_dout_gamma_xhat = 0.0;

                for j in 0..d {
                    let idx = row_start + j;
                    let dout = out_grad[idx];
                    let xh = x_hat[idx];
                    gamma_grad.grad[j] += dout * xh;
                    beta_grad.grad[j] += dout;

                    let dout_gamma = dout * g_data[j];
                    sum_dout_gamma += dout_gamma;
                    sum_dout_gamma_xhat += dout_gamma * xh;
                }

                for j in 0..d {
                    let idx = row_start + j;
                    let dout_gamma = out_grad[idx] * g_data[j];
                    let xh = x_hat[idx];
                    let dx = (istd / d as f32)
                        * (d as f32 * dout_gamma - sum_dout_gamma - xh * sum_dout_gamma_xhat);
                    inp_grad.grad[idx] += dx;
                }
            }
        }));

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}