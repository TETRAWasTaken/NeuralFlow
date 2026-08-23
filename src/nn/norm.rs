use super::module::Module;
use crate::tensor::Tensor;
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 32_768;

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
        let inp_inner = input.0.borrow();
        let gamma_inner = self.gamma.0.borrow();
        let beta_inner = self.beta.0.borrow();
        let x = &inp_inner.data;
        let gamma_data = &gamma_inner.data;
        let beta_data = &beta_inner.data;

        let mut out_data = vec![0.0; b * d];
        let mut x_hat = vec![0.0; b * d];
        let mut inv_std = vec![0.0; b];
        let eps = self.eps;

        let total_elements = b * d;
        if total_elements > PARALLEL_THRESHOLD {
            out_data
                .par_chunks_exact_mut(d)
                .zip(x_hat.par_chunks_exact_mut(d))
                .zip(inv_std.par_iter_mut())
                .zip(x.par_chunks_exact(d))
                .for_each(|(((out_row, xh_row), istd), x_row)| {
                    let mean: f32 = x_row.iter().sum::<f32>() / (d as f32);
                    let var: f32 = x_row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (d as f32);
                    let inv_s = 1.0 / (var + eps).sqrt();
                    *istd = inv_s;

                    for j in 0..d {
                        let norm_val = (x_row[j] - mean) * inv_s;
                        xh_row[j] = norm_val;
                        out_row[j] = norm_val * gamma_data[j] + beta_data[j];
                    }
                });
        } else {
            for i in 0..b {
                let row_start = i * d;
                let row = &x[row_start..row_start + d];
                let mean: f32 = row.iter().sum::<f32>() / (d as f32);
                let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (d as f32);
                let istd = 1.0 / (var + eps).sqrt();
                inv_std[i] = istd;

                for j in 0..d {
                    let idx = row_start + j;
                    let norm_val = (x[idx] - mean) * istd;
                    x_hat[idx] = norm_val;
                    out_data[idx] = norm_val * gamma_data[j] + beta_data[j];
                }
            }
        }

        drop(inp_inner);
        drop(gamma_inner);
        drop(beta_inner);

        let out = Tensor::new(out_data, (b, d));

        if crate::tensor::is_grad_enabled() {
            out.0.borrow_mut().prev = vec![input.clone(), self.gamma.clone(), self.beta.clone()];

            let input_clone = input.clone();
            let gamma_clone = self.gamma.clone();
            let beta_clone = self.beta.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;

                let mut inp_inner = input_clone.0.borrow_mut();
                let mut gamma_inner = gamma_clone.0.borrow_mut();
                let mut beta_inner = beta_clone.0.borrow_mut();

                let crate::tensor::inner::TensorInner { data: g_data, grad: gamma_grad, .. } = &mut *gamma_inner;
                let beta_grad = &mut beta_inner.grad;
                let inp_grad = &mut inp_inner.grad;

                // Accumulate parameter gradients across batch
                for i in 0..b {
                    let row_start = i * d;
                    for j in 0..d {
                        let dout = out_grad[row_start + j];
                        let xh = x_hat[row_start + j];
                        gamma_grad[j] += dout * xh;
                        beta_grad[j] += dout;
                    }
                }

                // Row-by-row input gradient backpropagation
                if total_elements > PARALLEL_THRESHOLD {
                    inp_grad
                        .par_chunks_exact_mut(d)
                        .zip(out_grad.par_chunks_exact(d))
                        .zip(x_hat.par_chunks_exact(d))
                        .zip(inv_std.par_iter())
                        .for_each(|(((inp_grad_row, out_grad_row), xh_row), &istd)| {
                            let mut sum_dout_gamma = 0.0;
                            let mut sum_dout_gamma_xhat = 0.0;

                            for j in 0..d {
                                let dout_gamma = out_grad_row[j] * g_data[j];
                                let xh = xh_row[j];
                                sum_dout_gamma += dout_gamma;
                                sum_dout_gamma_xhat += dout_gamma * xh;
                            }

                            let scale = istd / (d as f32);
                            let d_f32 = d as f32;
                            for j in 0..d {
                                let dout_gamma = out_grad_row[j] * g_data[j];
                                let xh = xh_row[j];
                                let dx = scale * (d_f32 * dout_gamma - sum_dout_gamma - xh * sum_dout_gamma_xhat);
                                inp_grad_row[j] += dx;
                            }
                        });
                } else {
                    for i in 0..b {
                        let row_start = i * d;
                        let istd = inv_std[i];
                        let mut sum_dout_gamma = 0.0;
                        let mut sum_dout_gamma_xhat = 0.0;

                        for j in 0..d {
                            let dout_gamma = out_grad[row_start + j] * g_data[j];
                            let xh = x_hat[row_start + j];
                            sum_dout_gamma += dout_gamma;
                            sum_dout_gamma_xhat += dout_gamma * xh;
                        }

                        let scale = istd / (d as f32);
                        let d_f32 = d as f32;
                        for j in 0..d {
                            let dout_gamma = out_grad[row_start + j] * g_data[j];
                            let xh = x_hat[row_start + j];
                            let dx = scale * (d_f32 * dout_gamma - sum_dout_gamma - xh * sum_dout_gamma_xhat);
                            inp_grad[row_start + j] += dx;
                        }
                    }
                }
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}