use super::inner::{is_grad_enabled, Tensor};
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 32_768;

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = self.0.borrow().shape;
        let (k_other, n) = other.0.borrow().shape;
        assert_eq!(k, k_other, "Shape mismatch for MatMul");

        let a_inner = self.0.borrow();
        let b_inner = other.0.borrow();
        let a_data = &a_inner.data;
        let b_data = &b_inner.data;
        let mut out_data = vec![0.0; m * n];

        unsafe {
            crate::tensor::blas::gemm(
                false,
                false,
                m,
                n,
                k,
                1.0,
                a_data.as_ptr(),
                b_data.as_ptr(),
                0.0,
                out_data.as_mut_ptr(),
            );
        }
        drop(a_inner);
        drop(b_inner);

        let out = Tensor::new(out_data, (m, n));

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![self.clone(), other.clone()];

            let self_clone = self.clone();
            let other_clone = other.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;

                let (m, k) = self_clone.0.borrow().shape;
                let (_k_other, n) = other_clone.0.borrow().shape;

                // Gradient w.r.t self (A): grad_A += out_grad * B^T
                {
                    let b_inner = other_clone.0.borrow();
                    let b_data = &b_inner.data;
                    let mut self_inner = self_clone.0.borrow_mut();
                    unsafe {
                        crate::tensor::blas::gemm(
                            false,
                            true, // B transposed: (k, n) -> (n, k)
                            m,
                            k,
                            n,
                            1.0,
                            out_grad.as_ptr(),
                            b_data.as_ptr(),
                            1.0, // accumulate directly into grad
                            self_inner.grad.as_mut_ptr(),
                        );
                    }
                }

                // Gradient w.r.t other (B): grad_B += A^T * out_grad
                {
                    let a_inner = self_clone.0.borrow();
                    let a_data = &a_inner.data;
                    let mut other_inner = other_clone.0.borrow_mut();
                    unsafe {
                        crate::tensor::blas::gemm(
                            true, // A transposed: (m, k) -> (k, m)
                            false,
                            k,
                            n,
                            m,
                            1.0,
                            a_data.as_ptr(),
                            out_grad.as_ptr(),
                            1.0, // accumulate directly into grad
                            other_inner.grad.as_mut_ptr(),
                        );
                    }
                }
            }));
        }

        out
    }

    pub fn relu(&self) -> Tensor {
        let inner = self.0.borrow();
        let len = inner.data.len();
        let out_data: Vec<f32> = if len > PARALLEL_THRESHOLD {
            inner
                .data
                .par_iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect()
        } else {
            inner
                .data
                .iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect()
        };
        let shape = inner.shape;
        let shape4d = inner.shape4d;
        drop(inner);

        let out = if let Some(s4) = shape4d {
            Tensor::new_4d(out_data, s4)
        } else {
            Tensor::new(out_data, shape)
        };

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![self.clone()];

            let self_clone = self.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut self_inner = self_clone.0.borrow_mut();
                let crate::tensor::inner::TensorInner { data, grad, .. } = &mut *self_inner;

                for i in 0..data.len() {
                    if data[i] > 0.0 {
                        grad[i] += out_grad[i];
                    }
                }
            }));
        }

        out
    }

    pub fn sigmoid(&self) -> Tensor {
        let inner = self.0.borrow();
        let len = inner.data.len();
        let out_data: Vec<f32> = if len > PARALLEL_THRESHOLD {
            inner
                .data
                .par_iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect()
        } else {
            inner
                .data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect()
        };
        let shape = inner.shape;
        let shape4d = inner.shape4d;
        drop(inner);

        let out = if let Some(s4) = shape4d {
            Tensor::new_4d(out_data, s4)
        } else {
            Tensor::new(out_data, shape)
        };

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![self.clone()];

            let self_clone = self.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut self_inner = self_clone.0.borrow_mut();
                let crate::tensor::inner::TensorInner { data, grad, .. } = &mut *self_inner;

                for i in 0..data.len() {
                    let s = 1.0 / (1.0 + (-data[i]).exp());
                    grad[i] += out_grad[i] * (s * (1.0 - s));
                }
            }));
        }

        out
    }

    pub fn tanh(&self) -> Tensor {
        let inner = self.0.borrow();
        let len = inner.data.len();
        let out_data: Vec<f32> = if len > PARALLEL_THRESHOLD {
            inner.data.par_iter().map(|&x| x.tanh()).collect()
        } else {
            inner.data.iter().map(|&x| x.tanh()).collect()
        };
        let shape = inner.shape;
        let shape4d = inner.shape4d;
        drop(inner);

        let out = if let Some(s4) = shape4d {
            Tensor::new_4d(out_data, s4)
        } else {
            Tensor::new(out_data, shape)
        };

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![self.clone()];

            let self_clone = self.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let out_data = &out_inner.data;
                let mut self_inner = self_clone.0.borrow_mut();
                let grad = &mut self_inner.grad;

                for i in 0..out_data.len() {
                    let t = out_data[i];
                    grad[i] += out_grad[i] * (1.0 - t * t);
                }
            }));
        }

        out
    }

    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
        let inner = self.0.borrow();
        let len = inner.data.len();
        let out_data: Vec<f32> = if len > PARALLEL_THRESHOLD {
            inner
                .data
                .par_iter()
                .map(|&x| if x > 0.0 { x } else { alpha * x })
                .collect()
        } else {
            inner
                .data
                .iter()
                .map(|&x| if x > 0.0 { x } else { alpha * x })
                .collect()
        };
        let shape = inner.shape;
        let shape4d = inner.shape4d;
        drop(inner);

        let out = if let Some(s4) = shape4d {
            Tensor::new_4d(out_data, s4)
        } else {
            Tensor::new(out_data, shape)
        };

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![self.clone()];

            let self_clone = self.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut self_inner = self_clone.0.borrow_mut();
                let crate::tensor::inner::TensorInner { data, grad, .. } = &mut *self_inner;

                for i in 0..data.len() {
                    let scale = if data[i] > 0.0 { 1.0 } else { alpha };
                    grad[i] += out_grad[i] * scale;
                }
            }));
        }

        out
    }
}


