use super::inner::Tensor;

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = self.0.borrow().shape;
        let (k_other, n) = other.0.borrow().shape;
        assert_eq!(k, k_other, "Shape mismatch for MatMul");

        let a_data = self.0.borrow().data.clone();
        let b_data = other.0.borrow().data.clone();
        let mut out_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                out_data[i * n + j] = sum;
            }
        }

        let out = Tensor::new(out_data, (m, n));
        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();

        out.0.borrow_mut().backward = Some(Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();
            let a_data = self_clone.0.borrow().data.clone();
            let b_data = other_clone.0.borrow().data.clone();

            let (m, k) = self_clone.0.borrow().shape;
            let (_k_other, n) = other_clone.0.borrow().shape;

            let mut self_inner = self_clone.0.borrow_mut();
            for i in 0..m {
                for p in 0..k {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += out_grad[i * n + j] * b_data[p * n + j];
                    }
                    self_inner.grad[i * k + p] += sum;
                }
            }

            let mut other_inner = other_clone.0.borrow_mut();
            for p in 0..k {
                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..m {
                        sum += a_data[i * k + p] * out_grad[i * n + j];
                    }
                    other_inner.grad[p * n + j] += sum;
                }
            }
        }));

        out
    }

    pub fn relu(&self) -> Tensor {
        let inner = self.0.borrow();
        let out_data: Vec<f32> = inner.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        let out = Tensor::new(out_data, inner.shape);
        out.0.borrow_mut().prev = vec![self.clone()];

        let self_clone = self.clone();
        let out_clone = out.clone();

        out.0.borrow_mut().backward = Some(Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();
            let self_data = self_clone.0.borrow().data.clone();
            let mut self_inner = self_clone.0.borrow_mut();
            for i in 0..self_data.len() {
                if self_data[i] > 0.0 {
                    self_inner.grad[i] += out_grad[i];
                }
            }
        }));

        out
    }
}