use crate::tensor::{is_grad_enabled, Tensor};
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 32_768;

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let p_inner = pred.0.borrow();
    let t_inner = target.0.borrow();
    let p_data = &p_inner.data;
    let t_data = &t_inner.data;

    let n = p_data.len();
    let total_loss: f32 = if n > PARALLEL_THRESHOLD {
        p_data
            .par_iter()
            .zip(t_data.par_iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                diff * diff
            })
            .sum()
    } else {
        let mut sum = 0.0;
        for i in 0..n {
            let diff = p_data[i] - t_data[i];
            sum += diff * diff;
        }
        sum
    };
    let loss_val = total_loss / (n as f32);
    drop(p_inner);
    drop(t_inner);

    let loss = Tensor::new(vec![loss_val], (1, 1));

    if is_grad_enabled() {
        loss.0.borrow_mut().prev = vec![pred.clone()];

        let pred_clone = pred.clone();
        let target_clone = target.clone();
        let loss_clone = loss.clone();

        loss.0.borrow_mut().backward = Some(Box::new(move || {
            let loss_grad = loss_clone.0.borrow().grad[0];
            let t_inner = target_clone.0.borrow();
            let t = &t_inner.data;
            let mut p_inner = pred_clone.0.borrow_mut();
            let crate::tensor::inner::TensorInner { data, grad, .. } = &mut *p_inner;

            let scale = (2.0 / n as f32) * loss_grad;
            if n > PARALLEL_THRESHOLD {
                grad.par_iter_mut()
                    .zip(data.par_iter())
                    .zip(t.par_iter())
                    .for_each(|((g, &d), &target_val)| {
                        *g += scale * (d - target_val);
                    });
            } else {
                for i in 0..data.len() {
                    grad[i] += scale * (data[i] - t[i]);
                }
            }
        }));
    }

    loss
}