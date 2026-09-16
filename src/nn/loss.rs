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

pub fn cross_entropy_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let p_inner = pred.0.borrow();
    let t_inner = target.0.borrow();
    let (n_samples, n_classes) = p_inner.shape;
    let (t_rows, t_cols) = t_inner.shape;
    let t_len = t_rows * t_cols;

    assert!(
        n_samples > 0 && n_classes > 0,
        "Prediction shape cannot have zero dimensions: {:?}",
        p_inner.shape
    );

    let is_class_index = if n_classes == 1 {
        assert_eq!(
            t_len, n_samples,
            "Target shape ({}, {}) does not match binary pred shape ({}, {})",
            t_rows, t_cols, n_samples, n_classes
        );
        true
    } else if t_len == n_samples {
        true
    } else if t_len == n_samples * n_classes {
        assert_eq!(
            (t_rows, t_cols),
            (n_samples, n_classes),
            "Target shape ({}, {}) must match pred shape ({}, {}) for distribution targets",
            t_rows, t_cols, n_samples, n_classes
        );
        false
    } else {
        panic!(
            "Shape mismatch in cross_entropy_loss: pred shape ({}, {}), target shape ({}, {}). \
             Target must either have shape ({}, 1) with class indices or ({}, {}) with target distributions.",
            n_samples, n_classes, t_rows, t_cols, n_samples, n_samples, n_classes
        );
    };

    let p_data = &p_inner.data;
    let t_data = &t_inner.data;
    let n_total = n_samples * n_classes;

    let total_loss: f32 = if n_classes == 1 {
        // Binary cross-entropy with logits: max(z, 0) - z * y + ln(1 + exp(-|z|))
        if n_samples > PARALLEL_THRESHOLD {
            p_data
                .par_iter()
                .zip(t_data.par_iter())
                .map(|(&z, &y)| {
                    let max_z = if z > 0.0 { z } else { 0.0 };
                    max_z - z * y + (1.0 + (-z.abs()).exp()).ln()
                })
                .sum()
        } else {
            let mut sum = 0.0;
            for i in 0..n_samples {
                let z = p_data[i];
                let y = t_data[i];
                let max_z = if z > 0.0 { z } else { 0.0 };
                sum += max_z - z * y + (1.0 + (-z.abs()).exp()).ln();
            }
            sum
        }
    } else if n_total > PARALLEL_THRESHOLD {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let row_z = &p_data[i * n_classes..(i + 1) * n_classes];
                let mut max_z = f32::NEG_INFINITY;
                for &val in row_z {
                    if val > max_z {
                        max_z = val;
                    }
                }
                let mut sum_exp = 0.0;
                for &val in row_z {
                    sum_exp += (val - max_z).exp();
                }
                let log_sum_exp = max_z + sum_exp.ln();

                if is_class_index {
                    let target_class = t_data[i].round() as usize;
                    assert!(
                        target_class < n_classes,
                        "Target class index {} is out of bounds for {} classes",
                        target_class,
                        n_classes
                    );
                    log_sum_exp - row_z[target_class]
                } else {
                    let row_t = &t_data[i * n_classes..(i + 1) * n_classes];
                    let mut row_loss = 0.0;
                    for c in 0..n_classes {
                        row_loss += row_t[c] * (log_sum_exp - row_z[c]);
                    }
                    row_loss
                }
            })
            .sum()
    } else {
        let mut sum = 0.0;
        for i in 0..n_samples {
            let row_z = &p_data[i * n_classes..(i + 1) * n_classes];
            let mut max_z = f32::NEG_INFINITY;
            for &val in row_z {
                if val > max_z {
                    max_z = val;
                }
            }
            let mut sum_exp = 0.0;
            for &val in row_z {
                sum_exp += (val - max_z).exp();
            }
            let log_sum_exp = max_z + sum_exp.ln();

            if is_class_index {
                let target_class = t_data[i].round() as usize;
                assert!(
                    target_class < n_classes,
                    "Target class index {} is out of bounds for {} classes",
                    target_class,
                    n_classes
                );
                sum += log_sum_exp - row_z[target_class];
            } else {
                let row_t = &t_data[i * n_classes..(i + 1) * n_classes];
                for c in 0..n_classes {
                    sum += row_t[c] * (log_sum_exp - row_z[c]);
                }
            }
        }
        sum
    };

    let loss_val = total_loss / (n_samples as f32);
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

            let scale = loss_grad / (n_samples as f32);

            if n_classes == 1 {
                if n_samples > PARALLEL_THRESHOLD {
                    grad.par_iter_mut()
                        .zip(data.par_iter())
                        .zip(t.par_iter())
                        .for_each(|((g, &z), &y)| {
                            let sig = 1.0 / (1.0 + (-z).exp());
                            *g += scale * (sig - y);
                        });
                } else {
                    for i in 0..n_samples {
                        let z = data[i];
                        let y = t[i];
                        let sig = 1.0 / (1.0 + (-z).exp());
                        grad[i] += scale * (sig - y);
                    }
                }
            } else if n_total > PARALLEL_THRESHOLD {
                grad.par_chunks_mut(n_classes)
                    .zip(data.par_chunks(n_classes))
                    .enumerate()
                    .for_each(|(i, (row_grad, row_z))| {
                        let mut max_z = f32::NEG_INFINITY;
                        for &val in row_z {
                            if val > max_z {
                                max_z = val;
                            }
                        }
                        let mut sum_exp = 0.0;
                        for &val in row_z {
                            sum_exp += (val - max_z).exp();
                        }
                        let inv_sum_exp = 1.0 / sum_exp;

                        if is_class_index {
                            let target_class = t[i].round() as usize;
                            for c in 0..n_classes {
                                let p_c = (row_z[c] - max_z).exp() * inv_sum_exp;
                                let target_val = if c == target_class { 1.0 } else { 0.0 };
                                row_grad[c] += scale * (p_c - target_val);
                            }
                        } else {
                            let row_t = &t[i * n_classes..(i + 1) * n_classes];
                            let mut sum_t = 0.0;
                            for &y in row_t {
                                sum_t += y;
                            }
                            for c in 0..n_classes {
                                let p_c = (row_z[c] - max_z).exp() * inv_sum_exp;
                                row_grad[c] += scale * (p_c * sum_t - row_t[c]);
                            }
                        }
                    });
            } else {
                for i in 0..n_samples {
                    let row_z = &data[i * n_classes..(i + 1) * n_classes];
                    let mut max_z = f32::NEG_INFINITY;
                    for &val in row_z {
                        if val > max_z {
                            max_z = val;
                        }
                    }
                    let mut sum_exp = 0.0;
                    for &val in row_z {
                        sum_exp += (val - max_z).exp();
                    }
                    let inv_sum_exp = 1.0 / sum_exp;

                    if is_class_index {
                        let target_class = t[i].round() as usize;
                        for c in 0..n_classes {
                            let p_c = (row_z[c] - max_z).exp() * inv_sum_exp;
                            let target_val = if c == target_class { 1.0 } else { 0.0 };
                            grad[i * n_classes + c] += scale * (p_c - target_val);
                        }
                    } else {
                        let row_t = &t[i * n_classes..(i + 1) * n_classes];
                        let mut sum_t = 0.0;
                        for &y in row_t {
                            sum_t += y;
                        }
                        for c in 0..n_classes {
                            let p_c = (row_z[c] - max_z).exp() * inv_sum_exp;
                            grad[i * n_classes + c] += scale * (p_c * sum_t - row_t[c]);
                        }
                    }
                }
            }
        }));
    }

    loss
}
pub fn bce_with_logits_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    cross_entropy_loss(pred, target)
}
pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let p_inner = pred.0.borrow();
    let t_inner = target.0.borrow();
    let n = p_inner.data.len();
    assert_eq!(
        n,
        t_inner.data.len(),
        "Prediction and target element count mismatch: {} vs {}",
        n,
        t_inner.data.len()
    );

    let p_data = &p_inner.data;
    let t_data = &t_inner.data;
    const EPS: f32 = 1e-7;

    let total_loss: f32 = if n > PARALLEL_THRESHOLD {
        p_data
            .par_iter()
            .zip(t_data.par_iter())
            .map(|(&p, &y)| {
                let p_clamped = p.clamp(EPS, 1.0 - EPS);
                -(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln())
            })
            .sum()
    } else {
        let mut sum = 0.0;
        for i in 0..n {
            let p_clamped = p_data[i].clamp(EPS, 1.0 - EPS);
            let y = t_data[i];
            sum += -(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln());
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

            let scale = loss_grad / (n as f32);
            if n > PARALLEL_THRESHOLD {
                grad.par_iter_mut()
                    .zip(data.par_iter())
                    .zip(t.par_iter())
                    .for_each(|((g, &p), &y)| {
                        let p_clamped = p.clamp(EPS, 1.0 - EPS);
                        let d_loss = (p_clamped - y) / (p_clamped * (1.0 - p_clamped));
                        *g += scale * d_loss;
                    });
            } else {
                for i in 0..data.len() {
                    let p_clamped = data[i].clamp(EPS, 1.0 - EPS);
                    let y = t[i];
                    let d_loss = (p_clamped - y) / (p_clamped * (1.0 - p_clamped));
                    grad[i] += scale * d_loss;
                }
            }
        }));
    }

    loss
}