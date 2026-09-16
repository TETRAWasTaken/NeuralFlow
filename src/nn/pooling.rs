use super::module::Module;
use crate::tensor::{is_grad_enabled, Tensor};
use rayon::prelude::*;

pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
        }
    }

    pub fn new_with_stride(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel_size, stride }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (n_batch, channels, height, width) = input
            .shape4d()
            .expect("Input to MaxPool2d must be a 4D tensor with shape (N, C, H, W)");

        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;

        assert!(
            height >= k_h && width >= k_w,
            "Input spatial dimensions ({}, {}) smaller than kernel ({}, {})",
            height, width, k_h, k_w
        );

        let out_h = (height - k_h) / s_h + 1;
        let out_w = (width - k_w) / s_w + 1;
        let out_spatial = out_h * out_w;
        let in_spatial = height * width;
        let in_sample_size = channels * in_spatial;
        let out_sample_size = channels * out_spatial;

        let in_inner = input.0.borrow();
        let in_data = &in_inner.data;

        let total_out = n_batch * out_sample_size;
        let mut out_data = vec![0.0; total_out];
        let is_2x2_s2 = k_h == 2 && k_w == 2 && s_h == 2 && s_w == 2;

        let mut max_indices = if is_grad_enabled() {
            Some(vec![0usize; total_out])
        } else {
            None
        };

        if let Some(ref mut indices_buf) = max_indices {
            out_data
                .par_chunks_exact_mut(out_sample_size)
                .zip(indices_buf.par_chunks_exact_mut(out_sample_size))
                .enumerate()
                .for_each(|(n, (out_slice, ind_slice))| {
                    let n_in_offset = n * in_sample_size;
                    let mut out_idx = 0;
                    for c in 0..channels {
                        let c_in_offset = n_in_offset + c * in_spatial;
                        if is_2x2_s2 {
                            for oh in 0..out_h {
                                let ih0 = (oh << 1) * width;
                                let ih1 = ih0 + width;
                                let row0 = c_in_offset + ih0;
                                let row1 = c_in_offset + ih1;
                                for ow in 0..out_w {
                                    let iw = ow << 1;
                                    let idx0 = row0 + iw;
                                    let idx1 = idx0 + 1;
                                    let idx2 = row1 + iw;
                                    let idx3 = idx2 + 1;

                                    let v0 = in_data[idx0];
                                    let v1 = in_data[idx1];
                                    let v2 = in_data[idx2];
                                    let v3 = in_data[idx3];

                                    let (m01, i01) = if v1 > v0 { (v1, idx1) } else { (v0, idx0) };
                                    let (m23, i23) = if v3 > v2 { (v3, idx3) } else { (v2, idx2) };
                                    let (max_val, max_idx) = if m23 > m01 { (m23, i23) } else { (m01, i01) };

                                    out_slice[out_idx] = max_val;
                                    ind_slice[out_idx] = max_idx;
                                    out_idx += 1;
                                }
                            }
                        } else {
                            for oh in 0..out_h {
                                let h_start = oh * s_h;
                                for ow in 0..out_w {
                                    let w_start = ow * s_w;
                                    let mut max_val = f32::NEG_INFINITY;
                                    let mut max_idx = 0;

                                    for ky in 0..k_h {
                                        let ih = h_start + ky;
                                        let row_offset = c_in_offset + ih * width;
                                        for kx in 0..k_w {
                                            let iw = w_start + kx;
                                            let idx = row_offset + iw;
                                            let val = in_data[idx];
                                            if val > max_val {
                                                max_val = val;
                                                max_idx = idx;
                                            }
                                        }
                                    }
                                    out_slice[out_idx] = max_val;
                                    ind_slice[out_idx] = max_idx;
                                    out_idx += 1;
                                }
                            }
                        }
                    }
                });
        } else {
            out_data
                .par_chunks_exact_mut(out_sample_size)
                .enumerate()
                .for_each(|(n, out_slice)| {
                    let n_in_offset = n * in_sample_size;
                    let mut out_idx = 0;
                    for c in 0..channels {
                        let c_in_offset = n_in_offset + c * in_spatial;
                        if is_2x2_s2 {
                            for oh in 0..out_h {
                                let ih0 = (oh << 1) * width;
                                let ih1 = ih0 + width;
                                let row0 = c_in_offset + ih0;
                                let row1 = c_in_offset + ih1;
                                for ow in 0..out_w {
                                    let iw = ow << 1;
                                    let idx0 = row0 + iw;
                                    let idx2 = row1 + iw;

                                    let v0 = in_data[idx0];
                                    let v1 = in_data[idx0 + 1];
                                    let v2 = in_data[idx2];
                                    let v3 = in_data[idx2 + 1];

                                    out_slice[out_idx] = v0.max(v1).max(v2.max(v3));
                                    out_idx += 1;
                                }
                            }
                        } else {
                            for oh in 0..out_h {
                                let h_start = oh * s_h;
                                for ow in 0..out_w {
                                    let w_start = ow * s_w;
                                    let mut max_val = f32::NEG_INFINITY;
                                    for ky in 0..k_h {
                                        let ih = h_start + ky;
                                        let row_offset = c_in_offset + ih * width;
                                        for kx in 0..k_w {
                                            let iw = w_start + kx;
                                            let idx = row_offset + iw;
                                            let val = in_data[idx];
                                            if val > max_val {
                                                max_val = val;
                                            }
                                        }
                                    }
                                    out_slice[out_idx] = max_val;
                                    out_idx += 1;
                                }
                            }
                        }
                    }
                });
        }
        drop(in_inner);

        let out = Tensor::new_4d(out_data, (n_batch, channels, out_h, out_w));

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![input.clone()];

            let input_clone = input.clone();
            let out_clone = out.clone();
            let saved_indices = max_indices.unwrap();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut in_inner = input_clone.0.borrow_mut();
                let in_grad = &mut in_inner.grad;

                in_grad
                    .par_chunks_exact_mut(in_sample_size)
                    .zip(out_grad.par_chunks_exact(out_sample_size))
                    .zip(saved_indices.par_chunks_exact(out_sample_size))
                    .enumerate()
                    .for_each(|(n, ((in_grad_slice, out_grad_slice), indices_slice))| {
                        let base_offset = n * in_sample_size;
                        for (&out_g, &max_idx) in out_grad_slice.iter().zip(indices_slice.iter()) {
                            let local_idx = max_idx - base_offset;
                            in_grad_slice[local_idx] += out_g;
                        }
                    });
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
        }
    }

    pub fn new_with_stride(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel_size, stride }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (n_batch, channels, height, width) = input
            .shape4d()
            .expect("Input to AvgPool2d must be a 4D tensor with shape (N, C, H, W)");

        let (k_h, k_w) = self.kernel_size;
        let (s_h, s_w) = self.stride;

        assert!(
            height >= k_h && width >= k_w,
            "Input spatial dimensions ({}, {}) smaller than kernel ({}, {})",
            height, width, k_h, k_w
        );

        let out_h = (height - k_h) / s_h + 1;
        let out_w = (width - k_w) / s_w + 1;
        let out_spatial = out_h * out_w;
        let in_spatial = height * width;
        let in_sample_size = channels * in_spatial;
        let out_sample_size = channels * out_spatial;
        let pool_area = (k_h * k_w) as f32;
        let is_2x2_s2 = k_h == 2 && k_w == 2 && s_h == 2 && s_w == 2;

        let in_inner = input.0.borrow();
        let in_data = &in_inner.data;

        let total_out = n_batch * out_sample_size;
        let mut out_data = vec![0.0; total_out];

        out_data
            .par_chunks_exact_mut(out_sample_size)
            .enumerate()
            .for_each(|(n, out_slice)| {
                let n_offset = n * in_sample_size;
                let mut out_idx = 0;
                for c in 0..channels {
                    let c_offset = n_offset + c * in_spatial;
                    if is_2x2_s2 {
                        for oh in 0..out_h {
                            let ih0 = (oh << 1) * width;
                            let ih1 = ih0 + width;
                            let row0 = c_offset + ih0;
                            let row1 = c_offset + ih1;
                            for ow in 0..out_w {
                                let iw = ow << 1;
                                let idx0 = row0 + iw;
                                let idx2 = row1 + iw;
                                let sum = in_data[idx0] + in_data[idx0 + 1] + in_data[idx2] + in_data[idx2 + 1];
                                out_slice[out_idx] = sum * 0.25;
                                out_idx += 1;
                            }
                        }
                    } else {
                        for oh in 0..out_h {
                            let h_start = oh * s_h;
                            for ow in 0..out_w {
                                let w_start = ow * s_w;
                                let mut sum = 0.0;
                                for ky in 0..k_h {
                                    let ih = h_start + ky;
                                    let row_offset = c_offset + ih * width;
                                    for kx in 0..k_w {
                                        let iw = w_start + kx;
                                        sum += in_data[row_offset + iw];
                                    }
                                }
                                out_slice[out_idx] = sum / pool_area;
                                out_idx += 1;
                            }
                        }
                    }
                }
            });
        drop(in_inner);

        let out = Tensor::new_4d(out_data, (n_batch, channels, out_h, out_w));

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![input.clone()];

            let input_clone = input.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut in_inner = input_clone.0.borrow_mut();
                let in_grad = &mut in_inner.grad;

                in_grad
                    .par_chunks_exact_mut(in_sample_size)
                    .zip(out_grad.par_chunks_exact(out_sample_size))
                    .for_each(|(in_grad_slice, out_grad_slice)| {
                        let mut out_idx = 0;
                        for c in 0..channels {
                            let c_offset = c * in_spatial;
                            if is_2x2_s2 {
                                let inv_4 = 0.25f32;
                                for oh in 0..out_h {
                                    let ih0 = (oh << 1) * width;
                                    let ih1 = ih0 + width;
                                    let row0 = c_offset + ih0;
                                    let row1 = c_offset + ih1;
                                    for ow in 0..out_w {
                                        let iw = ow << 1;
                                        let idx0 = row0 + iw;
                                        let idx2 = row1 + iw;
                                        let g = out_grad_slice[out_idx] * inv_4;
                                        in_grad_slice[idx0] += g;
                                        in_grad_slice[idx0 + 1] += g;
                                        in_grad_slice[idx2] += g;
                                        in_grad_slice[idx2 + 1] += g;
                                        out_idx += 1;
                                    }
                                }
                            } else {
                                let inv_area = 1.0 / pool_area;
                                for oh in 0..out_h {
                                    let h_start = oh * s_h;
                                    for ow in 0..out_w {
                                        let w_start = ow * s_w;
                                        let grad_scaled = out_grad_slice[out_idx] * inv_area;
                                        for ky in 0..k_h {
                                            let ih = h_start + ky;
                                            let row_offset = c_offset + ih * width;
                                            for kx in 0..k_w {
                                                let iw = w_start + kx;
                                                in_grad_slice[row_offset + iw] += grad_scaled;
                                            }
                                        }
                                        out_idx += 1;
                                    }
                                }
                            }
                        }
                    });
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
