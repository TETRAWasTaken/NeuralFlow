use super::module::Module;
use crate::tensor::{is_grad_enabled, Tensor};

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

        let in_inner = input.0.borrow();
        let in_data = &in_inner.data;

        let total_out = n_batch * channels * out_spatial;
        let mut out_data = Vec::with_capacity(total_out);
        let mut max_indices = if is_grad_enabled() {
            Some(Vec::with_capacity(total_out))
        } else {
            None
        };

        for n in 0..n_batch {
            let n_offset = n * channels * in_spatial;
            for c in 0..channels {
                let c_offset = n_offset + c * in_spatial;
                for oh in 0..out_h {
                    let h_start = oh * s_h;
                    for ow in 0..out_w {
                        let w_start = ow * s_w;

                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for ky in 0..k_h {
                            let ih = h_start + ky;
                            let row_offset = c_offset + ih * width;
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

                        out_data.push(max_val);
                        if let Some(ref mut indices) = max_indices {
                            indices.push(max_idx);
                        }
                    }
                }
            }
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

                for (i, &max_idx) in saved_indices.iter().enumerate() {
                    in_grad[max_idx] += out_grad[i];
                }
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
        let pool_area = (k_h * k_w) as f32;

        let in_inner = input.0.borrow();
        let in_data = &in_inner.data;

        let total_out = n_batch * channels * out_spatial;
        let mut out_data = Vec::with_capacity(total_out);

        for n in 0..n_batch {
            let n_offset = n * channels * in_spatial;
            for c in 0..channels {
                let c_offset = n_offset + c * in_spatial;
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
                        out_data.push(sum / pool_area);
                    }
                }
            }
        }
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

                let mut out_idx = 0;
                for n in 0..n_batch {
                    let n_offset = n * channels * in_spatial;
                    for c in 0..channels {
                        let c_offset = n_offset + c * in_spatial;
                        for oh in 0..out_h {
                            let h_start = oh * s_h;
                            for ow in 0..out_w {
                                let w_start = ow * s_w;
                                let grad_scaled = out_grad[out_idx] / pool_area;

                                for ky in 0..k_h {
                                    let ih = h_start + ky;
                                    let row_offset = c_offset + ih * width;
                                    for kx in 0..k_w {
                                        let iw = w_start + kx;
                                        in_grad[row_offset + iw] += grad_scaled;
                                    }
                                }
                                out_idx += 1;
                            }
                        }
                    }
                }
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
