use super::module::Module;
use crate::tensor::im2col::{col2im, im2col};
use crate::tensor::{is_grad_enabled, Tensor};

pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub weights: Tensor,
    pub bias: Option<Tensor>,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Self {
        Self::new_with_options(in_channels, out_channels, kernel_size, (1, 1), (0, 0), true)
    }

    pub fn new_with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
    ) -> Self {
        assert!(in_channels > 0 && out_channels > 0);
        assert!(kernel_size.0 > 0 && kernel_size.1 > 0);
        assert!(stride.0 > 0 && stride.1 > 0);

        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let total_weights = out_channels * fan_in;

        // Kaiming Normal initialization for conv weights
        let mut w_data = Vec::with_capacity(total_weights);
        while w_data.len() < total_weights {
            let u1 = rand::random::<f32>().max(f32::EPSILON);
            let u2 = rand::random::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            w_data.push(r * theta.cos() * std_dev);
            if w_data.len() < total_weights {
                w_data.push(r * theta.sin() * std_dev);
            }
        }

        let weights = Tensor::new(w_data, (out_channels, fan_in));
        let bias = if use_bias {
            Some(Tensor::zeros((1, out_channels)))
        } else {
            None
        };

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (n_batch, c_in, h_in, w_in) = input
            .shape4d()
            .expect("Input to Conv2d must be a 4D tensor with shape (N, C, H, W)");

        assert_eq!(
            c_in, self.in_channels,
            "Input channel count ({}) does not match Conv2d in_channels ({})",
            c_in, self.in_channels
        );

        let (k_h, k_w) = self.kernel_size;
        let (pad_h, pad_w) = self.padding;
        let (stride_h, stride_w) = self.stride;

        assert!(
            h_in + 2 * pad_h >= k_h && w_in + 2 * pad_w >= k_w,
            "Kernel size ({}, {}) is larger than padded input spatial size ({}, {})",
            k_h, k_w, h_in + 2 * pad_h, w_in + 2 * pad_w
        );

        let out_h = (h_in + 2 * pad_h - k_h) / stride_h + 1;
        let out_w = (w_in + 2 * pad_w - k_w) / stride_w + 1;
        let out_spatial = out_h * out_w;
        let col_rows = self.in_channels * k_h * k_w;

        let in_image_size = c_in * h_in * w_in;
        let out_image_size = self.out_channels * out_spatial;
        let mut out_data = vec![0.0; n_batch * out_image_size];

        let in_inner = input.0.borrow();
        let in_data = &in_inner.data;
        let w_inner = self.weights.0.borrow();
        let w_data = &w_inner.data;

        let mut all_cols = if is_grad_enabled() {
            Some(Vec::with_capacity(n_batch * col_rows * out_spatial))
        } else {
            None
        };

        let mut col = vec![0.0; col_rows * out_spatial];

        for n in 0..n_batch {
            let im_slice = &in_data[n * in_image_size..(n + 1) * in_image_size];
            im2col(
                im_slice,
                c_in,
                h_in,
                w_in,
                k_h,
                k_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                out_h,
                out_w,
                &mut col,
            );

            let out_slice =
                &mut out_data[n * out_image_size..(n + 1) * out_image_size];

            unsafe {
                crate::tensor::blas::gemm(
                    false,
                    false,
                    self.out_channels,
                    out_spatial,
                    col_rows,
                    1.0,
                    w_data.as_ptr(),
                    col.as_ptr(),
                    0.0,
                    out_slice.as_mut_ptr(),
                );
            }

            if let Some(ref mut cols_buf) = all_cols {
                cols_buf.extend_from_slice(&col);
            }
        }

        drop(w_inner);
        drop(in_inner);

        // Add bias if present
        if let Some(ref bias_tensor) = self.bias {
            let b_inner = bias_tensor.0.borrow();
            let b_data = &b_inner.data;
            for n in 0..n_batch {
                let out_offset = n * out_image_size;
                for c in 0..self.out_channels {
                    let b_val = b_data[c];
                    let ch_offset = out_offset + c * out_spatial;
                    for s in 0..out_spatial {
                        out_data[ch_offset + s] += b_val;
                    }
                }
            }
        }

        let out = Tensor::new_4d(out_data, (n_batch, self.out_channels, out_h, out_w));

        if is_grad_enabled() {
            let mut prev = vec![input.clone(), self.weights.clone()];
            if let Some(ref b) = self.bias {
                prev.push(b.clone());
            }
            out.0.borrow_mut().prev = prev;

            let input_clone = input.clone();
            let weights_clone = self.weights.clone();
            let bias_clone = self.bias.clone();
            let out_clone = out.clone();
            let saved_cols = all_cols.unwrap();

            let out_channels = self.out_channels;
            let in_channels = self.in_channels;

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;

                // 1. Bias gradient
                if let Some(ref b) = bias_clone {
                    let mut b_inner = b.0.borrow_mut();
                    let b_grad = &mut b_inner.grad;
                    for n in 0..n_batch {
                        let out_offset = n * out_image_size;
                        for c in 0..out_channels {
                            let ch_offset = out_offset + c * out_spatial;
                            let sum: f32 = out_grad[ch_offset..ch_offset + out_spatial]
                                .iter()
                                .sum();
                            b_grad[c] += sum;
                        }
                    }
                }

                // 2. Weights gradient
                let col_size = col_rows * out_spatial;
                {
                    let mut w_inner = weights_clone.0.borrow_mut();
                    for n in 0..n_batch {
                        let out_grad_slice =
                            &out_grad[n * out_image_size..(n + 1) * out_image_size];
                        let col_slice = &saved_cols[n * col_size..(n + 1) * col_size];

                        unsafe {
                            crate::tensor::blas::gemm(
                                false,
                                true, // Col transposed: (col_rows, out_spatial) -> (out_spatial, col_rows)
                                out_channels,
                                col_rows,
                                out_spatial,
                                1.0,
                                out_grad_slice.as_ptr(),
                                col_slice.as_ptr(),
                                1.0, // Accumulate directly into w_inner.grad
                                w_inner.grad.as_mut_ptr(),
                            );
                        }
                    }
                }

                // 3. Input gradient
                {
                    let w_inner = weights_clone.0.borrow();
                    let w_data = &w_inner.data;
                    let mut in_inner = input_clone.0.borrow_mut();
                    let mut d_col = vec![0.0; col_size];

                    for n in 0..n_batch {
                        let out_grad_slice =
                            &out_grad[n * out_image_size..(n + 1) * out_image_size];

                        unsafe {
                            crate::tensor::blas::gemm(
                                true, // W transposed: (out_channels, col_rows) -> (col_rows, out_channels)
                                false,
                                col_rows,
                                out_spatial,
                                out_channels,
                                1.0,
                                w_data.as_ptr(),
                                out_grad_slice.as_ptr(),
                                0.0,
                                d_col.as_mut_ptr(),
                            );
                        }

                        let in_grad_slice =
                            &mut in_inner.grad[n * in_image_size..(n + 1) * in_image_size];
                        col2im(
                            &d_col,
                            in_channels,
                            h_in,
                            w_in,
                            k_h,
                            k_w,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
                            out_h,
                            out_w,
                            in_grad_slice,
                        );
                    }
                }
            }));
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
