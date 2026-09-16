/// im2col and col2im implementations for 2D convolutions.
///
/// Layout:
/// - Image: (C, H, W) in row-major order.
/// - Column matrix: (C * Kh * Kw, H_out * W_out) in row-major order.

pub fn im2col(
    data_im: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    out_h: usize,
    out_w: usize,
    data_col: &mut [f32],
) {
    let out_spatial = out_h * out_w;
    let hw = height * width;
    let kh_kw = k_h * k_w;

    let mut col_row = 0;
    for c in 0..channels {
        let im_c_offset = c * hw;
        for ky in 0..k_h {
            for kx in 0..k_w {
                let col_row_offset = col_row * out_spatial;
                for oh in 0..out_h {
                    let ih = (oh * stride_h) as isize - pad_h as isize + ky as isize;
                    let col_oh_offset = col_row_offset + oh * out_w;

                    if ih >= 0 && ih < height as isize {
                        let im_row_offset = im_c_offset + (ih as usize) * width;
                        for ow in 0..out_w {
                            let iw = (ow * stride_w) as isize - pad_w as isize + kx as isize;
                            if iw >= 0 && iw < width as isize {
                                data_col[col_oh_offset + ow] = data_im[im_row_offset + iw as usize];
                            } else {
                                data_col[col_oh_offset + ow] = 0.0;
                            }
                        }
                    } else {
                        for ow in 0..out_w {
                            data_col[col_oh_offset + ow] = 0.0;
                        }
                    }
                }
                col_row += 1;
            }
        }
    }
    debug_assert_eq!(col_row, channels * kh_kw);
}

pub fn col2im(
    data_col: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    out_h: usize,
    out_w: usize,
    data_im: &mut [f32],
) {
    let out_spatial = out_h * out_w;
    let hw = height * width;

    let mut col_row = 0;
    for c in 0..channels {
        let im_c_offset = c * hw;
        for ky in 0..k_h {
            for kx in 0..k_w {
                let col_row_offset = col_row * out_spatial;
                for oh in 0..out_h {
                    let ih = (oh * stride_h) as isize - pad_h as isize + ky as isize;
                    let col_oh_offset = col_row_offset + oh * out_w;

                    if ih >= 0 && ih < height as isize {
                        let im_row_offset = im_c_offset + (ih as usize) * width;
                        for ow in 0..out_w {
                            let iw = (ow * stride_w) as isize - pad_w as isize + kx as isize;
                            if iw >= 0 && iw < width as isize {
                                data_im[im_row_offset + iw as usize] += data_col[col_oh_offset + ow];
                            }
                        }
                    }
                }
                col_row += 1;
            }
        }
    }
}
