use NeuralFlow::prelude::*;
use NeuralFlow::tensor::im2col::{col2im, im2col};

#[test]
fn test_im2col_col2im_consistency() {
    let channels = 1;
    let height = 3;
    let width = 3;
    let k_h = 2;
    let k_w = 2;
    let pad_h = 0;
    let pad_w = 0;
    let stride_h = 1;
    let stride_w = 1;
    let out_h = 2;
    let out_w = 2;

    let img = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];

    let mut col = vec![0.0; channels * k_h * k_w * out_h * out_w];
    im2col(
        &img,
        channels,
        height,
        width,
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

    // col matrix has shape (channels * kh * kw, out_h * out_w) = (4, 4)
    // Row 0: ky=0, kx=0 -> [1, 2, 4, 5]
    // Row 1: ky=0, kx=1 -> [2, 3, 5, 6]
    // Row 2: ky=1, kx=0 -> [4, 5, 7, 8]
    // Row 3: ky=1, kx=1 -> [5, 6, 8, 9]
    assert_eq!(&col[0..4], &[1.0, 2.0, 4.0, 5.0]);
    assert_eq!(&col[4..8], &[2.0, 3.0, 5.0, 6.0]);
    assert_eq!(&col[8..12], &[4.0, 5.0, 7.0, 8.0]);
    assert_eq!(&col[12..16], &[5.0, 6.0, 8.0, 9.0]);

    // Now test col2im with unit gradient (all 1.0 in col)
    let ones_col = vec![1.0; 16];
    let mut reconstructed = vec![0.0; 9];
    col2im(
        &ones_col,
        channels,
        height,
        width,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        out_h,
        out_w,
        &mut reconstructed,
    );

    // Each position counts how many 2x2 windows cover it:
    // corners (0, 2, 6, 8) -> 1 window
    // edges (1, 3, 5, 7) -> 2 windows
    // center (4) -> 4 windows
    let expected = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    assert_eq!(reconstructed, expected);
}

#[test]
fn test_conv2d_forward_analytical() {
    let conv = Conv2d::new_with_options(1, 1, (2, 2), (1, 1), (0, 0), false);
    // Set weights to known identity-like kernel: [[1.0, 0.0], [0.0, 1.0]]
    {
        let mut w_inner = conv.weights.0.borrow_mut();
        w_inner.data = vec![1.0, 0.0, 0.0, 1.0];
    }

    let input_data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let input = Tensor::new_4d(input_data, (1, 1, 3, 3));
    let out = conv.forward(&input);

    assert_eq!(out.shape4d(), Some((1, 1, 2, 2)));

    let out_data = out.0.borrow().data.clone();
    // (0,0): 1*1 + 5*1 = 6
    // (0,1): 2*1 + 6*1 = 8
    // (1,0): 4*1 + 8*1 = 12
    // (1,1): 5*1 + 9*1 = 14
    assert_eq!(out_data, vec![6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn test_conv2d_autograd() {
    let conv = Conv2d::new_with_options(1, 1, (2, 2), (1, 1), (0, 0), true);
    {
        let mut w_inner = conv.weights.0.borrow_mut();
        w_inner.data = vec![0.5, -0.5, 1.0, 0.0];
    }
    if let Some(ref b) = conv.bias {
        let mut b_inner = b.0.borrow_mut();
        b_inner.data = vec![0.1];
    }

    let input_data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let input = Tensor::new_4d(input_data, (1, 1, 3, 3));
    let target = Tensor::new_4d(vec![1.0, 2.0, 3.0, 4.0], (1, 1, 2, 2));

    let out = conv.forward(&input);
    let loss = mse_loss(&out, &target);
    loss.backward();

    let w_grad = conv.weights.0.borrow().grad.clone();
    let in_grad = input.0.borrow().grad.clone();

    // Verify gradients are non-empty, non-zero and contain no NaNs/Infs
    assert_eq!(w_grad.len(), 4);
    assert!(w_grad.iter().all(|&g| !g.is_nan() && !g.is_infinite()));
    assert!(w_grad.iter().any(|&g| g.abs() > 1e-4));

    assert_eq!(in_grad.len(), 9);
    assert!(in_grad.iter().all(|&g| !g.is_nan() && !g.is_infinite()));
    assert!(in_grad.iter().any(|&g| g.abs() > 1e-4));

    if let Some(ref b) = conv.bias {
        let b_grad = b.0.borrow().grad.clone();
        assert_eq!(b_grad.len(), 1);
        assert!(!b_grad[0].is_nan() && b_grad[0].abs() > 1e-4);
    }
}

#[test]
fn test_maxpool2d_forward_and_backward() {
    let pool = MaxPool2d::new((2, 2));

    let input_data = vec![
        1.0, 4.0,  2.0, 3.0,
        3.0, 2.0,  0.0, 1.0,

        8.0, 5.0,  7.0, 6.0,
        6.0, 7.0,  5.0, 9.0,
    ];
    let input = Tensor::new_4d(input_data, (1, 1, 4, 4));
    let out = pool.forward(&input);

    assert_eq!(out.shape4d(), Some((1, 1, 2, 2)));
    let out_data = out.0.borrow().data.clone();
    // Max of top-left (1,4,3,2) -> 4
    // Max of top-right (2,3,0,1) -> 3
    // Max of bot-left (8,5,6,7) -> 8
    // Max of bot-right (7,6,5,9) -> 9
    assert_eq!(out_data, vec![4.0, 3.0, 8.0, 9.0]);

    // Backward pass
    let target = Tensor::new_4d(vec![0.0, 0.0, 0.0, 0.0], (1, 1, 2, 2));
    let loss = mse_loss(&out, &target);
    loss.backward();

    let in_grad = input.0.borrow().grad.clone();
    // Only the argmax positions should receive gradients:
    // (0,1)=4.0 -> index 1
    // (0,3)=3.0 -> index 3
    // (2,0)=8.0 -> index 8
    // (3,3)=9.0 -> index 15
    assert!(in_grad[1] > 0.0);
    assert!(in_grad[3] > 0.0);
    assert!(in_grad[8] > 0.0);
    assert!(in_grad[15] > 0.0);

    // Non-max positions should have zero gradient:
    assert_eq!(in_grad[0], 0.0);
    assert_eq!(in_grad[2], 0.0);
    assert_eq!(in_grad[4], 0.0);
}

#[test]
fn test_avgpool2d_forward_and_backward() {
    let pool = AvgPool2d::new((2, 2));

    let input_data = vec![
        1.0, 3.0,  2.0, 4.0,
        1.0, 3.0,  2.0, 4.0,

        4.0, 4.0,  0.0, 8.0,
        4.0, 4.0,  0.0, 8.0,
    ];
    let input = Tensor::new_4d(input_data, (1, 1, 4, 4));
    let out = pool.forward(&input);

    assert_eq!(out.shape4d(), Some((1, 1, 2, 2)));
    let out_data = out.0.borrow().data.clone();
    // Top-left: (1+3+1+3)/4 = 2.0
    // Top-right: (2+4+2+4)/4 = 3.0
    // Bot-left: (4+4+4+4)/4 = 4.0
    // Bot-right: (0+8+0+8)/4 = 4.0
    assert_eq!(out_data, vec![2.0, 3.0, 4.0, 4.0]);

    let target = Tensor::new_4d(vec![0.0, 0.0, 0.0, 0.0], (1, 1, 2, 2));
    let loss = mse_loss(&out, &target);
    loss.backward();

    let in_grad = input.0.borrow().grad.clone();
    // In avgpool, all elements within a window share equal gradients
    assert!((in_grad[0] - in_grad[1]).abs() < 1e-6);
    assert!((in_grad[0] - in_grad[4]).abs() < 1e-6);
    assert!((in_grad[0] - in_grad[5]).abs() < 1e-6);
}

#[test]
fn test_flatten_layer() {
    let flatten = Flatten::new();
    let data = vec![1.0; 2 * 3 * 4 * 4]; // (2, 3, 4, 4)
    let input = Tensor::new_4d(data, (2, 3, 4, 4));

    let out = flatten.forward(&input);
    assert_eq!(out.shape(), (2, 48));
    assert_eq!(out.shape4d(), None);

    let loss = mse_loss(&out, &Tensor::zeros((2, 48)));
    loss.backward();

    let in_grad = input.0.borrow().grad.clone();
    assert_eq!(in_grad.len(), 2 * 3 * 4 * 4);
    assert!(in_grad.iter().all(|&g| (g - 2.0 / 96.0).abs() < 1e-5));
}

#[test]
fn test_cnn_end_to_end_training() {
    struct SimpleCNN {
        conv: Conv2d,
        pool: MaxPool2d,
        flatten: Flatten,
        fc: Linear,
    }

    impl SimpleCNN {
        fn new() -> Self {
            Self {
                // Input: (N, 1, 6, 6) -> Conv (1, 2, (3, 3), pad=(1,1)) -> (N, 2, 6, 6)
                conv: Conv2d::new_with_options(1, 2, (3, 3), (1, 1), (1, 1), true),
                // Pool (2, 2) -> (N, 2, 3, 3)
                pool: MaxPool2d::new((2, 2)),
                flatten: Flatten::new(),
                // Flatten: 2 * 3 * 3 = 18 features -> 2 classes
                fc: Linear::new(18, 2),
            }
        }
    }

    impl Module for SimpleCNN {
        fn forward(&self, x: &Tensor) -> Tensor {
            let x = self.conv.forward(x).relu();
            let x = self.pool.forward(&x);
            let x = self.flatten.forward(&x);
            self.fc.forward(&x)
        }

        fn parameters(&self) -> Vec<Tensor> {
            let mut p = self.conv.parameters();
            p.extend(self.fc.parameters());
            p
        }
    }

    let model = SimpleCNN::new();
    let mut optimizer = Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0);

    // 4 training images (6x6):
    // Class 0: horizontal stripe patterns
    // Class 1: vertical stripe patterns
    let mut img0 = vec![0.0; 36];
    for col in 0..6 {
        img0[2 * 6 + col] = 1.0;
        img0[3 * 6 + col] = 1.0;
    }

    let mut img1 = vec![0.0; 36];
    for col in 0..6 {
        img1[1 * 6 + col] = 1.0;
        img1[4 * 6 + col] = 1.0;
    }

    let mut img2 = vec![0.0; 36];
    for row in 0..6 {
        img2[row * 6 + 2] = 1.0;
        img2[row * 6 + 3] = 1.0;
    }

    let mut img3 = vec![0.0; 36];
    for row in 0..6 {
        img3[row * 6 + 1] = 1.0;
        img3[row * 6 + 4] = 1.0;
    }

    let mut all_data = Vec::new();
    all_data.extend(img0);
    all_data.extend(img1);
    all_data.extend(img2);
    all_data.extend(img3);

    let inputs = Tensor::new_4d(all_data, (4, 1, 6, 6));
    let targets = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], (4, 1));

    let initial_loss = cross_entropy_loss(&model.forward(&inputs), &targets)
        .0
        .borrow()
        .data[0];

    for _ in 0..50 {
        model.zero_grad();
        let pred = model.forward(&inputs);
        let loss = cross_entropy_loss(&pred, &targets);
        loss.backward();
        optimizer.step(&model.parameters());
    }

    let final_pred = model.forward(&inputs);
    let final_loss = cross_entropy_loss(&final_pred, &targets).0.borrow().data[0];

    assert!(
        final_loss < initial_loss,
        "CNN training should decrease loss: initial={}, final={}",
        initial_loss,
        final_loss
    );

    let pred_data = final_pred.0.borrow().data.clone();
    let target_data = targets.0.borrow().data.clone();
    let acc = Metrics::accuracy_logits(&pred_data, &target_data, 2);
    assert_eq!(acc, 1.0, "CNN should achieve 100% accuracy on pattern classification");
}

#[test]
fn test_sequential_module() {
    let seq = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLu),
        Box::new(Dropout::new(0.5)),
        Box::new(Linear::new(4, 1)),
    ]);

    assert_eq!(seq.parameters().len(), 2);

    let x = Tensor::new(vec![1.0, 2.0], (1, 2));

    // Training mode
    seq.train();
    assert!(seq.is_training());
    let _out_train = seq.forward(&x);

    // Eval mode
    seq.eval();
    assert!(!seq.is_training());
    let out_eval = seq.forward(&x);
    assert_eq!(out_eval.shape(), (1, 1));
}

