use NeuralFlow::prelude::*;

#[test]
fn test_no_grad_guard() {
    assert!(is_grad_enabled());

    let a = Tensor::new(vec![2.0, 3.0], (1, 2));
    let relu_out = a.relu();
    assert_eq!(relu_out.0.borrow().prev.len(), 1);
    assert!(relu_out.0.borrow().backward.is_some());

    {
        let _guard = no_grad();
        assert!(!is_grad_enabled());

        let b = Tensor::new(vec![2.0, 3.0], (1, 2));
        let relu_b = b.relu();
        assert_eq!(relu_b.0.borrow().prev.len(), 0);
        assert!(relu_b.0.borrow().backward.is_none());

        let matmul_b = b.matmul(&Tensor::new(vec![1.0, 2.0], (2, 1)));
        assert_eq!(matmul_b.0.borrow().prev.len(), 0);
        assert!(matmul_b.0.borrow().backward.is_none());
    }

    assert!(is_grad_enabled());
}

#[test]
fn test_matmul_and_loss_autograd() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    let w = Tensor::new(vec![0.5, -0.5, 1.0, 0.0], (2, 2));
    let target = Tensor::new(vec![1.0, 0.0, 2.0, 1.0], (2, 2));

    let out = x.matmul(&w);
    let loss = mse_loss(&out, &target);
    loss.backward();

    assert!(w.0.borrow().grad.iter().all(|&g| !g.is_nan()));
    assert!(x.0.borrow().grad.iter().all(|&g| !g.is_nan()));
}

#[test]
fn test_activations_autograd() {
    let x = Tensor::new(vec![-2.0, 0.5, 3.0, -1.0], (2, 2));
    let relu_out = x.relu();
    let loss = mse_loss(&relu_out, &Tensor::zeros((2, 2)));
    loss.backward();

    let grad = &x.0.borrow().grad;
    assert_eq!(grad[0], 0.0);
    assert!(grad[1] > 0.0);
    assert!(grad[2] > 0.0);
    assert_eq!(grad[3], 0.0);
}

#[test]
fn test_flat_dataset_and_dataloader() {
    let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
    let targets = vec![vec![10.0], vec![20.0], vec![30.0], vec![40.0]];

    let dataset = TensorDataset::new(inputs, targets);
    assert_eq!(dataset.len(), 4);

    let (x0, y0) = dataset.get(0);
    assert_eq!(x0, vec![1.0, 2.0]);
    assert_eq!(y0, vec![10.0]);

    let mut loader = Dataloader::new(&dataset, 2, false);
    let mut batch_count = 0;
    for (x_batch, y_batch) in loader.iter_batches() {
        assert_eq!(x_batch.0.borrow().shape, (2, 2));
        assert_eq!(y_batch.0.borrow().shape, (2, 1));
        batch_count += 1;
    }
    assert_eq!(batch_count, 2);
}

#[test]
fn test_sgd_step() {
    let w = Tensor::new(vec![1.0, 2.0, 3.0], (1, 3));
    {
        let mut inner = w.0.borrow_mut();
        inner.grad = vec![0.1, 0.2, 0.3];
    }
    let mut optimizer = SGD::new(0.5);
    optimizer.step(&[w.clone()]);

    let inner = w.0.borrow();
    assert!((inner.data[0] - 0.95).abs() < 1e-5);
    assert!((inner.data[1] - 1.90).abs() < 1e-5);
    assert!((inner.data[2] - 2.85).abs() < 1e-5);
}

#[test]
fn test_gemm_rectangular_correctness() {
    // A: 3x2, B: 2x4 -> C: 3x4
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.5, -1.0, 2.0, 0.0, 1.5, 0.5, -0.5, 3.0];
    let a = Tensor::new(a_data, (3, 2));
    let b = Tensor::new(b_data, (2, 4));

    let c = a.matmul(&b);
    {
        let c_inner = c.0.borrow();
        assert_eq!(c_inner.shape, (3, 4));

        // Expected row 0: [1*0.5 + 2*1.5, 1*-1 + 2*0.5, 1*2 + 2*-0.5, 1*0 + 2*3] = [3.5, 0.0, 1.0, 6.0]
        assert!((c_inner.data[0] - 3.5).abs() < 1e-5);
        assert!((c_inner.data[1] - 0.0).abs() < 1e-5);
        assert!((c_inner.data[2] - 1.0).abs() < 1e-5);
        assert!((c_inner.data[3] - 6.0).abs() < 1e-5);
    }

    let loss = mse_loss(&c, &Tensor::zeros((3, 4)));
    loss.backward();

    assert!(a.0.borrow().grad.iter().all(|&g| !g.is_nan()));
    assert!(b.0.borrow().grad.iter().all(|&g| !g.is_nan()));
}


