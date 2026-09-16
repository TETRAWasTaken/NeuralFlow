use NeuralFlow::prelude::*;

#[test]
fn test_cross_entropy_multiclass_class_indices() {
    // 2 samples, 3 classes
    let logits = Tensor::new(vec![1.0, 2.0, 0.0, 0.5, -1.0, 3.0], (2, 3));
    let targets = Tensor::new(vec![1.0, 2.0], (2, 1));

    let loss = cross_entropy_loss(&logits, &targets);
    let loss_val = loss.0.borrow().data[0];

    // Reference from PyTorch: F.cross_entropy(z, target) == 0.25164017
    assert!((loss_val - 0.25164017).abs() < 1e-5, "Expected ~0.25164, got {}", loss_val);

    loss.backward();

    let grad = logits.0.borrow().grad.clone();
    let expected_grad = vec![
        0.1223641, -0.1673796, 0.0450155,
        0.0373024, 0.0083226, -0.0456250,
    ];

    for i in 0..expected_grad.len() {
        assert!(
            (grad[i] - expected_grad[i]).abs() < 1e-4,
            "At index {}: expected {}, got {}",
            i,
            expected_grad[i],
            grad[i]
        );
    }
}

#[test]
fn test_cross_entropy_multiclass_one_hot() {
    let logits_indices = Tensor::new(vec![1.0, 2.0, 0.0, 0.5, -1.0, 3.0], (2, 3));
    let targets_indices = Tensor::new(vec![1.0, 2.0], (2, 1));
    let loss_indices = cross_entropy_loss(&logits_indices, &targets_indices);
    loss_indices.backward();

    let logits_onehot = Tensor::new(vec![1.0, 2.0, 0.0, 0.5, -1.0, 3.0], (2, 3));
    let targets_onehot = Tensor::new(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], (2, 3));
    let loss_onehot = cross_entropy_loss(&logits_onehot, &targets_onehot);
    loss_onehot.backward();

    let loss_val_idx = loss_indices.0.borrow().data[0];
    let loss_val_oh = loss_onehot.0.borrow().data[0];
    assert!((loss_val_idx - loss_val_oh).abs() < 1e-6);

    let grad_idx = &logits_indices.0.borrow().grad;
    let grad_oh = &logits_onehot.0.borrow().grad;
    for i in 0..grad_idx.len() {
        assert!((grad_idx[i] - grad_oh[i]).abs() < 1e-6);
    }
}

#[test]
fn test_cross_entropy_soft_labels() {
    let logits = Tensor::new(vec![1.0, 2.0, 0.0], (1, 3));
    let targets = Tensor::new(vec![0.1, 0.8, 0.1], (1, 3));

    let loss = cross_entropy_loss(&logits, &targets);
    let loss_val = loss.0.borrow().data[0];

    // Reference from PyTorch: 0.70760596
    assert!((loss_val - 0.70760596).abs() < 1e-5, "Got {}", loss_val);

    loss.backward();
    let grad = logits.0.borrow().grad.clone();
    let expected_grad = vec![0.144728, -0.134759, -0.009969];

    for i in 0..expected_grad.len() {
        assert!(
            (grad[i] - expected_grad[i]).abs() < 1e-4,
            "At index {}: expected {}, got {}",
            i,
            expected_grad[i],
            grad[i]
        );
    }
}

#[test]
fn test_cross_entropy_numerical_stability() {
    // Extreme logits that would overflow exp(1000.0) without log-sum-exp trick
    let logits = Tensor::new(vec![1000.0, 999.0, -1000.0], (1, 3));
    let targets = Tensor::new(vec![0.0], (1, 1));

    let loss = cross_entropy_loss(&logits, &targets);
    let loss_val = loss.0.borrow().data[0];

    assert!(!loss_val.is_nan() && !loss_val.is_infinite());
    // log(1 + e^-1 + e^-2000) ~ ln(1 + 0.367879) ~ 0.31326
    assert!((loss_val - 0.31326168).abs() < 1e-4);

    loss.backward();
    let grad = &logits.0.borrow().grad;
    for &g in grad.iter() {
        assert!(!g.is_nan() && !g.is_infinite());
    }
}

#[test]
fn test_cross_entropy_binary_classification() {
    let logits = Tensor::new(vec![1.5, -2.0], (2, 1));
    let targets = Tensor::new(vec![1.0, 0.0], (2, 1));

    let loss = cross_entropy_loss(&logits, &targets);
    let loss_val = loss.0.borrow().data[0];

    // PyTorch reference: F.binary_cross_entropy_with_logits is 0.16417068
    assert!((loss_val - 0.16417068).abs() < 1e-5, "Got {}", loss_val);

    loss.backward();
    let grad = &logits.0.borrow().grad;
    // Expected grads: (sigmoid(1.5) - 1.0)/2 ~ -0.09117, (sigmoid(-2.0) - 0.0)/2 ~ 0.059699
    assert!((grad[0] - (-0.091170)).abs() < 1e-4);
    assert!((grad[1] - 0.059699).abs() < 1e-4);
}

#[test]
fn test_bce_loss_probabilities() {
    let probs = Tensor::new(vec![0.8, 0.1], (2, 1));
    let targets = Tensor::new(vec![1.0, 0.0], (2, 1));

    let loss = bce_loss(&probs, &targets);
    let loss_val = loss.0.borrow().data[0];

    // Loss = -0.5 * (ln(0.8) + ln(0.9)) = -0.5 * (-0.223143 - 0.10536) = 0.16425
    assert!((loss_val - 0.16425).abs() < 1e-4);

    loss.backward();
    let grad = &probs.0.borrow().grad;
    assert!(grad.iter().all(|&g| !g.is_nan() && !g.is_infinite()));
}

#[test]
fn test_accuracy_metrics() {
    let preds = vec![0.9, 0.1, 0.8, 0.3];
    let targets = vec![1.0, 0.0, 1.0, 1.0];
    // rounded preds: [1, 0, 1, 0] vs targets [1, 0, 1, 1] -> 3/4 = 0.75
    let acc = Metrics::accuracy(&preds, &targets);
    assert!((acc - 0.75).abs() < 1e-5);

    // Multi-class logits: 3 samples, 3 classes
    let logits = vec![
        1.0, 5.0, 2.0, // argmax = 1
        0.0, -1.0, 3.0, // argmax = 2
        4.0, 2.0, 1.0, // argmax = 0
    ];
    let class_targets = vec![1.0, 2.0, 2.0]; // sample 2 target is class 2, but argmax is 0
    // 2 out of 3 correct = 0.66667
    let multi_acc = Metrics::accuracy_logits(&logits, &class_targets, 3);
    assert!((multi_acc - (2.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_classification_training_synthetic() {
    struct Classifier {
        fc1: Linear,
        fc2: Linear,
    }

    impl Classifier {
        fn new() -> Self {
            Self {
                fc1: Linear::new(2, 8),
                fc2: Linear::new(8, 3),
            }
        }
    }

    impl Module for Classifier {
        fn forward(&self, x: &Tensor) -> Tensor {
            self.fc2.forward(&self.fc1.forward(x).relu())
        }

        fn parameters(&self) -> Vec<Tensor> {
            let mut p = self.fc1.parameters();
            p.extend(self.fc2.parameters());
            p
        }
    }

    let model = Classifier::new();
    let mut optimizer = Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0);

    // 6 samples across 3 distinct 2D clusters:
    // Class 0: (1.0, 1.0), (1.2, 0.9)
    // Class 1: (-1.0, 1.0), (-1.1, 0.8)
    // Class 2: (0.0, -1.5), (-0.2, -1.3)
    let inputs = Tensor::new(
        vec![
            1.0, 1.0,
            1.2, 0.9,
            -1.0, 1.0,
            -1.1, 0.8,
            0.0, -1.5,
            -0.2, -1.3,
        ],
        (6, 2),
    );
    let targets = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], (6, 1));

    let initial_loss = cross_entropy_loss(&model.forward(&inputs), &targets)
        .0
        .borrow()
        .data[0];

    for _ in 0..100 {
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
        "Loss should decrease: initial={}, final={}",
        initial_loss,
        final_loss
    );

    let pred_data = final_pred.0.borrow().data.clone();
    let target_data = targets.0.borrow().data.clone();
    let acc = Metrics::accuracy_logits(&pred_data, &target_data, 3);
    assert_eq!(acc, 1.0, "Model should achieve 100% accuracy on training clusters");
}

#[test]
fn test_cross_entropy_parallel_execution() {
    // 35,000 samples (> 32,768 PARALLEL_THRESHOLD), 4 classes
    let n_samples = 35_000;
    let n_classes = 4;
    let mut logits_data = Vec::with_capacity(n_samples * n_classes);
    let mut targets_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class_idx = (i % n_classes) as f32;
        targets_data.push(class_idx);
        for c in 0..n_classes {
            logits_data.push(((i + c) % 5) as f32 * 0.1);
        }
    }

    let logits = Tensor::new(logits_data, (n_samples, n_classes));
    let targets = Tensor::new(targets_data, (n_samples, 1));

    let loss = cross_entropy_loss(&logits, &targets);
    let loss_val = loss.0.borrow().data[0];

    assert!(!loss_val.is_nan() && !loss_val.is_infinite());
    assert!(loss_val > 0.0);

    loss.backward();
    let grad = &logits.0.borrow().grad;
    assert_eq!(grad.len(), n_samples * n_classes);
    assert!(grad.iter().all(|&g| !g.is_nan() && !g.is_infinite()));
}

#[test]
fn test_kaiming_normal_statistics() {
    let fan_in = 100;
    let out_features = 500;
    let t = Tensor::kaiming_normal((fan_in, out_features));
    let inner = t.0.borrow();
    assert_eq!(inner.shape, (fan_in, out_features));

    let n = (fan_in * out_features) as f32;
    let mean: f32 = inner.data.iter().sum::<f32>() / n;

    let var: f32 = inner
        .data
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / n;

    // Expected: Mean ~ 0.0, Variance ~ 2.0 / fan_in = 0.02
    let expected_var = 2.0 / (fan_in as f32);
    assert!(
        mean.abs() < 0.015,
        "Mean should be close to 0.0, got {}",
        mean
    );
    assert!(
        (var - expected_var).abs() < 0.005,
        "Variance should be close to {}, got {}",
        expected_var,
        var
    );
}


