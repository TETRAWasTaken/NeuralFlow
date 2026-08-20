use NeuralFlow::prelude::*;
use std::collections::HashMap;

pub struct DeepNet {
    fc1: Linear,
    ln1: LayerNorm,
    drop1: Dropout,
    fc2: Linear,
    ln2: LayerNorm,
    drop2: Dropout,
    fc3: Linear,
    ln3: LayerNorm,
    drop3: Dropout,
    fc4: Linear,
}

impl DeepNet {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        Self {
            fc1: Linear::new(in_dim, hidden_dim),
            ln1: LayerNorm::new(hidden_dim),
            drop1: Dropout::new(0.2),
            fc2: Linear::new(hidden_dim, hidden_dim),
            ln2: LayerNorm::new(hidden_dim),
            drop2: Dropout::new(0.2),
            fc3: Linear::new(hidden_dim, hidden_dim),
            ln3: LayerNorm::new(hidden_dim),
            drop3: Dropout::new(0.2),
            fc4: Linear::new(hidden_dim, out_dim),
        }
    }
}

impl Module for DeepNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(&x);
        let x = self.ln1.forward(&x);
        let x = self.drop1.forward(&x);
        let x = x.relu();
        let x = self.fc2.forward(&x);
        let x = self.ln2.forward(&x);
        let x = self.drop2.forward(&x);
        let x = x.relu();
        let x = self.fc3.forward(&x);
        let x = self.ln3.forward(&x);
        let x = self.drop3.forward(&x);
        let x = x.relu();
        let x = self.fc4.forward(&x);
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.fc1.parameters();
        p.extend(self.ln1.parameters());
        p.extend(self.drop1.parameters());
        p.extend(self.fc2.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.drop2.parameters());
        p.extend(self.fc3.parameters());
        p.extend(self.ln3.parameters());
        p.extend(self.drop3.parameters());
        p.extend(self.fc4.parameters());
        p
    }

    fn set_training(&self, mode: bool) {
        self.drop1.set_training(mode);
        self.drop2.set_training(mode);
        self.drop3.set_training(mode);
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_features = 9;
    let hidden_dim = 64;
    let out_features = 1;

    let model = DeepNet::new(in_features, hidden_dim, out_features);
    let optimizer = SGD::new(5e-2);

    println!("--- Loading Dataset ---");
    let dataset = CsvDataset::from_file("examples/housing.csv", 8, true)?;
    println!("Loaded {} total samples.", dataset.len());

    let mut scaler = StandardScaler::new();
    scaler.fit(&dataset);
    let mut target_scaler = StandardScaler::new();
    target_scaler.fit_targets(&dataset);
    let dataset = TransformedDataset::with_target_scaler(&dataset, &scaler, &target_scaler);

    let (train_data, val_data, test_data) =
        train_val_test_split(&dataset, 0.7, 0.15, 0.15, true);
    println!(
        "Dataset split: {} train, {} validation, {} test samples.",
        train_data.len(),
        val_data.len(),
        test_data.len(),
    );

    let batch_size = 256;
    let epochs = 50;
    let checkpoint_path = "best_deepnet.bin";

    let mut train_loader = Dataloader::new(&train_data, batch_size, true);
    let mut val_loader = Dataloader::new(&val_data, batch_size, false);
    let mut best_saver = BestModelSaver::new(checkpoint_path);

    println!("\n--- Starting Training ---");
    let start_time = std::time::Instant::now();

    for epoch in 1..=epochs {
        model.train();
        let mut total_train_loss = 0.0;
        let mut train_batches = 0;

        for (inputs, targets) in train_loader.iter_batches() {
            model.zero_grad();

            let pred = model.forward(&inputs);
            let loss = mse_loss(&pred, &targets);

            loss.backward();
            optimizer.step(&model.parameters());

            let loss_val = loss.0.borrow().data[0];
            if !loss_val.is_nan() {
                total_train_loss += loss_val;
                train_batches += 1;
            }
        }

        let avg_train_loss = if train_batches > 0 {
            total_train_loss / train_batches as f32
        } else {
            0.0
        };

        // 1. Evaluation using Evaluator
        let val_report = Evaluator::evaluate(&model, &mut val_loader, mse_loss);

        // 2. Save best checkpoint
        let mut metrics_map = HashMap::new();
        metrics_map.insert("mae".to_string(), val_report.mae);
        metrics_map.insert("rmse".to_string(), val_report.rmse);
        metrics_map.insert("r2".to_string(), val_report.r2_score);
        best_saver.check_and_save(&model, epoch, val_report.loss, metrics_map);

        println!(
            "Epoch {:2}/{} - Train MSE: {:.4e} | Val: {}",
            epoch, epochs, avg_train_loss, val_report
        );
    }

    let elapsed = start_time.elapsed();
    println!("\nTraining completed in {:.2}s", elapsed.as_secs_f32());

    // 3. Load the best saved checkpoint
    println!("\n--- Loading Best Checkpoint for Testing ---");
    let mut loaded_model = DeepNet::new(in_features, hidden_dim, out_features);
    let checkpoint = ModelCheckpoint::load(&mut loaded_model, checkpoint_path)?;
    println!(
        "Loaded model from epoch {} with best val loss: {:.4e}",
        checkpoint.epoch, checkpoint.val_loss
    );

    let mut test_loader = Dataloader::new(&test_data, batch_size, false);
    let test_report = Evaluator::evaluate(&loaded_model, &mut test_loader, mse_loss);
    println!("Test Set Evaluation -> {}", test_report);

    // 4. Run sample inference
    println!("\n--- Inference on Sample Test Data ---");
    let sample_count = 5.min(test_data.len());
    for i in 0..sample_count {
        let (sample_inputs, sample_targets) = test_data.get(i);
        let input_tensor = Tensor::new(sample_inputs.clone(), (1, in_features));
        let pred_tensor = loaded_model.forward(&input_tensor);

        let pred_scaled = pred_tensor.0.borrow().data[0];
        let actual_scaled = sample_targets[0];
        let pred_val = target_scaler.inverse_transform_sample(&[pred_scaled])[0];
        let actual_val = target_scaler.inverse_transform_sample(&[actual_scaled])[0];
        let error = (pred_val - actual_val).abs();

        println!(
            "Sample #{}: Predicted = ${:.2} | Actual = ${:.2} | Absolute Error = ${:.2}",
            i + 1,
            pred_val,
            actual_val,
            error
        );
    }

    Ok(())
}
