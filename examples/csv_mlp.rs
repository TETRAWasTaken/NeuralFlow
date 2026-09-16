use NeuralFlow::prelude::*;

pub struct HousingModel {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    fc4: Linear,
}

impl HousingModel {
    pub fn new(in_features: usize, hidden: usize, out_features: usize) -> Self {
        Self {
            fc1: Linear::new(in_features, hidden),
            fc2: Linear::new(hidden, hidden),
            fc3: Linear::new(hidden, hidden),
            fc4: Linear::new(hidden, out_features),
        }
    }
}

impl Module for HousingModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x).tanh();
        let h2 = self.fc2.forward(&h).tanh();
        let h3 = self.fc3.forward(&h2).tanh();
        self.fc4.forward(&h3)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params.extend(self.fc4.parameters());
        params
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading housing dataset from CSV...");
    // Column index 8 is `median_house_value` (target), with header row skipped
    let dataset = CsvDataset::from_file("examples/housing.csv", 8, true)?;
    println!("Loaded {} total samples.", dataset.len());

    let (train_data, test_data) = train_test_split(&dataset, 0.2, true);
    println!(
        "Dataset split: {} training samples, {} testing samples.",
        train_data.len(),
        test_data.len()
    );

    // 9 input features -> hidden layers -> 1 output feature (house price)
    let in_features = 9;
    let hidden_dim = 16;
    let out_features = 1;

    let model = HousingModel::new(in_features, hidden_dim, out_features);
    let mut optimizer = SGD::new(1e-9);

    let batch_size = 64;
    let epochs = 50;

    let mut train_loader = Dataloader::new(&train_data, batch_size, true);
    let mut test_loader = Dataloader::new(&test_data, batch_size, false);

    println!("Starting training for {} epochs...", epochs);

    for epoch in 1..=epochs {
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

        if epoch % 5 == 0 || epoch == 1 {
            // Evaluate on test set
            let mut total_test_loss = 0.0;
            let mut test_batches = 0;

            for (test_inputs, test_targets) in test_loader.iter_batches() {
                let test_pred = model.forward(&test_inputs);
                let test_loss = mse_loss(&test_pred, &test_targets);
                let loss_val = test_loss.0.borrow().data[0];
                if !loss_val.is_nan() {
                    total_test_loss += loss_val;
                    test_batches += 1;
                }
            }

            let avg_test_loss = if test_batches > 0 {
                total_test_loss / test_batches as f32
            } else {
                0.0
            };

            println!(
                "Epoch {:3}/{} - Train MSE Loss: {:.4e} - Test MSE Loss: {:.4e}",
                epoch, epochs, avg_train_loss, avg_test_loss
            );
        }
    }

    println!("Training finished successfully!");
    Ok(())
}
