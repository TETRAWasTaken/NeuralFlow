use NeuralFlow::prelude::*;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

/// Loads raw MNIST IDX binary files directly into a TensorDataset.
///
/// Images: 16-byte header followed by (num_samples * 28 * 28) unsigned bytes normalized to [0, 1].
/// Labels: 8-byte header followed by (num_samples) unsigned bytes in [0..9].
fn load_mnist(
    image_path: &str,
    label_path: &str,
    max_samples: Option<usize>,
) -> std::io::Result<TensorDataset> {
    let mut img_file = File::open(image_path)?;
    let mut lbl_file = File::open(label_path)?;

    let mut img_buf = Vec::new();
    let mut lbl_buf = Vec::new();

    img_file.read_to_end(&mut img_buf)?;
    lbl_file.read_to_end(&mut lbl_buf)?;

    let total_images = u32::from_be_bytes(img_buf[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(img_buf[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(img_buf[12..16].try_into().unwrap()) as usize;
    let image_pixels = rows * cols; // 784

    let total_labels = u32::from_be_bytes(lbl_buf[4..8].try_into().unwrap()) as usize;
    assert_eq!(total_images, total_labels);

    let count = match max_samples {
        Some(m) => m.min(total_images),
        None => total_images,
    };

    let mut all_images = Vec::with_capacity(count);
    let mut all_labels = Vec::with_capacity(count);

    for i in 0..count {
        let img_start = 16 + i * image_pixels;
        let img_end = img_start + image_pixels;
        let pixels: Vec<f32> = img_buf[img_start..img_end]
            .iter()
            .map(|&b| (b as f32) / 255.0)
            .collect();

        let label = lbl_buf[8 + i] as f32;

        all_images.push(pixels);
        all_labels.push(vec![label]);
    }

    Ok(TensorDataset::new(all_images, all_labels))
}

/// Creates the benchmark CNN architecture:
/// Input: (N, 1, 28, 28)
/// - Conv2d(1, 16, 3x3, pad=1, stride=1) -> ReLU -> MaxPool2d(2x2) => (N, 16, 14, 14)
/// - Conv2d(16, 32, 3x3, pad=1, stride=1) -> ReLU -> MaxPool2d(2x2) => (N, 32, 7, 7)
/// - Flatten => (N, 1568)
/// - Linear(1568, 128) -> ReLU -> Dropout(0.2)
/// - Linear(128, 10) => Logits
fn build_cnn_model() -> Sequential {
    Sequential::new(vec![
        // Block 1
        Box::new(Conv2d::new_with_options(
            1,
            16,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )),
        Box::new(ReLu),
        Box::new(MaxPool2d::new((2, 2))),
        // Block 2
        Box::new(Conv2d::new_with_options(
            16,
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )),
        Box::new(ReLu),
        Box::new(MaxPool2d::new((2, 2))),
        // Classifier
        Box::new(Flatten::new()),
        Box::new(Linear::new(32 * 7 * 7, 128)),
        Box::new(ReLu),
        Box::new(Dropout::new(0.2)),
        Box::new(Linear::new(128, 10)),
    ])
}

fn count_parameters(model: &Sequential) -> usize {
    model
        .parameters()
        .iter()
        .map(|p| p.0.borrow().data.len())
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  NeuralFlow: MNIST CNN Training & Benchmark");
    println!("============================================================");

    // Paths to MNIST binary files
    let train_images_path = "data/mnist/train-images-idx3-ubyte";
    let train_labels_path = "data/mnist/train-labels-idx1-ubyte";
    let test_images_path = "data/mnist/t10k-images-idx3-ubyte";
    let test_labels_path = "data/mnist/t10k-labels-idx1-ubyte";

    // Dataset size configuration
    // (Set to None for full 60,000 train / 10,000 test, or use subset for fast comparative benchmarks)
    let train_subset_size = Some(2_000);
    let test_subset_size = Some(1_000);

    println!("\nLoading MNIST dataset...");
    let load_start = Instant::now();
    let train_data = load_mnist(train_images_path, train_labels_path, train_subset_size)?;
    let test_data = load_mnist(test_images_path, test_labels_path, test_subset_size)?;
    println!(
        "Loaded {} train samples, {} test samples in {:.2}s",
        train_data.len(),
        test_data.len(),
        load_start.elapsed().as_secs_f32()
    );

    // Hyperparameters (matched 1:1 with PyTorch)
    let batch_size = 128;
    let epochs = 20;
    let learning_rate = 0.001;

    let mut train_loader = Dataloader::new(&train_data, batch_size, true);
    let mut test_loader = Dataloader::new(&test_data, batch_size, false);

    // Build Model
    let model = build_cnn_model();
    let total_params = count_parameters(&model);
    println!(
        "Model initialized with {} trainable parameters",
        total_params
    );
    println!(
        "Optimizer: Adam (lr={}) | Loss: CrossEntropyLoss",
        learning_rate
    );
    println!("Batch size: {} | Epochs: {}\n", batch_size, epochs);

    let mut optimizer = Adam::new(learning_rate, 0.9, 0.999, 1e-8, 0.0);

    println!("{:-<75}", "");
    println!(
        "{:<6} | {:<10} | {:<9} | {:<10} | {:<8} | {:<12}",
        "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Throughput"
    );
    println!("{:-<75}", "");

    let overall_start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        // Training Phase
        model.train();
        let mut total_train_loss = 0.0;
        let mut train_correct = 0;
        let mut train_samples = 0;

        for (x_batch, y_batch) in train_loader.iter_batches() {
            let current_b = x_batch.shape().0;
            // Reshape (N, 784) -> (N, 1, 28, 28) for Conv2d
            let x_4d = x_batch.reshape_4d((current_b, 1, 28, 28));

            model.zero_grad();

            let logits = model.forward(&x_4d);
            let loss = cross_entropy_loss(&logits, &y_batch);

            loss.backward();
            optimizer.step(&model.parameters());

            let loss_val = loss.0.borrow().data[0];
            total_train_loss += loss_val * (current_b as f32);

            let acc =
                Metrics::accuracy_logits(&logits.0.borrow().data, &y_batch.0.borrow().data, 10);
            train_correct += (acc * current_b as f32).round() as usize;
            train_samples += current_b;
        }

        let epoch_duration = epoch_start.elapsed().as_secs_f32();
        let samples_per_sec = (train_samples as f32) / epoch_duration;
        let avg_train_loss = total_train_loss / (train_samples as f32);
        let train_accuracy = (train_correct as f32) / (train_samples as f32) * 100.0;

        // Validation Phase
        model.eval();
        let _guard = no_grad();
        let mut total_val_loss = 0.0;
        let mut val_correct = 0;
        let mut val_samples = 0;

        for (x_batch, y_batch) in test_loader.iter_batches() {
            let current_b = x_batch.shape().0;
            let x_4d = x_batch.reshape_4d((current_b, 1, 28, 28));

            let logits = model.forward(&x_4d);
            let loss = cross_entropy_loss(&logits, &y_batch);

            let loss_val = loss.0.borrow().data[0];
            total_val_loss += loss_val * (current_b as f32);

            let acc =
                Metrics::accuracy_logits(&logits.0.borrow().data, &y_batch.0.borrow().data, 10);
            val_correct += (acc * current_b as f32).round() as usize;
            val_samples += current_b;
        }

        let avg_val_loss = total_val_loss / (val_samples as f32);
        let val_accuracy = (val_correct as f32) / (val_samples as f32) * 100.0;

        println!(
            "{:<6} | {:<10.4} | {:<8.2}% | {:<10.4} | {:<7.2}% | {:<7.1} img/s",
            format!("{}/{}", epoch, epochs),
            avg_train_loss,
            train_accuracy,
            avg_val_loss,
            val_accuracy,
            samples_per_sec
        );
    }

    println!("{:-<75}", "");
    println!(
        "\nBenchmark completed in {:.2}s",
        overall_start.elapsed().as_secs_f32()
    );

    Ok(())
}
