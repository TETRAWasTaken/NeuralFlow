use super::dataset::Dataset;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct CsvDataset {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    num_samples: usize,
    input_dim: usize,
    target_dim: usize,
}

impl CsvDataset {
    /// Loads a CSV file where `target_col_index` specifies which column is the label.
    /// `has_header` skips the first row if set to true.
    pub fn from_file(
        path: &str,
        target_col_index: usize,
        has_header: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut num_samples = 0;
        let mut input_dim = 0;
        let target_dim = 1;

        for (line_idx, line) in reader.lines().enumerate() {
            let line_str = line?;
            if line_idx == 0 && has_header {
                continue;
            }

            let values: Vec<f32> = line_str
                .split(',')
                .map(|val| val.trim().parse::<f32>().unwrap_or(0.0))
                .collect();

            if values.is_empty() {
                continue;
            }

            targets.push(values[target_col_index]);

            let mut row_input_count = 0;
            for (idx, val) in values.into_iter().enumerate() {
                if idx != target_col_index {
                    inputs.push(val);
                    row_input_count += 1;
                }
            }

            if num_samples == 0 {
                input_dim = row_input_count;
            } else {
                assert_eq!(row_input_count, input_dim, "Inconsistent column count in CSV");
            }

            num_samples += 1;
        }

        Ok(Self {
            inputs,
            targets,
            num_samples,
            input_dim,
            target_dim,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn target_dim(&self) -> usize {
        self.target_dim
    }
}

impl Dataset for CsvDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        assert!(index < self.num_samples, "Index out of bounds");
        let x_start = index * self.input_dim;
        let y_start = index * self.target_dim;
        (
            self.inputs[x_start..x_start + self.input_dim].to_vec(),
            self.targets[y_start..y_start + self.target_dim].to_vec(),
        )
    }

    fn get_into(&self, index: usize, x_dst: &mut Vec<f32>, y_dst: &mut Vec<f32>) {
        assert!(index < self.num_samples, "Index out of bounds");
        let x_start = index * self.input_dim;
        let y_start = index * self.target_dim;
        x_dst.extend_from_slice(&self.inputs[x_start..x_start + self.input_dim]);
        y_dst.extend_from_slice(&self.targets[y_start..y_start + self.target_dim]);
    }
}