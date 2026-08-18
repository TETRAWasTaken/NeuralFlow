use super::dataset::Dataset;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct CsvDataset {
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
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

            let target = vec![values[target_col_index]];
            let input: Vec<f32> = values
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| *idx != target_col_index)
                .map(|(_, val)| val)
                .collect();

            inputs.push(input);
            targets.push(target);
        }

        Ok(Self { inputs, targets })
    }
}

impl Dataset for CsvDataset {
    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        (self.inputs[index].clone(), self.targets[index].clone())
    }
}