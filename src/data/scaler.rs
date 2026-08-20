use super::dataset::Dataset;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub ops: f32,
    pub is_fitted: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            std: Vec::new(),
            ops: 1e-8,
            is_fitted: false,
        }
    }

    pub fn fit<D: Dataset>(&mut self, dataset: &D) {
        assert!(!dataset.is_empty(), "Dataset is empty");

        let n = dataset.len() as f32;
        let (first_x, _) = dataset.get(0);
        let num_features = first_x.len();

        let mut sum = vec![0.0; num_features];
        let mut sum_eq = vec![0.0; num_features];
        
        for i in 0..dataset.len() {
            let (x, _) = dataset.get(i);
            assert_eq!(
                x.len(),
                num_features,
                "All samples must have the same number of features"
            );
            for j in 0..num_features {
                sum[j] += x[j];
                sum_eq[j] += x[j] * x[j];
            }
        }

        self.mean = sum.iter().map(|&s| s / n).collect();
        self.std = (0..num_features)
            .map(|j| {
                let variance = (sum_eq[j] / n) - (self.mean[j] * self.mean[j]);
                variance.max(0.0).sqrt() + self.ops
            })
            .collect();

        self.is_fitted = true;
    }

    pub fn fit_targets<D: Dataset>(&mut self, dataset: &D) {
        assert!(!dataset.is_empty(), "Dataset is empty");

        let n = dataset.len() as f32;
        let (_, first_y) = dataset.get(0);
        let num_targets = first_y.len();

        let mut sum = vec![0.0; num_targets];
        let mut sum_eq = vec![0.0; num_targets];

        for i in 0..dataset.len() {
            let (_, y) = dataset.get(i);
            assert_eq!(
                y.len(),
                num_targets,
                "All samples must have the same number of targets"
            );
            for j in 0..num_targets {
                sum[j] += y[j];
                sum_eq[j] += y[j] * y[j];
            }
        }

        self.mean = sum.iter().map(|&s| s / n).collect();
        self.std = (0..num_targets)
            .map(|j| {
                let variance = (sum_eq[j] / n) - (self.mean[j] * self.mean[j]);
                variance.max(0.0).sqrt() + self.ops
            })
            .collect();

        self.is_fitted = true;
    }

    pub fn transform_sample(&self, features: &[f32]) -> Vec<f32> {
        assert!(
            self.is_fitted, "Scaler has not been fitted yet"
        );
        assert_eq!(
            features.len(),
            self.mean.len(),
            "Input features must have the same length as the fitted mean"
        );

        features
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(&x, (&m, &s))| (x - m) / s)
            .collect()
    }

    pub fn inverse_transform_sample(&self, features: &[f32]) -> Vec<f32> {
        assert!(
            self.is_fitted, "Scaler has not been fitted yet"
        );
        assert_eq!(
            features.len(),
            self.mean.len(),
            "Input features must have the same length as the fitted mean"
        );

        features
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(&z, (&m, &s))| (z * s) + m)
            .collect()
    }

    pub fn transform_dataset<'a, D: Dataset>(&'a self, dataset: &'a D) -> TransformedDataset<'a, D> {
        assert!(
            self.is_fitted, "Scaler has not been fitted yet"
        );
        TransformedDataset::new(dataset, self)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TransformedDataset<'a, D: Dataset> {
    scaler: &'a StandardScaler,
    target_scaler: Option<&'a StandardScaler>,
    dataset: &'a D,
}

impl<'a, D: Dataset> TransformedDataset<'a, D> {
    pub fn new(dataset: &'a D, scaler: &'a StandardScaler) -> Self {
        Self {
            scaler,
            target_scaler: None,
            dataset,
        }
    }

    pub fn with_target_scaler(
        dataset: &'a D,
        scaler: &'a StandardScaler,
        target_scaler: &'a StandardScaler,
    ) -> Self {
        Self {
            scaler,
            target_scaler: Some(target_scaler),
            dataset,
        }
    }
}

impl<'a, D: Dataset> Dataset for TransformedDataset<'a, D> {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        let (x, y) = self.dataset.get(index);
        let scaled_x = self.scaler.transform_sample(&x);
        let scaled_y = match self.target_scaler {
            Some(ts) => ts.transform_sample(&y),
            None => y,
        };
        (scaled_x, scaled_y)
    }

    fn get_into(&self, index: usize, x_dst: &mut Vec<f32>, y_dst: &mut Vec<f32>) {
        let (x, y) = self.dataset.get(index);
        let scaled_x = self.scaler.transform_sample(&x);
        x_dst.extend_from_slice(&scaled_x);
        if let Some(ts) = self.target_scaler {
            let scaled_y = ts.transform_sample(&y);
            y_dst.extend_from_slice(&scaled_y);
        } else {
            y_dst.extend_from_slice(&y);
        }
    }
}