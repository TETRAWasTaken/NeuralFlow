use super::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Subset<'a, D: Dataset> {
    dataset: &'a D,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> Subset<'a, D> {
    pub fn new(dataset: &'a D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

impl<'a, D: Dataset> Dataset for Subset<'a, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        let original_index = self.indices[index];
        self.dataset.get(original_index)
    }
}

pub fn train_test_split<'a, D: Dataset>(
    dataset: &'a D,
    test_ratio: f32,
    shuffle: bool,
) -> (Subset<'a, D>, Subset<'a, D>) {
    assert!(
        test_ratio > 0.0 && test_ratio < 1.0,
        "test_ratio must be between 0.0 and 1.0"
    );

    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    if shuffle {
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
    }

    let test_len = ((dataset.len() as f32) * test_ratio).round() as usize;
    let train_len = dataset.len().saturating_sub(test_len);

    let train_indices = indices[..train_len].to_vec();
    let test_indices = indices[train_len..].to_vec();

    (
        Subset::new(dataset, train_indices),
        Subset::new(dataset, test_indices),
    )
}

pub fn train_val_test_split<'a, D: Dataset>(
    dataset: &'a D,
    train_ratio: f32,
    val_ratio: f32,
    test_ratio: f32,
    shuffle: bool,
) -> (Subset<'a, D>, Subset<'a, D>, Subset<'a, D>) {
    let total = train_ratio + val_ratio + test_ratio;
    assert!(
        (total - 1.0).abs() < 1e-4,
        "Ratios must sum to 1.0 (got {})",
        total
    );

    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    if shuffle {
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
    }

    let n = dataset.len() as f32;
    let train_len = (n * train_ratio).round() as usize;
    let val_len = (n * val_ratio).round() as usize;

    let train_end = train_len;
    let val_end = (train_end + val_len).min(dataset.len());

    let train_indices = indices[..train_end].to_vec();
    let val_indices = indices[train_end..val_end].to_vec();
    let test_indices = indices[val_end..].to_vec();

    (
        Subset::new(dataset, train_indices),
        Subset::new(dataset, val_indices),
        Subset::new(dataset, test_indices),
    )
}

/// K-Fold cross validation split generator.
pub struct KFold<'a, D: Dataset> {
    dataset: &'a D,
    k: usize,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> KFold<'a, D> {
    pub fn new(dataset: &'a D, k: usize, shuffle: bool) -> Self {
        assert!(k >= 2, "k must be at least 2 for K-Fold splitting");
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }
        Self { dataset, k, indices }
    }

    /// Returns a vector of k pairs: `[(Train_Fold_0, Val_Fold_0), ..., (Train_Fold_k, Val_Fold_k)]`
    pub fn split(&self) -> Vec<(Subset<'a, D>, Subset<'a, D>)> {
        let n = self.dataset.len();
        let fold_size = n / self.k;
        let mut folds = Vec::new();

        for i in 0..self.k {
            let val_start = i * fold_size;
            let val_end = if i == self.k - 1 { n } else { (i + 1) * fold_size };

            let val_indices = self.indices[val_start..val_end].to_vec();
            let mut train_indices = Vec::new();
            train_indices.extend_from_slice(&self.indices[..val_start]);
            train_indices.extend_from_slice(&self.indices[val_end..]);

            folds.push((
                Subset::new(self.dataset, train_indices),
                Subset::new(self.dataset, val_indices),
            ));
        }

        folds
    }
}