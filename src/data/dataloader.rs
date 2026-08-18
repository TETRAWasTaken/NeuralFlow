use super::dataset::Dataset;
use crate::tensor::Tensor;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Dataloader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> Dataloader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
        }
    }
    pub fn iter_batches(&mut self) -> DataLoaderIter<'a, '_, D> {
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
        DataLoaderIter {
            loader: self,
            cursor: 0,
        }
    }
}

pub struct DataLoaderIter<'a, 'b, D: Dataset> {
    loader: &'b Dataloader<'a, D>,
    cursor: usize,
}

impl<'a, 'b, D: Dataset> Iterator for DataLoaderIter<'a, 'b, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.loader.indices.len() {
            return None;
        }

        let end = (self.cursor + self.loader.batch_size).min(self.loader.indices.len());
        let batch_indices = &self.loader.indices[self.cursor..end];
        let current_batch_size = batch_indices.len();
        self.cursor = end;

        let mut batch_inputs = Vec::new();
        let mut batch_targets = Vec::new();

        let (first_x, first_y) = self.loader.dataset.get(batch_indices[0]);
        let x_dim = first_x.len();
        let y_dim = first_y.len();

        batch_inputs.extend(first_x);
        batch_targets.extend(first_y);

        for &idx in &batch_indices[1..] {
            let (x, y) = self.loader.dataset.get(idx);
            assert_eq!(x.len(), x_dim, "Inconsistent input dimensions");
            assert_eq!(y.len(), y_dim, "inconsistent input dimensions");
            batch_inputs.extend(x);
            batch_targets.extend(y);
        }

        let input_tensor = Tensor::new(batch_inputs, (current_batch_size, x_dim));
        let target_tensor = Tensor::new(batch_targets, (current_batch_size, y_dim));

        Some((input_tensor, target_tensor))
    }
}
