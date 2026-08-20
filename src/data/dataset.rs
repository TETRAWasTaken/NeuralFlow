pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>);

    fn get_into(&self, index: usize, x_dst: &mut Vec<f32>, y_dst: &mut Vec<f32>) {
        let (x, y) = self.get(index);
        x_dst.extend_from_slice(&x);
        y_dst.extend_from_slice(&y);
    }
}

pub struct TensorDataset {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    num_samples: usize,
    input_dim: usize,
    target_dim: usize,
}

impl TensorDataset {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Input and target must have the same length"
        );
        let num_samples = inputs.len();
        if num_samples == 0 {
            return Self {
                inputs: Vec::new(),
                targets: Vec::new(),
                num_samples: 0,
                input_dim: 0,
                target_dim: 0,
            };
        }
        let input_dim = inputs[0].len();
        let target_dim = targets[0].len();
        let mut flat_inputs = Vec::with_capacity(num_samples * input_dim);
        let mut flat_targets = Vec::with_capacity(num_samples * target_dim);
        for (x, y) in inputs.into_iter().zip(targets.into_iter()) {
            assert_eq!(x.len(), input_dim, "Inconsistent input feature dimension");
            assert_eq!(y.len(), target_dim, "Inconsistent target feature dimension");
            flat_inputs.extend(x);
            flat_targets.extend(y);
        }
        Self {
            inputs: flat_inputs,
            targets: flat_targets,
            num_samples,
            input_dim,
            target_dim,
        }
    }

    pub fn from_flat(
        inputs: Vec<f32>,
        targets: Vec<f32>,
        num_samples: usize,
        input_dim: usize,
        target_dim: usize,
    ) -> Self {
        assert_eq!(inputs.len(), num_samples * input_dim);
        assert_eq!(targets.len(), num_samples * target_dim);
        Self {
            inputs,
            targets,
            num_samples,
            input_dim,
            target_dim,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn target_dim(&self) -> usize {
        self.target_dim
    }
}

impl Dataset for TensorDataset {
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