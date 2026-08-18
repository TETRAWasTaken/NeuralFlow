pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>);
}

pub struct TensorDataset {
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
}

impl TensorDataset {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Input and target must have the same length"    
        );
        Self {inputs, targets}
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> (Vec<f32>, Vec<f32>) {
        (self.inputs[index].clone(), self.targets[index].clone())
    }
}