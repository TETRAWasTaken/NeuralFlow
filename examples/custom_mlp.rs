use NeuralFlow::prelude::*;

pub struct MyCustomModel {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MyCustomModel {
    pub fn new() -> Self {
        Self {
            fc1: Linear::new(2, 4),
            fc2: Linear::new(4, 4),
            fc3: Linear::new(4, 1),
        }
    }
}

impl Module for MyCustomModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h).relu();
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

fn main() {
    let model = MyCustomModel::new();
    let optimizer = SGD::new(0.01);

    let input = Tensor::new(vec![0.5, -0.2], (1, 2));
    let target = Tensor::new(vec![1.0], (1, 1));

    println!("Training custom model...");
    for epoch in 0..1000 {
        model.zero_grad();

        let pred = model.forward(&input);
        let loss = mse_loss(&pred, &target);

        loss.backward();
        optimizer.step(&model.parameters());

        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.0.borrow().data[0]);
        }
    }
}
