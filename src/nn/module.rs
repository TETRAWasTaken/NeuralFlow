use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn set_training(&self, _mode: bool) {}

    fn train(&self) {
        self.set_training(true);
    }

    fn eval(&self) {
        self.set_training(false);
    }
}