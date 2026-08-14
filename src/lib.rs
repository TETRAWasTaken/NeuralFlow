pub mod nn;
pub mod optim;
pub mod tensor;

pub mod prelude {
    pub use crate::nn::{linear::Linear, loss::mse_loss, module::Module};
    pub use crate::optim::{optimizer::Optimizer, sgd::SGD};
    pub use crate::tensor::Tensor;
}