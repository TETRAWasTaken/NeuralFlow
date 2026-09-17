#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod data;
pub mod eval;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod utils;
pub mod xla;

pub mod prelude {
    pub use crate::data::{
        CsvDataset, Dataloader, Dataset, KFold, Subset, TensorDataset, train_test_split,
        train_val_test_split, StandardScaler, TransformedDataset,
    };
    pub use crate::eval::{EvaluationReport, Evaluator, Metrics};
    pub use crate::nn::{
        bce_loss, bce_with_logits_loss, cross_entropy_loss, AvgPool2d, Conv2d, Dropout, Flatten,
        LayerNorm, LeakyReLu, Linear, MaxPool2d, Module, mse_loss, ReLu, Sequential, Sigmoid, Tanh,
    };
    pub use crate::optim::{optimizer::Optimizer, sgd::SGD, adam::Adam};
    pub use crate::tensor::{
        is_grad_enabled, no_grad, set_grad_enabled, Device, NoGradGuard, Shape, Tensor,
    };
    pub use crate::utils::checkpoint::{BestModelSaver, ModelCheckpoint};
    pub use crate::utils::serialisation::TensorState;
    pub use crate::xla::XlaTraceable;
}

