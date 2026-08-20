#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod data;
pub mod eval;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod utils;

pub mod prelude {
    pub use crate::data::{
        CsvDataset, Dataloader, Dataset, KFold, Subset, TensorDataset, train_test_split,
        train_val_test_split, StandardScaler, TransformedDataset,
    };
    pub use crate::eval::{EvaluationReport, Evaluator, Metrics};
    pub use crate::nn::{
        Dropout, LayerNorm, LeakyReLu, Linear, Module, ReLu, Sigmoid, Tanh, mse_loss,
    };
    pub use crate::optim::{optimizer::Optimizer, sgd::SGD};
    pub use crate::tensor::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard, Tensor};
    pub use crate::utils::checkpoint::{BestModelSaver, ModelCheckpoint};
    pub use crate::utils::serialisation::TensorState;
}

