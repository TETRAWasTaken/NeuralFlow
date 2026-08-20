use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorState {
    pub data: Vec<f32>,
    pub shape: (usize, usize),
}
