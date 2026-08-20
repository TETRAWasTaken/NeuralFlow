use crate::nn::Module;
use crate::utils::serialisation::TensorState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelCheckpoint {
    pub epoch: usize,
    pub val_loss: f32,
    pub metrics: HashMap<String, f32>,
    pub weights: Vec<TensorState>,
}

impl ModelCheckpoint {
    pub fn save<M: Module>(
        model: &M,
        epoch: usize,
        val_loss: f32,
        metrics: HashMap<String, f32>,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let weights: Vec<TensorState> = model
            .parameters()
            .iter()
            .map(|p| {
                let inner = p.0.borrow();
                TensorState {
                    data: inner.data.clone(),
                    shape: inner.shape,
                }
            })
            .collect();

        let checkpoint = ModelCheckpoint {
            epoch,
            val_loss,
            metrics,
            weights,
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &checkpoint)?;
        Ok(())
    }

    pub fn load<M: Module>(
        model: &mut M,
        path: &str,
    ) -> Result<ModelCheckpoint, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint: ModelCheckpoint = bincode::deserialize_from(reader)?;

        let params = model.parameters();
        assert_eq!(
            params.len(),
            checkpoint.weights.len(),
            "Parameter counts do not match"
        );

        for (param, state) in params.iter().zip(checkpoint.weights.iter()) {
            let mut inner = param.0.borrow_mut();
            assert_eq!(
                inner.shape, state.shape,
                "Shape mismatch while loading weights"
            );
            inner.data = state.data.clone();
        }

        Ok(checkpoint)
    }
}

pub struct BestModelSaver {
    save_path: String,
    pub best_loss: f32,
}

impl BestModelSaver {
    pub fn new(save_path: &str) -> Self {
        Self {
            save_path: save_path.to_string(),
            best_loss: f32::INFINITY,
        }
    }

    pub fn check_and_save<M: Module>(
        &mut self,
        model: &M,
        epoch: usize,
        val_loss: f32,
        metrics: HashMap<String, f32>,
    ) -> bool {
        if val_loss < self.best_loss {
            println!(
                "Loss improved from {:.5} to {:.5}. Saving checkpoint to '{}'",
                self.best_loss, val_loss, self.save_path
            );
            self.best_loss = val_loss;
            ModelCheckpoint::save(model, epoch, val_loss, metrics, &self.save_path)
                .expect("Failed to save the best model");
            true
        } else {
            false
        }
    }
}