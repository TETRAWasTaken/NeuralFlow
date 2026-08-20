use super::metrics::Metrics;
use crate::data::{Dataloader, Dataset};
use crate::nn::Module;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct EvaluationReport {
    pub loss: f32,
    pub r2_score: f32,
    pub mae: f32,
    pub rmse: f32,
}

impl std::fmt::Display for EvaluationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Loss: {:.5} | R2: {:.4} | MAE: {:.4} | RMSE: {:.4}",
            self.loss, self.r2_score, self.mae, self.rmse
        )
    }
}

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate<M: Module, D: Dataset>(
        model: &M,
        loader: &mut Dataloader<D>,
        loss_fn: fn(&Tensor, &Tensor) -> Tensor,
    ) -> EvaluationReport {
        model.eval();

        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let mut all_preds = Vec::new();
        let mut all_targets = Vec::new();

        for (x_batch, y_batch) in loader.iter_batches() {
            let pred = model.forward(&x_batch);
            let loss = loss_fn(&pred, &y_batch);

            total_loss += loss.0.borrow().data[0];
            batch_count += 1;

            all_preds.extend(pred.0.borrow().data.clone());
            all_targets.extend(y_batch.0.borrow().data.clone());
        }

        let avg_loss = if batch_count > 0 {
            total_loss / (batch_count as f32)
        } else {
            0.0
        };

        let rmse = Metrics::rmse(&all_preds, &all_targets);
        let mae = Metrics::mae(&all_preds, &all_targets);
        let r2 = Metrics::r2_score(&all_preds, &all_targets);

        EvaluationReport {
            loss: avg_loss,
            r2_score: r2,
            mae,
            rmse,
        }
    }
}