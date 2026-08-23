use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 4_096;

pub struct Metrics;

impl Metrics {
    pub fn mae(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let len = preds.len();
        if len == 0 {
            return 0.0;
        }

        let sum: f32 = if len > PARALLEL_THRESHOLD {
            preds
                .par_iter()
                .zip(targets.par_iter())
                .map(|(&p, &t)| (p - t).abs())
                .sum()
        } else {
            preds
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| (p - t).abs())
                .sum()
        };

        sum / (len as f32)
    }

    pub fn mse(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let len = preds.len();
        if len == 0 {
            return 0.0;
        }

        let sum: f32 = if len > PARALLEL_THRESHOLD {
            preds
                .par_iter()
                .zip(targets.par_iter())
                .map(|(&p, &t)| {
                    let diff = p - t;
                    diff * diff
                })
                .sum()
        } else {
            preds
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| {
                    let diff = p - t;
                    diff * diff
                })
                .sum()
        };

        sum / (len as f32)
    }

    pub fn rmse(preds: &[f32], targets: &[f32]) -> f32 {
        Self::mse(preds, targets).sqrt()
    }

    pub fn r2_score(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let len = targets.len();
        if len == 0 {
            return 0.0;
        }
        let n = len as f32;

        let (mean_target, ss_tot, ss_res) = if len > PARALLEL_THRESHOLD {
            let sum_target: f32 = targets.par_iter().sum();
            let mean = sum_target / n;
            let total_ss: f32 = targets
                .par_iter()
                .map(|&t| {
                    let diff = t - mean;
                    diff * diff
                })
                .sum();
            let res_ss: f32 = preds
                .par_iter()
                .zip(targets.par_iter())
                .map(|(&p, &t)| {
                    let diff = p - t;
                    diff * diff
                })
                .sum();
            (mean, total_ss, res_ss)
        } else {
            let sum_target: f32 = targets.iter().sum();
            let mean = sum_target / n;
            let total_ss: f32 = targets
                .iter()
                .map(|&t| {
                    let diff = t - mean;
                    diff * diff
                })
                .sum();
            let res_ss: f32 = preds
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| {
                    let diff = p - t;
                    diff * diff
                })
                .sum();
            (mean, total_ss, res_ss)
        };

        let _ = mean_target;
        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}