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

    pub fn accuracy(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let len = preds.len();
        if len == 0 {
            return 0.0;
        }

        let correct: usize = if len > PARALLEL_THRESHOLD {
            preds
                .par_iter()
                .zip(targets.par_iter())
                .filter(|&(p, t)| p.round() == t.round())
                .count()
        } else {
            preds
                .iter()
                .zip(targets.iter())
                .filter(|&(p, t)| p.round() == t.round())
                .count()
        };

        (correct as f32) / (len as f32)
    }

    pub fn accuracy_logits(logits: &[f32], targets: &[f32], num_classes: usize) -> f32 {
        assert!(num_classes > 0, "num_classes must be greater than 0");
        assert_eq!(
            logits.len() % num_classes,
            0,
            "logits length must be divisible by num_classes"
        );
        let num_samples = logits.len() / num_classes;
        assert_eq!(
            targets.len(),
            num_samples,
            "targets count must match num_samples"
        );
        if num_samples == 0 {
            return 0.0;
        }

        let correct: usize = if num_samples > PARALLEL_THRESHOLD {
            (0..num_samples)
                .into_par_iter()
                .filter(|&i| {
                    let row = &logits[i * num_classes..(i + 1) * num_classes];
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;
                    for (c, &val) in row.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = c;
                        }
                    }
                    max_idx == targets[i].round() as usize
                })
                .count()
        } else {
            (0..num_samples)
                .filter(|&i| {
                    let row = &logits[i * num_classes..(i + 1) * num_classes];
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;
                    for (c, &val) in row.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = c;
                        }
                    }
                    max_idx == targets[i].round() as usize
                })
                .count()
        };

        (correct as f32) / (num_samples as f32)
    }
}