pub struct Metrics;

impl Metrics {
    pub fn mae(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let sum: f32 = preds
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum();

        sum / preds.len() as f32
    }

    pub fn mse(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let sum: f32 = preds
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        sum / preds.len() as f32
    }

    pub fn rmse(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let sum: f32 = preds
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        (sum / preds.len() as f32).sqrt()
    }

    pub fn r2_score(preds: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(preds.len(), targets.len());
        let n = targets.len() as f32;
        let mean_target = targets.iter().sum::<f32>() / n;

        let ss_tot: f32 = targets.iter().map(|&t| (t - mean_target).powi(2)).sum();
        let ss_res: f32 = preds
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}