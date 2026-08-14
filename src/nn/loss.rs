use crate::tensor::Tensor;

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let p_data = pred.0.borrow().data.clone();
    let t_data = target.0.borrow().data.clone();

    let mut diff_sum = 0.0;
    let n = p_data.len();
    for i in 0..n {
        let d = p_data[i] - t_data[i];
        diff_sum += d * d;
    }
    let loss_val = diff_sum / (n as f32);

    let loss = Tensor::new(vec![loss_val], (1, 1));
    loss.0.borrow_mut().prev = vec![pred.clone()];

    let pred_clone = pred.clone();
    let target_clone = target.clone();
    let loss_clone = loss.clone();

    loss.0.borrow_mut().backward = Some(Box::new(move || {
        let loss_grad = loss_clone.0.borrow().grad[0];
        let p = pred_clone.0.borrow().data.clone();
        let t = target_clone.0.borrow().data.clone();
        let mut p_grad = pred_clone.0.borrow_mut();

        for i in 0..p.len() {
            p_grad.grad[i] += (2.0 / n as f32) * (p[i] - t[i]) * loss_grad;
        }      
    }));

    loss
}