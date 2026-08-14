use super::inner::{Tensor, TensorInner};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

impl Tensor {
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(tensor: &Tensor, visited: &mut HashSet<*const RefCell<TensorInner>>, topo: &mut Vec<Tensor>) {
            let ptr = Rc::as_ptr(&tensor.0);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for parent in &tensor.0.borrow().prev {
                    build_topo(parent, visited, topo);
                }
                topo.push(tensor.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);

        {
            let mut inner = self.0.borrow_mut();
            for g in inner.grad.iter_mut() {
                *g = 1.0;
            }
        }
        for node in topo.iter().rev() {
            if let Some(ref backward_fn) = node.0.borrow().backward {
                backward_fn();
            }   
        }
    }
}