// Tree-walking interpreter — evaluates a Graph given concrete input tensors.
// Returns a Vec<Tensor> indexed by NodeId (one value per node).

use ndarray::{ArrayD, IxDyn, s};
use crate::ir::{Graph, Op};

pub type Tensor = ArrayD<f64>;

pub fn eval(g: &Graph, inputs: &[Tensor]) -> Vec<Tensor> {
    assert_eq!(inputs.len(), g.num_inputs(), "wrong number of inputs");
    let mut vals: Vec<Option<Tensor>> = vec![None; g.nodes.len()];

    for id in 0..g.nodes.len() {
        let node = &g.nodes[id];
        let v = match &node.op {
            Op::Input(i) => inputs[*i].clone(),

            Op::Const(c) => {
                if node.shape.is_empty() {
                    ArrayD::from_elem(IxDyn(&[]), *c)
                } else {
                    ArrayD::from_elem(IxDyn(&node.shape), *c)
                }
            }

            Op::Add => get(&vals, node.inputs[0]) + get(&vals, node.inputs[1]),
            Op::Sub => get(&vals, node.inputs[0]) - get(&vals, node.inputs[1]),
            Op::Mul => get(&vals, node.inputs[0]) * get(&vals, node.inputs[1]),
            Op::Div => get(&vals, node.inputs[0]) / get(&vals, node.inputs[1]),

            Op::Neg => -get(&vals, node.inputs[0]),
            Op::Relu => get(&vals, node.inputs[0]).mapv(|x| x.max(0.0)),
            Op::Step => get(&vals, node.inputs[0]).mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Op::Exp  => get(&vals, node.inputs[0]).mapv(f64::exp),
            Op::Log  => get(&vals, node.inputs[0]).mapv(f64::ln),
            Op::Tanh => get(&vals, node.inputs[0]).mapv(f64::tanh),
            Op::Sigmoid => get(&vals, node.inputs[0]).mapv(|x| 1.0 / (1.0 + (-x).exp())),

            Op::MatMul => matmul_op(get(&vals, node.inputs[0]), get(&vals, node.inputs[1])),

            Op::Transpose => {
                let a = get(&vals, node.inputs[0]);
                a.t().to_owned()
            }

            Op::Broadcast(shape) => {
                let a = get(&vals, node.inputs[0]);
                // scalar → any shape
                if a.ndim() == 0 {
                    ArrayD::from_elem(IxDyn(shape), *a.first().unwrap())
                } else {
                    a.broadcast(IxDyn(shape)).expect("broadcast failed").to_owned()
                }
            }

            Op::Reshape(shape) => {
                get(&vals, node.inputs[0]).into_shape_with_order(IxDyn(shape)).unwrap()
            }

            Op::Sum => {
                let v = get(&vals, node.inputs[0]).sum();
                ArrayD::from_elem(IxDyn(&[]), v)
            }

            Op::SumAxis(axis) => {
                get(&vals, node.inputs[0]).sum_axis(ndarray::Axis(*axis))
            }

            Op::Softmax => {
                let a = get(&vals, node.inputs[0]);
                let max = a.fold(f64::NEG_INFINITY, |m, &x| m.max(x));
                let e = a.mapv(|x| (x - max).exp());
                let s = e.sum();
                e / s
            }

            Op::CrossEntropy => {
                // cross_entropy(logits, labels): sum(-labels * log(softmax(logits)))
                let logits = get(&vals, node.inputs[0]);
                let labels = get(&vals, node.inputs[1]);
                let max = logits.fold(f64::NEG_INFINITY, |m, &x| m.max(x));
                let e = logits.mapv(|x| (x - max).exp());
                let s = e.sum();
                let log_softmax = logits.mapv(|x| x - max - s.ln());
                let loss = (-&labels * &log_softmax).sum();
                ArrayD::from_elem(IxDyn(&[]), loss)
            }

            Op::Conv2d { .. } | Op::MaxPool { .. } | Op::BatchNorm => {
                unimplemented!("Phase 7 op: {:?}", node.op)
            }
        };
        vals[id] = Some(v);
    }

    vals.into_iter().map(|v| v.expect("node not evaluated")).collect()
}

fn get(vals: &[Option<Tensor>], id: usize) -> Tensor {
    vals[id].clone().expect("dependency not evaluated")
}

pub(crate) fn matmul_op(a: Tensor, b: Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    assert_eq!(b.ndim(), 2);
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut out = ArrayD::zeros(IxDyn(&[m, n]));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for p in 0..k {
                sum += a[IxDyn(&[i, p])] * b[IxDyn(&[p, j])];
            }
            out[IxDyn(&[i, j])] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Graph;
    use ndarray::array;

    #[test]
    fn add_scalars() {
        let mut g = Graph::default();
        let a = g.input(vec![]);
        let b = g.input(vec![]);
        let c = g.add(a, b);
        let vals = eval(&g, &[
            ArrayD::from_elem(IxDyn(&[]), 3.0),
            ArrayD::from_elem(IxDyn(&[]), 4.0),
        ]);
        assert_eq!(vals[c][[]], 7.0);
    }

    #[test]
    fn matmul_2x2() {
        let mut g = Graph::default();
        let a = g.input(vec![2, 2]);
        let b = g.input(vec![2, 2]);
        let c = g.matmul(a, b);
        let av = array![[1.0_f64, 2.0], [3.0, 4.0]].into_dyn();
        let bv = array![[5.0_f64, 6.0], [7.0, 8.0]].into_dyn();
        let vals = eval(&g, &[av, bv]);
        assert_eq!(vals[c][[0, 0]], 19.0); // 1*5 + 2*7
        assert_eq!(vals[c][[1, 1]], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn relu_clamps_negatives() {
        let mut g = Graph::default();
        let a = g.input(vec![4]);
        let r = g.relu(a);
        let av = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -1.0, 0.0, 3.0]).unwrap();
        let vals = eval(&g, &[av]);
        assert_eq!(vals[r][[0]], 0.0);
        assert_eq!(vals[r][[3]], 3.0);
    }

    #[test]
    fn sum_reduces_to_scalar() {
        let mut g = Graph::default();
        let a = g.input(vec![3]);
        let s = g.sum(a);
        let av = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let vals = eval(&g, &[av]);
        assert_eq!(vals[s][[]], 6.0);
    }

    #[test]
    fn broadcast_bias_add() {
        // [2,3] + [3]  → bias broadcast across rows (the NN bias-add case)
        let mut g = Graph::default();
        let a = g.input(vec![2, 3]);
        let b = g.input(vec![3]);
        let c = g.add(a, b);
        assert_eq!(g.nodes[c].shape, vec![2, 3]);
        let av = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let bv = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10., 20., 30.]).unwrap();
        let vals = eval(&g, &[av, bv]);
        assert_eq!(vals[c][[0, 0]], 11.0);
        assert_eq!(vals[c][[1, 2]], 36.0);
    }

    #[test]
    fn broadcast_column_vector() {
        // [2,1] * [2,3]  → column broadcast across columns
        let mut g = Graph::default();
        let a = g.input(vec![2, 1]);
        let b = g.input(vec![2, 3]);
        let c = g.mul(a, b);
        assert_eq!(g.nodes[c].shape, vec![2, 3]);
        let av = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2., 3.]).unwrap();
        let bv = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1., 1., 1., 1., 1., 1.]).unwrap();
        let vals = eval(&g, &[av, bv]);
        assert_eq!(vals[c][[0, 1]], 2.0);
        assert_eq!(vals[c][[1, 2]], 3.0);
    }

    #[test]
    fn loss_example() {
        // loss(W, x, y) = sum((relu(W·x) - y)²)
        // W: 2×2, x: 2×1, y: 2×1
        let mut g = Graph::default();
        let w = g.input(vec![2, 2]);
        let x = g.input(vec![2, 1]);
        let y = g.input(vec![2, 1]);
        let wx   = g.matmul(w, x);
        let act  = g.relu(wx);
        let diff = g.sub(act, y);
        let sq   = g.mul(diff.clone(), diff);
        let loss = g.sum(sq);

        let wv = array![[1.0_f64, 0.0], [0.0, 1.0]].into_dyn();
        let xv = array![[2.0_f64], [3.0]].into_dyn();
        let yv = array![[1.0_f64], [1.0]].into_dyn();

        let vals = eval(&g, &[wv, xv, yv]);
        // relu(Ix) = [2,3], diff = [1,2], sq = [1,4], sum = 5
        assert_eq!(vals[loss][[]], 5.0);
    }
}
