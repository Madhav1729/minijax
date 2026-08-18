// Reverse-mode automatic differentiation.
// grad(g, output, wrt) appends new nodes to g and returns the NodeId
// of d(output)/d(wrt[i]) for each i.

use crate::ir::{Graph, NodeId, Op};

pub fn grad(g: &mut Graph, output: NodeId, wrt: &[NodeId]) -> Vec<NodeId> {
    let n = g.nodes.len();
    let mut adj: Vec<Option<NodeId>> = vec![None; n];

    // seed: d(output)/d(output) = 1
    let one = g.constant(1.0, g.nodes[output].shape.clone());
    adj[output] = Some(one);

    // reverse topological order — arena ordering means id n-1 down to 0
    for id in (0..n).rev() {
        let gy = match adj[id] { Some(g) => g, None => continue };
        let node = g.nodes[id].clone();

        match node.op {
            Op::Add => {
                accumulate(g, &mut adj, node.inputs[0], gy);
                accumulate(g, &mut adj, node.inputs[1], gy);
            }
            Op::Sub => {
                accumulate(g, &mut adj, node.inputs[0], gy);
                let neg = g.neg(gy);
                accumulate(g, &mut adj, node.inputs[1], neg);
            }
            Op::Mul => {
                let (a, b) = (node.inputs[0], node.inputs[1]);
                let ga = g.mul(gy, b);
                accumulate(g, &mut adj, a, ga);
                let gb = g.mul(gy, a);
                accumulate(g, &mut adj, b, gb);
            }
            Op::Div => {
                // d/da (a/b) = 1/b,  d/db (a/b) = -a/b²
                let (a, b) = (node.inputs[0], node.inputs[1]);
                let ga = g.div(gy, b);
                accumulate(g, &mut adj, a, ga);
                let b2  = g.mul(b, b);
                let neg_a = g.neg(a);
                let tmp  = g.div(neg_a, b2);
                let gb   = g.mul(gy, tmp);
                accumulate(g, &mut adj, b, gb);
            }
            Op::Neg => {
                let ga = g.neg(gy);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Relu => {
                // d/dx relu(x) = step(x)
                let mask = g.step(node.inputs[0]);
                let ga   = g.mul(gy, mask);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Exp => {
                // d/dx exp(x) = exp(x) = output of this node
                let ga = g.mul(gy, id);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Log => {
                // d/dx log(x) = 1/x
                let ga_raw = g.div(gy, node.inputs[0]);
                accumulate(g, &mut adj, node.inputs[0], ga_raw);
            }
            Op::Tanh => {
                // d/dx tanh(x) = 1 - tanh(x)²
                let tanh_sq = g.mul(id, id);
                let one = g.constant(1.0, g.nodes[id].shape.clone());
                let dtanh = g.sub(one, tanh_sq);
                let ga = g.mul(gy, dtanh);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Sigmoid => {
                // d/dx σ(x) = σ(x)·(1 - σ(x))
                let one   = g.constant(1.0, g.nodes[id].shape.clone());
                let one_m = g.sub(one, id);
                let ds    = g.mul(id, one_m);
                let ga    = g.mul(gy, ds);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::MatMul => {
                // C = A·B  →  dA = dC·Bᵀ,  dB = Aᵀ·dC
                let (a, b) = (node.inputs[0], node.inputs[1]);
                let bt  = g.transpose(b);
                let ga  = g.matmul(gy, bt);
                accumulate(g, &mut adj, a, ga);
                let at  = g.transpose(a);
                let gb  = g.matmul(at, gy);
                accumulate(g, &mut adj, b, gb);
            }
            Op::Transpose => {
                let ga = g.transpose(gy);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Sum => {
                // d/dx sum(x) = broadcast(gy) to input shape
                let in_shape = g.nodes[node.inputs[0]].shape.clone();
                let ga = g.broadcast(gy, in_shape);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::SumAxis(axis) => {
                // re-insert the reduced axis and broadcast
                let in_shape = g.nodes[node.inputs[0]].shape.clone();
                let mut exp_shape = g.nodes[gy].shape.clone();
                exp_shape.insert(axis, 1);
                let expanded = g.reshape(gy, exp_shape);
                let ga = g.broadcast(expanded, in_shape);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Broadcast(_) => {
                // grad of a broadcast = sum the upstream grad back down to the
                // input shape, reducing over exactly the broadcasted axes.
                let in_shape = g.nodes[node.inputs[0]].shape.clone();
                let ga = sum_to(g, gy, &in_shape);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }
            Op::Reshape(_) => {
                let in_shape = g.nodes[node.inputs[0]].shape.clone();
                let ga = g.reshape(gy, in_shape);
                accumulate(g, &mut adj, node.inputs[0], ga);
            }

            // Leaves — no backward pass needed
            Op::Input(_) | Op::Const(_) | Op::Step => {}

            // Phase 7+
            Op::Softmax | Op::CrossEntropy | Op::Conv2d { .. }
            | Op::MaxPool { .. } | Op::BatchNorm => {
                unimplemented!("backward for {:?} — Phase 7", node.op)
            }
        }
    }

    wrt.iter()
        .map(|&w| adj[w].expect("no gradient path to requested input"))
        .collect()
}

/// Reduce `node` down to `target` shape by summing over the axes that numpy
/// broadcasting would have expanded: first the leading (prepended) axes, then
/// any axis that was size-1 in `target` but larger after broadcast.
fn sum_to(g: &mut Graph, mut node: NodeId, target: &[usize]) -> NodeId {
    let cur = g.nodes[node].shape.clone();
    // 1. collapse leading prepended axes (reduces ndim down to target's)
    let extra = cur.len() - target.len();
    for _ in 0..extra {
        node = g.sum_axis(node, 0);
    }
    // 2. sum-with-keepdim over axes that were size-1 in target but expanded
    let cur2 = g.nodes[node].shape.clone();
    for i in 0..target.len() {
        if target[i] == 1 && cur2[i] != 1 {
            node = g.sum_axis(node, i); // removes axis i
            let mut s = g.nodes[node].shape.clone();
            s.insert(i, 1); // reinsert it as size-1
            node = g.reshape(node, s);
        }
    }
    node
}

fn accumulate(g: &mut Graph, adj: &mut Vec<Option<NodeId>>, id: NodeId, contrib: NodeId) {
    if id >= adj.len() { adj.resize(id + 1, None); }
    adj[id] = Some(match adj[id] {
        Some(prev) => g.add(prev, contrib),
        None       => contrib,
    });
}

// ── Forward-mode AD (JVP) ──────────────────────────────────────────────────
// Propagates a tangent (directional derivative) alongside the primal value.
// Given seed tangents for each Input, returns the tangent of every node —
// i.e. the directional derivative of the whole graph along the seed direction.
// This is an independent implementation of the chain rule and is used as a
// cross-check against reverse mode: for a scalar output f, <∇f, v> must equal
// the forward directional derivative Df·v.

use crate::interp::{self, Tensor};
use ndarray::{ArrayD, Axis, IxDyn};

fn bcast(a: &Tensor, shape: &[usize]) -> Tensor {
    if a.ndim() == 0 {
        ArrayD::from_elem(IxDyn(shape), *a.first().unwrap())
    } else {
        a.broadcast(IxDyn(shape)).expect("tangent broadcast failed").to_owned()
    }
}

/// Forward-mode: returns (primals, tangents), each indexed by NodeId.
pub fn jvp(g: &Graph, inputs: &[Tensor], seeds: &[Tensor]) -> (Vec<Tensor>, Vec<Tensor>) {
    assert_eq!(seeds.len(), inputs.len(), "one tangent seed per input");
    let primals = interp::eval(g, inputs);
    let mut tan: Vec<Option<Tensor>> = vec![None; g.nodes.len()];
    let t = |tan: &[Option<Tensor>], id: NodeId| tan[id].clone().expect("tangent missing");
    let p = |id: NodeId| primals[id].clone();

    for id in 0..g.nodes.len() {
        let node = &g.nodes[id];
        let dv = match &node.op {
            Op::Input(i) => seeds[*i].clone(),
            Op::Const(_) => ArrayD::zeros(IxDyn(&node.shape)),

            Op::Add => t(&tan, node.inputs[0]) + t(&tan, node.inputs[1]),
            Op::Sub => t(&tan, node.inputs[0]) - t(&tan, node.inputs[1]),
            Op::Mul => {
                let (a, b) = (node.inputs[0], node.inputs[1]);
                t(&tan, a) * p(b) + p(a) * t(&tan, b)
            }
            Op::Div => {
                let (a, b) = (node.inputs[0], node.inputs[1]);
                let (ta, tb) = (t(&tan, a), t(&tan, b));
                &ta / &p(b) - &p(a) * &tb / (&p(b) * &p(b))
            }
            Op::Neg => -t(&tan, node.inputs[0]),
            Op::Relu => {
                let mask = p(node.inputs[0]).mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                mask * t(&tan, node.inputs[0])
            }
            Op::Step => ArrayD::zeros(IxDyn(&node.shape)),
            Op::Exp => p(id) * t(&tan, node.inputs[0]),
            Op::Log => t(&tan, node.inputs[0]) / p(node.inputs[0]),
            Op::Tanh => {
                let o = p(id);
                (1.0 - &o * &o) * t(&tan, node.inputs[0])
            }
            Op::Sigmoid => {
                let o = p(id);
                (&o * (1.0 - &o)) * t(&tan, node.inputs[0])
            }
            Op::MatMul => {
                let (a, b) = (node.inputs[0], node.inputs[1]);
                interp::matmul_op(t(&tan, a), p(b)) + interp::matmul_op(p(a), t(&tan, b))
            }
            Op::Transpose => t(&tan, node.inputs[0]).t().to_owned(),
            Op::Broadcast(shape) => bcast(&t(&tan, node.inputs[0]), shape),
            Op::Reshape(shape) => {
                t(&tan, node.inputs[0]).into_shape_with_order(IxDyn(shape)).unwrap()
            }
            Op::Sum => {
                let s = t(&tan, node.inputs[0]).sum();
                ArrayD::from_elem(IxDyn(&[]), s)
            }
            Op::SumAxis(axis) => t(&tan, node.inputs[0]).sum_axis(Axis(*axis)),

            Op::Softmax | Op::CrossEntropy | Op::Conv2d { .. }
            | Op::MaxPool { .. } | Op::BatchNorm => {
                unimplemented!("forward-mode for {:?}", node.op)
            }
        };
        tan[id] = Some(dv);
    }

    (primals, tan.into_iter().map(|v| v.unwrap()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::eval;
    use ndarray::{ArrayD, IxDyn};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn randn(shape: &[usize], rng: &mut StdRng) -> Tensor {
        let n: usize = shape.iter().product::<usize>().max(1);
        let data: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
    }

    /// Central-difference numeric gradient of a scalar `output` wrt each input.
    fn numeric_grads<F>(build: &F, inputs: &[Tensor]) -> Vec<Tensor>
    where F: Fn(&mut Graph) -> NodeId {
        let eps = 1e-6;
        let mut gg = Graph::default();
        let out = build(&mut gg);
        let eval_at = |ins: &[Tensor]| {
            let vals = eval(&gg, ins);
            *vals[out].first().unwrap()
        };
        let mut grads = Vec::new();
        for (ii, inp) in inputs.iter().enumerate() {
            let n = inp.len();
            let mut deriv = vec![0.0; n.max(1)];
            for k in 0..n {
                let mut plus = inputs.to_vec();
                let mut minus = inputs.to_vec();
                *plus[ii].iter_mut().nth(k).unwrap() += eps;
                *minus[ii].iter_mut().nth(k).unwrap() -= eps;
                deriv[k] = (eval_at(&plus) - eval_at(&minus)) / (2.0 * eps);
            }
            grads.push(ArrayD::from_shape_vec(inp.raw_dim(), deriv).unwrap());
        }
        grads
    }

    /// Reverse-mode analytic gradient of a scalar `output` wrt every input.
    fn analytic_grads<F>(build: &F, inputs: &[Tensor]) -> Vec<Tensor>
    where F: Fn(&mut Graph) -> NodeId {
        let mut g = Graph::default();
        let out = build(&mut g);
        let wrt = g.inputs.clone();
        let grad_ids = grad(&mut g, out, &wrt);
        let vals = eval(&g, inputs);
        grad_ids.iter().map(|&id| vals[id].clone()).collect()
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f64, label: &str) {
        assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch {:?} vs {:?}", a.shape(), b.shape());
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            let rel = d / (x.abs().max(y.abs()).max(1e-8));
            assert!(d < tol || rel < 1e-4, "{label}: {x} vs {y} (abs {d})");
        }
    }

    fn check<F>(build: F, inputs: Vec<Tensor>)
    where F: Fn(&mut Graph) -> NodeId {
        let analytic = analytic_grads(&build, &inputs);
        let numeric = numeric_grads(&build, &inputs);
        for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
            assert_close(a, n, 1e-4, &format!("input {i}"));
        }
    }

    // ── per-op gradient checks (all wrap the op in a final `sum` → scalar) ──
    #[test]
    fn grad_add() {
        let mut r = StdRng::seed_from_u64(1);
        check(|g| { let a=g.input(vec![2,3]); let b=g.input(vec![2,3]); let c=g.add(a,b); g.sum(c) },
              vec![randn(&[2,3], &mut r), randn(&[2,3], &mut r)]);
    }
    #[test]
    fn grad_sub_mul_div() {
        let mut r = StdRng::seed_from_u64(2);
        check(|g| { let a=g.input(vec![3]); let b=g.input(vec![3]); let s=g.sub(a,b); let m=g.mul(s,a); g.sum(m) },
              vec![randn(&[3], &mut r), randn(&[3], &mut r)]);
        // div: keep denominator away from 0
        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0,2.0,3.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0,4.0,5.0]).unwrap();
        check(|g| { let a=g.input(vec![3]); let b=g.input(vec![3]); let d=g.div(a,b); g.sum(d) }, vec![a,b]);
    }
    #[test]
    fn grad_relu() {
        let mut r = StdRng::seed_from_u64(3);
        check(|g| { let a=g.input(vec![5]); let x=g.relu(a); g.sum(x) }, vec![randn(&[5], &mut r)]);
    }
    #[test]
    fn grad_exp_log() {
        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5,1.0,1.5]).unwrap();
        check(|g| { let a=g.input(vec![3]); let e=g.exp(a); g.sum(e) }, vec![a.clone()]);
        check(|g| { let a=g.input(vec![3]); let l=g.log(a); g.sum(l) }, vec![a]);
    }
    #[test]
    fn grad_tanh_sigmoid() {
        let mut r = StdRng::seed_from_u64(4);
        check(|g| { let a=g.input(vec![4]); let t=g.tanh(a); g.sum(t) }, vec![randn(&[4], &mut r)]);
        check(|g| { let a=g.input(vec![4]); let s=g.sigmoid(a); g.sum(s) }, vec![randn(&[4], &mut r)]);
    }
    #[test]
    fn grad_matmul() {
        let mut r = StdRng::seed_from_u64(5);
        check(|g| { let a=g.input(vec![2,3]); let b=g.input(vec![3,4]); let c=g.matmul(a,b); g.sum(c) },
              vec![randn(&[2,3], &mut r), randn(&[3,4], &mut r)]);
    }
    #[test]
    fn grad_transpose_sumaxis() {
        let mut r = StdRng::seed_from_u64(6);
        check(|g| { let a=g.input(vec![2,3]); let t=g.transpose(a); let s=g.sum_axis(t,1); g.sum(s) },
              vec![randn(&[2,3], &mut r)]);
    }
    #[test]
    fn grad_broadcast_bias() {
        let mut r = StdRng::seed_from_u64(7);
        // [2,3] + bias[3] then sum — exercises broadcast-grad (sum_to)
        check(|g| { let a=g.input(vec![2,3]); let b=g.input(vec![3]); let c=g.add(a,b); g.sum(c) },
              vec![randn(&[2,3], &mut r), randn(&[3], &mut r)]);
    }

    #[test]
    fn grad_full_loss() {
        // loss(W,x,y) = sum((relu(W·x) - y)²)
        let mut r = StdRng::seed_from_u64(8);
        check(|g| {
            let w=g.input(vec![3,3]); let x=g.input(vec![3,1]); let y=g.input(vec![3,1]);
            let wx=g.matmul(w,x); let a=g.relu(wx); let d=g.sub(a,y);
            let sq=g.mul(d,d); g.sum(sq)
        }, vec![randn(&[3,3], &mut r), randn(&[3,1], &mut r), randn(&[3,1], &mut r)]);
    }

    // ── forward-mode (JVP) as an independent cross-check of reverse-mode ──
    #[test]
    fn forward_matches_reverse() {
        let mut r = StdRng::seed_from_u64(9);
        let build = |g: &mut Graph| {
            let w=g.input(vec![3,3]); let x=g.input(vec![3,1]);
            let wx=g.matmul(w,x); let a=g.tanh(wx); g.sum(a)
        };
        let inputs = vec![randn(&[3,3], &mut r), randn(&[3,1], &mut r)];
        let seeds = vec![randn(&[3,3], &mut r), randn(&[3,1], &mut r)];

        // forward: directional derivative Df·v
        let mut g = Graph::default();
        let out = build(&mut g);
        let (_p, tan) = jvp(&g, &inputs, &seeds);
        let directional = *tan[out].first().unwrap();

        // reverse: <∇f, v> should equal the same directional derivative
        let grads = analytic_grads(&build, &inputs);
        let dot: f64 = grads.iter().zip(seeds.iter())
            .map(|(gr, se)| gr.iter().zip(se.iter()).map(|(a,b)| a*b).sum::<f64>())
            .sum();
        assert!((directional - dot).abs() < 1e-9, "fwd {directional} vs rev {dot}");
    }
}
