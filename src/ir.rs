// Computation graph IR — arena-based DAG.
// Each NodeId is an index into Graph::nodes; the arena is implicitly topologically
// ordered because nodes can only reference earlier indices.

pub type NodeId = usize;

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    // Leaves
    Input(usize),       // index into the runtime input slice
    Const(f64),         // scalar constant (broadcast to shape on eval)

    // Elementwise binary
    Add,
    Sub,
    Mul,
    Div,

    // Elementwise unary
    Neg,
    Relu,
    Step,               // Heaviside: 1 if x>0 else 0  (relu backward)
    Exp,
    Log,
    Tanh,
    Sigmoid,

    // Linear algebra
    MatMul,
    Transpose,          // 2-D transpose

    // Shape / broadcast
    Broadcast(Vec<usize>),   // broadcast input to given shape
    Reshape(Vec<usize>),
    Sum,                // reduce all elements to scalar
    SumAxis(usize),     // reduce along one axis

    // Placeholders — filled in later phases
    Softmax,
    CrossEntropy,       // inputs: logits, labels
    Conv2d { stride: usize, pad: usize },
    MaxPool { kernel: usize, stride: usize },
    BatchNorm,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,    // output shape; [] = scalar
}

#[derive(Default, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// NodeIds of the Input leaves, in declaration order. inputs[i] is the node
    /// for runtime input slot i. Avoids scanning the arena to count inputs.
    pub inputs: Vec<NodeId>,
}

impl Graph {
    // ── primitive push ────────────────────────────────────────────────────
    pub fn push(&mut self, op: Op, inputs: Vec<NodeId>, shape: Vec<usize>) -> NodeId {
        self.nodes.push(Node { op, inputs, shape });
        self.nodes.len() - 1
    }

    // ── leaves ────────────────────────────────────────────────────────────
    pub fn input(&mut self, shape: Vec<usize>) -> NodeId {
        let idx = self.inputs.len();
        let id = self.push(Op::Input(idx), vec![], shape);
        self.inputs.push(id);
        id
    }

    pub fn constant(&mut self, val: f64, shape: Vec<usize>) -> NodeId {
        self.push(Op::Const(val), vec![], shape)
    }

    // ── elementwise binary ────────────────────────────────────────────────
    // All binary ops follow numpy broadcasting: operands are broadcast to a
    // common shape via explicit Broadcast nodes (XLA/HLO style — the IR never
    // broadcasts implicitly), then the op runs on equal shapes.
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId { self.binop(Op::Add, a, b) }
    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId { self.binop(Op::Sub, a, b) }
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId { self.binop(Op::Mul, a, b) }
    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId { self.binop(Op::Div, a, b) }

    fn binop(&mut self, op: Op, a: NodeId, b: NodeId) -> NodeId {
        let out = broadcast_shapes(&self.nodes[a].shape, &self.nodes[b].shape);
        let a = self.broadcast_to(a, &out);
        let b = self.broadcast_to(b, &out);
        self.push(op, vec![a, b], out)
    }

    /// Insert a Broadcast node iff `a`'s shape differs from `target`.
    pub fn broadcast_to(&mut self, a: NodeId, target: &[usize]) -> NodeId {
        if self.nodes[a].shape == target {
            a
        } else {
            self.broadcast(a, target.to_vec())
        }
    }

    pub fn neg(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Neg, vec![a], shape)
    }

    // ── elementwise unary ─────────────────────────────────────────────────
    pub fn relu(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Relu, vec![a], shape)
    }

    pub fn step(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Step, vec![a], shape)
    }

    pub fn exp(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Exp, vec![a], shape)
    }

    pub fn log(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Log, vec![a], shape)
    }

    pub fn tanh(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Tanh, vec![a], shape)
    }

    pub fn sigmoid(&mut self, a: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        self.push(Op::Sigmoid, vec![a], shape)
    }

    // ── linear algebra ────────────────────────────────────────────────────
    /// matmul: a is [M,K], b is [K,N] → [M,N]
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let sa = &self.nodes[a].shape;
        let sb = &self.nodes[b].shape;
        assert_eq!(sa.len(), 2, "matmul expects 2-D inputs");
        assert_eq!(sb.len(), 2, "matmul expects 2-D inputs");
        assert_eq!(sa[1], sb[0], "matmul inner-dim mismatch: {}x{} vs {}x{}", sa[0], sa[1], sb[0], sb[1]);
        let shape = vec![sa[0], sb[1]];
        self.push(Op::MatMul, vec![a, b], shape)
    }

    pub fn transpose(&mut self, a: NodeId) -> NodeId {
        let sa = &self.nodes[a].shape;
        assert_eq!(sa.len(), 2, "transpose expects 2-D input");
        let shape = vec![sa[1], sa[0]];
        self.push(Op::Transpose, vec![a], shape)
    }

    // ── reductions ────────────────────────────────────────────────────────
    pub fn sum(&mut self, a: NodeId) -> NodeId {
        self.push(Op::Sum, vec![a], vec![]) // scalar
    }

    pub fn sum_axis(&mut self, a: NodeId, axis: usize) -> NodeId {
        let mut shape = self.nodes[a].shape.clone();
        shape.remove(axis);
        self.push(Op::SumAxis(axis), vec![a], shape)
    }

    // ── shape ─────────────────────────────────────────────────────────────
    pub fn broadcast(&mut self, a: NodeId, shape: Vec<usize>) -> NodeId {
        self.push(Op::Broadcast(shape.clone()), vec![a], shape)
    }

    pub fn reshape(&mut self, a: NodeId, shape: Vec<usize>) -> NodeId {
        self.push(Op::Reshape(shape.clone()), vec![a], shape)
    }

    // ── shape helpers ─────────────────────────────────────────────────────
    pub fn shape_of(&self, id: NodeId) -> &[usize] {
        &self.nodes[id].shape
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }
}

/// numpy broadcasting: align shapes from the right; each dim must be equal, or
/// one of them 1 (or missing). Result dim is the max. Panics on incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        // right-aligned index into each operand (1 if the dim is absent)
        let da = if i + a.len() < n { 1 } else { a[i + a.len() - n] };
        let db = if i + b.len() < n { 1 } else { b[i + b.len() - n] };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            panic!("incompatible broadcast shapes {:?} vs {:?}", a, b);
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_graph() {
        let mut g = Graph::default();
        let w = g.input(vec![2, 3]);
        let x = g.input(vec![3, 1]);
        let wx = g.matmul(w, x);
        assert_eq!(g.nodes[wx].shape, vec![2, 1]);
    }

    #[test]
    fn broadcast_shape_rules() {
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), vec![2, 3]); // prepend
        assert_eq!(broadcast_shapes(&[2, 1], &[2, 3]), vec![2, 3]); // stretch size-1
        assert_eq!(broadcast_shapes(&[], &[4]), vec![4]); // scalar
        assert_eq!(broadcast_shapes(&[5, 1, 4], &[3, 1]), vec![5, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn broadcast_incompatible_panics() {
        broadcast_shapes(&[3], &[4]); // 3 vs 4, neither is 1
    }

    #[test]
    #[should_panic]
    fn matmul_shape_mismatch_panics() {
        let mut g = Graph::default();
        let a = g.input(vec![2, 3]);
        let b = g.input(vec![4, 1]); // inner dim mismatch
        g.matmul(a, b);
    }
}
