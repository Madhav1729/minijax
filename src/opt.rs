// Equality-saturation optimizer built on `egg`.
//
// Pipeline:  Graph → RecExpr<Mjx>  →  (equality saturation w/ rewrite rules)
//            →  extract cheapest by a FLOP-aware cost model  →  RecExpr → Graph.
//
// The two (de)serialization hops are the fiddly part: the e-graph is untyped, so
// shapes that are *free parameters* of a node (Input/Const/Broadcast/Reshape
// targets, SumAxis axis) are encoded as explicit child nodes, while shapes of
// *computed* nodes (add, matmul, …) are re-inferred on the way back out by
// replaying the Graph builder methods (which already do shape inference).

use std::fmt;
use std::str::FromStr;

use egg::{
    define_language, rewrite, Analysis, DidMerge, EGraph, Extractor, Id, Language, Pattern,
    RecExpr, Rewrite, Runner, Subst, Var,
};

use crate::ir::{Graph, NodeId, Op};

// ── float leaf that is Ord + Hash (egg requires it) ─────────────────────────
// Stored as raw bits so it can be Eq/Ord/Hash; the ordering is meaningless but
// consistent, which is all egg needs for canonicalization. Displayed with a
// trailing `f` so a float token never collides with an integer `Num` token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrdF64(u64);
impl OrdF64 {
    pub fn new(v: f64) -> Self { OrdF64(v.to_bits()) }
    pub fn get(self) -> f64 { f64::from_bits(self.0) }
}
impl fmt::Display for OrdF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}f", self.get())
    }
}
impl FromStr for OrdF64 {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        s.strip_suffix('f')
            .ok_or_else(|| "no f suffix".to_string())?
            .parse::<f64>()
            .map(OrdF64::new)
            .map_err(|e| e.to_string())
    }
}

define_language! {
    /// egg mirror of the IR. Metadata (indices, axes, shape dims, scalar values)
    /// live as dedicated leaf nodes so the whole program is a single RecExpr.
    pub enum Mjx {
        Num(i64),            // integer leaf: input index, axis, or a shape dim
        Scalar(OrdF64),      // float leaf: a constant's value

        "shape"  = Shape(Box<[Id]>),   // children are Num dims (variadic)

        "input"     = Input([Id; 2]),      // [Num(index), shape]
        "const"     = Const([Id; 2]),      // [Scalar(value), shape]
        "broadcast" = Broadcast([Id; 2]),  // [arg, shape]
        "reshape"   = Reshape([Id; 2]),    // [arg, shape]
        "sumaxis"   = SumAxis([Id; 2]),    // [arg, Num(axis)]

        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "matmul" = MatMul([Id; 2]),

        "neg"  = Neg([Id; 1]),
        "relu" = Relu([Id; 1]),
        "step" = Step([Id; 1]),
        "exp"  = Exp([Id; 1]),
        "log"  = Log([Id; 1]),
        "tanh" = Tanh([Id; 1]),
        "sigmoid" = Sigmoid([Id; 1]),
        "transpose" = Transpose([Id; 1]),
        "sum"  = Sum([Id; 1]),
    }
}

// ── shape analysis: every e-class carries the shape it evaluates to ──────────
// For metadata leaves the "shape" field is repurposed: Num carries [value],
// Shape carries the dim list itself. This lets parents read their free-shape
// operands straight out of the analysis.
#[derive(Default)]
pub struct ShapeAnalysis;

impl Analysis<Mjx> for ShapeAnalysis {
    type Data = Vec<usize>;

    fn make(egraph: &EGraph<Mjx, Self>, enode: &Mjx) -> Vec<usize> {
        let g = |id: Id| egraph[id].data.clone();
        let d0 = |id: Id| egraph[id].data.first().copied().unwrap_or(0);
        match enode {
            Mjx::Num(n) => vec![*n as usize],
            Mjx::Scalar(_) => vec![],
            Mjx::Shape(dims) => dims.iter().map(|&d| d0(d)).collect(),

            Mjx::Input([_, s]) | Mjx::Const([_, s]) | Mjx::Broadcast([_, s]) | Mjx::Reshape([_, s]) => g(*s),
            Mjx::SumAxis([a, ax]) => {
                let mut sh = g(*a);
                let axis = d0(*ax);
                if axis < sh.len() { sh.remove(axis); }
                sh
            }
            Mjx::Sum(_) => vec![],
            Mjx::Transpose([a]) => {
                let mut sh = g(*a);
                sh.reverse();
                sh
            }
            Mjx::MatMul([a, b]) => {
                let (sa, sb) = (g(*a), g(*b));
                if sa.len() == 2 && sb.len() == 2 { vec![sa[0], sb[1]] } else { sa }
            }
            Mjx::Add([a, _]) | Mjx::Sub([a, _]) | Mjx::Mul([a, _]) | Mjx::Div([a, _]) => g(*a),
            Mjx::Neg([a]) | Mjx::Relu([a]) | Mjx::Step([a]) | Mjx::Exp([a]) | Mjx::Log([a])
            | Mjx::Tanh([a]) | Mjx::Sigmoid([a]) => g(*a),
        }
    }

    fn merge(&mut self, a: &mut Vec<usize>, b: Vec<usize>) -> DidMerge {
        // Equivalent e-nodes must share a shape; keep `a`, prefer a non-empty one.
        if a.is_empty() && !b.is_empty() {
            *a = b;
            DidMerge(true, false)
        } else {
            DidMerge(false, false)
        }
    }
}

fn numel(shape: &[usize]) -> f64 {
    shape.iter().product::<usize>().max(1) as f64
}

// ── FLOP/kernel cost model ──────────────────────────────────────────────────
// Cost = this node's estimated FLOPs (shape-aware) + sum of children costs.
// matmul dominates, so matmul-reassociation is chosen by real arithmetic cost.
struct FlopCost<'a> {
    egraph: &'a EGraph<Mjx, ShapeAnalysis>,
}
impl egg::CostFunction<Mjx> for FlopCost<'_> {
    type Cost = f64;
    fn cost<C: FnMut(Id) -> f64>(&mut self, enode: &Mjx, mut costs: C) -> f64 {
        let shape = |id: Id| self.egraph[id].data.clone();
        let here = match enode {
            Mjx::MatMul([a, b]) => {
                let (sa, sb) = (shape(*a), shape(*b));
                if sa.len() == 2 && sb.len() == 2 {
                    2.0 * (sa[0] * sa[1] * sb[1]) as f64
                } else {
                    1.0
                }
            }
            Mjx::Add([a, _]) | Mjx::Sub([a, _]) | Mjx::Mul([a, _]) | Mjx::Div([a, _]) => numel(&shape(*a)),
            Mjx::Neg([a]) | Mjx::Relu([a]) | Mjx::Step([a]) | Mjx::Exp([a]) | Mjx::Log([a])
            | Mjx::Tanh([a]) | Mjx::Sigmoid([a]) => numel(&shape(*a)),
            Mjx::Sum([a]) | Mjx::SumAxis([a, _]) => numel(&shape(*a)),
            Mjx::Transpose([a]) | Mjx::Broadcast([a, _]) | Mjx::Reshape([a, _]) => numel(&shape(*a)),
            // leaves and metadata are free
            _ => 0.0,
        };
        enode.fold(here, |sum, id| sum + costs(id))
    }
}

// ── rewrite rules ───────────────────────────────────────────────────────────
fn var(s: &str) -> Var { s.parse().unwrap() }

pub fn rules() -> Vec<Rewrite<Mjx, ShapeAnalysis>> {
    let mut rs: Vec<Rewrite<Mjx, ShapeAnalysis>> = vec![
        // lift a broadcasted constant to a constant of the wider shape
        rewrite!("const-lift"; "(broadcast (const ?v ?s1) ?s2)" => "(const ?v ?s2)"),
        // commutativity
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        // associativity (structural — enables reassociation choices)
        rewrite!("add-assoc"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        rewrite!("mul-assoc"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),
        // matmul reassociation — the FLOP-relevant (and float-unsound) rewrite
        rewrite!("matmul-assoc"; "(matmul (matmul ?a ?b) ?c)" => "(matmul ?a (matmul ?b ?c))"),
        rewrite!("matmul-assoc-rev"; "(matmul ?a (matmul ?b ?c))" => "(matmul (matmul ?a ?b) ?c)"),
        // involutions
        rewrite!("transpose-twice"; "(transpose (transpose ?a))" => "?a"),
        rewrite!("neg-twice"; "(neg (neg ?a))" => "?a"),
        rewrite!("relu-idem"; "(relu (relu ?a))" => "(relu ?a)"),
        // sum linearity (kernel fusion: two reductions → one)
        rewrite!("sum-of-add"; "(sum (add ?a ?b))" => "(add (sum ?a) (sum ?b))"),
    ];
    // identity/annihilator rules gated on a constant operand
    let add_pat: Pattern<Mjx> = "(add ?a ?b)".parse().unwrap();
    let mul_pat: Pattern<Mjx> = "(mul ?a ?b)".parse().unwrap();
    rs.push(Rewrite::new("add-zero", add_pat, ConstApplier { keep: var("?a"), which: var("?b"), cond: 0.0 }).unwrap());
    rs.push(Rewrite::new("mul-one", mul_pat.clone(), ConstApplier { keep: var("?a"), which: var("?b"), cond: 1.0 }).unwrap());
    rs.push(Rewrite::new("mul-zero", mul_pat, ConstApplier { keep: var("?b"), which: var("?b"), cond: 0.0 }).unwrap());
    rs
}

// Applier that unions the eclass with `keep` only when `which` is a const == `cond`.
struct ConstApplier {
    keep: Var,
    which: Var,
    cond: f64,
}
impl egg::Applier<Mjx, ShapeAnalysis> for ConstApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Mjx, ShapeAnalysis>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&egg::PatternAst<Mjx>>,
        _rule_name: egg::Symbol,
    ) -> Vec<Id> {
        let w = subst[self.which];
        let is = egraph[w].nodes.iter().any(|n| match n {
            Mjx::Const([sv, _]) => egraph[*sv]
                .nodes
                .iter()
                .any(|s| matches!(s, Mjx::Scalar(o) if o.get() == self.cond)),
            _ => false,
        });
        if !is {
            return vec![];
        }
        let k = subst[self.keep];
        if egraph.union(eclass, k) {
            vec![eclass]
        } else {
            vec![]
        }
    }
}

// ── Graph → RecExpr ─────────────────────────────────────────────────────────
fn push_shape(expr: &mut RecExpr<Mjx>, shape: &[usize]) -> Id {
    let dims: Vec<Id> = shape.iter().map(|&d| expr.add(Mjx::Num(d as i64))).collect();
    expr.add(Mjx::Shape(dims.into_boxed_slice()))
}

pub fn to_recexpr(g: &Graph, output: NodeId) -> (RecExpr<Mjx>, Id) {
    let mut expr = RecExpr::default();
    let mut map: Vec<Id> = vec![Id::from(0); g.nodes.len()];
    for (id, node) in g.nodes.iter().enumerate() {
        let c = |i: usize| map[node.inputs[i]];
        let e = match &node.op {
            Op::Input(idx) => {
                let n = expr.add(Mjx::Num(*idx as i64));
                let s = push_shape(&mut expr, &node.shape);
                Mjx::Input([n, s])
            }
            Op::Const(v) => {
                let sv = expr.add(Mjx::Scalar(OrdF64::new(*v)));
                let s = push_shape(&mut expr, &node.shape);
                Mjx::Const([sv, s])
            }
            Op::Broadcast(shape) => {
                let s = push_shape(&mut expr, shape);
                Mjx::Broadcast([c(0), s])
            }
            Op::Reshape(shape) => {
                let s = push_shape(&mut expr, shape);
                Mjx::Reshape([c(0), s])
            }
            Op::SumAxis(axis) => {
                let ax = expr.add(Mjx::Num(*axis as i64));
                Mjx::SumAxis([c(0), ax])
            }
            Op::Add => Mjx::Add([c(0), c(1)]),
            Op::Sub => Mjx::Sub([c(0), c(1)]),
            Op::Mul => Mjx::Mul([c(0), c(1)]),
            Op::Div => Mjx::Div([c(0), c(1)]),
            Op::MatMul => Mjx::MatMul([c(0), c(1)]),
            Op::Neg => Mjx::Neg([c(0)]),
            Op::Relu => Mjx::Relu([c(0)]),
            Op::Step => Mjx::Step([c(0)]),
            Op::Exp => Mjx::Exp([c(0)]),
            Op::Log => Mjx::Log([c(0)]),
            Op::Tanh => Mjx::Tanh([c(0)]),
            Op::Sigmoid => Mjx::Sigmoid([c(0)]),
            Op::Transpose => Mjx::Transpose([c(0)]),
            Op::Sum => Mjx::Sum([c(0)]),
            other => panic!("op not supported by optimizer yet: {:?}", other),
        };
        map[id] = expr.add(e);
    }
    (expr, map[output])
}

// ── RecExpr → Graph ─────────────────────────────────────────────────────────
fn read_shape(expr: &RecExpr<Mjx>, id: Id) -> Vec<usize> {
    match &expr[id] {
        Mjx::Shape(dims) => dims.iter().map(|&d| read_num(expr, d) as usize).collect(),
        _ => panic!("expected shape node"),
    }
}
fn read_num(expr: &RecExpr<Mjx>, id: Id) -> i64 {
    match &expr[id] {
        Mjx::Num(n) => *n,
        _ => panic!("expected num node"),
    }
}
fn read_scalar(expr: &RecExpr<Mjx>, id: Id) -> f64 {
    match &expr[id] {
        Mjx::Scalar(o) => o.get(),
        _ => panic!("expected scalar node"),
    }
}

/// Rebuild a Graph from an extracted expression. `num_inputs` fixes the input
/// arity so the result stays call-compatible with the original input slice.
pub fn from_recexpr(expr: &RecExpr<Mjx>, root: Id, num_inputs: usize) -> (Graph, NodeId) {
    let mut g = Graph::default();
    g.inputs = vec![usize::MAX; num_inputs]; // placeholder slots for unused inputs
    let nodes = expr.as_ref();
    let mut map: Vec<Option<NodeId>> = vec![None; nodes.len()];

    for (i, enode) in nodes.iter().enumerate() {
        let ch = |k: usize| map[usize::from(enode.children()[k])].expect("child not built");
        let nid = match enode {
            // metadata leaves don't become graph nodes
            Mjx::Num(_) | Mjx::Scalar(_) | Mjx::Shape(_) => None,

            Mjx::Input([nidx, s]) => {
                let idx = read_num(expr, *nidx) as usize;
                let shape = read_shape(expr, *s);
                let id = g.push(Op::Input(idx), vec![], shape);
                if idx < g.inputs.len() {
                    g.inputs[idx] = id;
                }
                Some(id)
            }
            Mjx::Const([sv, s]) => {
                let v = read_scalar(expr, *sv);
                let shape = read_shape(expr, *s);
                Some(g.constant(v, shape))
            }
            Mjx::Broadcast([_, s]) => {
                let shape = read_shape(expr, *s);
                Some(g.broadcast(ch(0), shape))
            }
            Mjx::Reshape([_, s]) => {
                let shape = read_shape(expr, *s);
                Some(g.reshape(ch(0), shape))
            }
            Mjx::SumAxis([_, ax]) => {
                let axis = read_num(expr, *ax) as usize;
                Some(g.sum_axis(ch(0), axis))
            }
            Mjx::Add([_, _]) => Some(g.add(ch(0), ch(1))),
            Mjx::Sub([_, _]) => Some(g.sub(ch(0), ch(1))),
            Mjx::Mul([_, _]) => Some(g.mul(ch(0), ch(1))),
            Mjx::Div([_, _]) => Some(g.div(ch(0), ch(1))),
            Mjx::MatMul([_, _]) => Some(g.matmul(ch(0), ch(1))),
            Mjx::Neg([_]) => Some(g.neg(ch(0))),
            Mjx::Relu([_]) => Some(g.relu(ch(0))),
            Mjx::Step([_]) => Some(g.step(ch(0))),
            Mjx::Exp([_]) => Some(g.exp(ch(0))),
            Mjx::Log([_]) => Some(g.log(ch(0))),
            Mjx::Tanh([_]) => Some(g.tanh(ch(0))),
            Mjx::Sigmoid([_]) => Some(g.sigmoid(ch(0))),
            Mjx::Transpose([_]) => Some(g.transpose(ch(0))),
            Mjx::Sum([_]) => Some(g.sum(ch(0))),
        };
        map[i] = nid;
    }

    let out = map[usize::from(root)].expect("root not built");
    (g, out)
}

// ── public entry point ──────────────────────────────────────────────────────
pub struct OptStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub cost_before: f64,
    pub cost_after: f64,
}

/// Run equality saturation and return the optimized graph rooted at `output`,
/// plus before/after node counts and modeled FLOP cost.
pub fn optimize(g: &Graph, output: NodeId) -> (Graph, NodeId, OptStats) {
    let (expr, _root) = to_recexpr(g, output);
    let cost_before = {
        let mut eg: EGraph<Mjx, ShapeAnalysis> = EGraph::default();
        let r = eg.add_expr(&expr);
        extract_from(&eg, r).1
    };

    let runner = Runner::default().with_expr(&expr).run(&rules());
    let egraph_root = runner.egraph.find(runner.roots[0]);
    let (best, cost_after) = extract_from(&runner.egraph, egraph_root);

    let num_inputs = g.num_inputs();
    let (og, oout) = from_recexpr(&best, Id::from(best.as_ref().len() - 1), num_inputs);

    let stats = OptStats {
        nodes_before: reachable_count(g, output),
        nodes_after: og.nodes.len(),
        cost_before,
        cost_after,
    };
    (og, oout, stats)
}

fn extract_from(egraph: &EGraph<Mjx, ShapeAnalysis>, root: Id) -> (RecExpr<Mjx>, f64) {
    let extractor = Extractor::new(egraph, FlopCost { egraph });
    let (cost, expr) = extractor.find_best(root);
    (expr, cost)
}

fn reachable_count(g: &Graph, output: NodeId) -> usize {
    let mut seen = vec![false; g.nodes.len()];
    let mut stack = vec![output];
    while let Some(id) = stack.pop() {
        if seen[id] {
            continue;
        }
        seen[id] = true;
        for &c in &g.nodes[id].inputs {
            stack.push(c);
        }
    }
    seen.iter().filter(|&&b| b).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::eval;
    use ndarray::{ArrayD, IxDyn};

    fn t(shape: &[usize], data: Vec<f64>) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
    }

    #[test]
    fn roundtrip_identity() {
        // Graph → RecExpr → Graph must evaluate identically (no rules applied).
        let mut g = Graph::default();
        let w = g.input(vec![2, 2]);
        let x = g.input(vec![2, 2]);
        let wx = g.matmul(w, x);
        let r = g.relu(wx);
        let out = g.sum(r);

        let (expr, root) = to_recexpr(&g, out);
        let (g2, out2) = from_recexpr(&expr, root, g.num_inputs());

        let wv = t(&[2, 2], vec![1., -2., 3., 4.]);
        let xv = t(&[2, 2], vec![0.5, 1., -1., 2.]);
        let a = eval(&g, &[wv.clone(), xv.clone()])[out].clone();
        let b = eval(&g2, &[wv, xv])[out2].clone();
        assert_eq!(a, b);
    }

    #[test]
    fn eliminates_add_zero_mul_one() {
        // (x*1) + 0  should collapse to x
        let mut g = Graph::default();
        let x = g.input(vec![3]);
        let one = g.constant(1.0, vec![]);
        let zero = g.constant(0.0, vec![]);
        let m = g.mul(x, one);
        let a = g.add(m, zero);
        let out = g.sum(a);

        let (og, oout, stats) = optimize(&g, out);
        assert!(
            stats.nodes_after < stats.nodes_before,
            "no reduction: {} -> {}",
            stats.nodes_before,
            stats.nodes_after
        );

        let xv = t(&[3], vec![2., 5., -1.]);
        let orig = eval(&g, &[xv.clone()])[out].clone();
        let opt = eval(&og, &[xv])[oout].clone();
        assert_eq!(orig, opt);
    }

    #[test]
    fn matmul_reassoc_by_flops() {
        // (A[1,64]·B[64,64])·C[64,1] costs ~2·1·64·64 + 2·1·64·1 FLOPs.
        // A·(B·C) costs 2·64·64·1 + 2·1·64·1 — cheaper. Optimizer must pick it.
        let mut g = Graph::default();
        let a = g.input(vec![1, 64]);
        let b = g.input(vec![64, 64]);
        let c = g.input(vec![64, 1]);
        let ab = g.matmul(a, b);
        let abc = g.matmul(ab, c);
        let out = g.sum(abc);

        let (og, oout, stats) = optimize(&g, out);
        assert!(stats.cost_after <= stats.cost_before, "cost went up");

        use rand::{Rng, SeedableRng};
        let mut r = rand::rngs::StdRng::seed_from_u64(0);
        let av = t(&[1, 64], (0..64).map(|_| r.gen_range(-1.0..1.0)).collect());
        let bv = t(&[64, 64], (0..64 * 64).map(|_| r.gen_range(-1.0..1.0)).collect());
        let cv = t(&[64, 1], (0..64).map(|_| r.gen_range(-1.0..1.0)).collect());
        let orig = eval(&g, &[av.clone(), bv.clone(), cv.clone()])[out].clone();
        let opt = eval(&og, &[av, bv, cv])[oout].clone();
        // reassociation changes float summation order slightly
        let (o, p) = (*orig.first().unwrap(), *opt.first().unwrap());
        assert!((o - p).abs() < 1e-9, "{o} vs {p}");
    }
}
