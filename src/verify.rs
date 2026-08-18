// SMT verification of rewrite rules with Z3.
//
// Each rule carries a *proof obligation*: a pair of symbolic scalar expressions
// (lhs, rhs) that the rewrite claims are equal for all inputs. We discharge that
// obligation twice —
//   • over Z3's `Real` theory   (exact real arithmetic), and
//   • over Z3's IEEE-754 `Float` theory (64-bit, round-to-nearest-even) —
// by asserting the negation `lhs != rhs` and checking satisfiability:
//   UNSAT  ⇒ no counterexample exists ⇒ the rule is sound over that theory.
//   SAT    ⇒ Z3 produced a concrete counterexample ⇒ the rule is unsound.
//
// Rules split into three classes:
//   SoundEverywhere — sound over reals AND floats
//   RealsOnly       — sound over reals but NOT floats  (the "fast-math" rules)
//   Unsound         — not even sound over reals        (a genuine bug in the rule)
//
// Float equality here is Z3's structural `=` (bit-pattern equality, one NaN
// class). We additionally constrain every float variable to be *finite* via the
// identity `(a - a) == +0`, which holds iff `a` is neither NaN nor ±∞. So a SAT
// result is a real, finite counterexample — reassociation rounding, or the
// signed-zero subtlety that makes even `x + 0 → x` unsound under IEEE-754
// (exactly why LLVM gates it behind `-ffast-math`/`nsz`).

use z3::ast::{Ast, Float, Real, RoundingMode};
use z3::{SatResult, Solver};

// ── symbolic scalar expression (theory-agnostic) ────────────────────────────
#[derive(Clone, Debug)]
pub enum Sym {
    Var(usize),
    Const(f64),
    Neg(Box<Sym>),
    Add(Box<Sym>, Box<Sym>),
    Sub(Box<Sym>, Box<Sym>),
    Mul(Box<Sym>, Box<Sym>),
}

// tiny constructor helpers
pub fn v(i: usize) -> Sym { Sym::Var(i) }
pub fn c(x: f64) -> Sym { Sym::Const(x) }
pub fn add(a: Sym, b: Sym) -> Sym { Sym::Add(Box::new(a), Box::new(b)) }
pub fn sub(a: Sym, b: Sym) -> Sym { Sym::Sub(Box::new(a), Box::new(b)) }
pub fn mul(a: Sym, b: Sym) -> Sym { Sym::Mul(Box::new(a), Box::new(b)) }
pub fn neg(a: Sym) -> Sym { Sym::Neg(Box::new(a)) }

fn real_const(x: f64) -> Real {
    // rule constants are integer-valued (0, 1); represent exactly as a rational
    Real::from_real(x as i32, 1)
}

fn to_real(s: &Sym, vars: &[Real]) -> Real {
    match s {
        Sym::Var(i) => vars[*i].clone(),
        Sym::Const(x) => real_const(*x),
        Sym::Neg(a) => -(&to_real(a, vars)),
        Sym::Add(a, b) => to_real(a, vars) + to_real(b, vars),
        Sym::Sub(a, b) => to_real(a, vars) - to_real(b, vars),
        Sym::Mul(a, b) => to_real(a, vars) * to_real(b, vars),
    }
}

fn to_float(s: &Sym, vars: &[Float], rm: &RoundingMode) -> Float {
    match s {
        Sym::Var(i) => vars[*i].clone(),
        Sym::Const(x) => Float::from_f64(*x),
        Sym::Neg(a) => -(&to_float(a, vars, rm)),
        Sym::Add(a, b) => to_float(a, vars, rm).add_with_rounding_mode(to_float(b, vars, rm), rm),
        Sym::Sub(a, b) => to_float(a, vars, rm).sub_with_rounding_mode(to_float(b, vars, rm), rm),
        Sym::Mul(a, b) => to_float(a, vars, rm).mul_with_rounding_mode(to_float(b, vars, rm), rm),
    }
}

// ── classification ──────────────────────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    SoundEverywhere,
    RealsOnly,
    Unsound,
}
impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Class::SoundEverywhere => "sound (reals + IEEE-754)",
            Class::RealsOnly => "reals only  [FAST-MATH]",
            Class::Unsound => "UNSOUND",
        }
    }
}

pub struct Obligation {
    pub name: String,
    pub nvars: usize,
    pub lhs: Sym,
    pub rhs: Sym,
}

pub struct VerifyResult {
    pub name: String,
    pub reals_sound: bool,
    pub floats_sound: bool,
    pub class: Class,
    pub counterexample: Option<Vec<f64>>,
}

/// Sound over reals iff `lhs != rhs` is unsatisfiable.
fn reals_sound(ob: &Obligation) -> bool {
    let solver = Solver::new();
    let vars: Vec<Real> = (0..ob.nvars).map(|_| Real::fresh_const("r")).collect();
    let lhs = to_real(&ob.lhs, &vars);
    let rhs = to_real(&ob.rhs, &vars);
    solver.assert(&!lhs._eq(&rhs));
    matches!(solver.check(), SatResult::Unsat)
}

/// Sound over IEEE-754 iff, restricted to finite inputs, `lhs != rhs` is unsat.
/// Returns (sound, counterexample-if-any).
fn floats_sound(ob: &Obligation) -> (bool, Option<Vec<f64>>) {
    let solver = Solver::new();
    let rm = RoundingMode::round_nearest_ties_to_even();
    let vars: Vec<Float> = (0..ob.nvars).map(|_| Float::fresh_const_double("f")).collect();
    let zero = Float::from_f64(0.0);
    // finiteness: (a - a) == +0  ⇔  a is neither NaN nor ±∞
    for a in &vars {
        let diff = a.sub_with_rounding_mode(a.clone(), &rm);
        solver.assert(&diff._eq(&zero));
    }
    let lhs = to_float(&ob.lhs, &vars, &rm);
    let rhs = to_float(&ob.rhs, &vars, &rm);
    solver.assert(&!lhs._eq(&rhs));
    match solver.check() {
        SatResult::Unsat => (true, None),
        SatResult::Sat => {
            let m = solver.get_model().unwrap();
            let cex = vars
                .iter()
                .map(|a| m.eval(a, true).map(|f| f.as_f64()).unwrap_or(f64::NAN))
                .collect();
            (false, Some(cex))
        }
        SatResult::Unknown => (false, None),
    }
}

pub fn verify(ob: &Obligation) -> VerifyResult {
    let reals = reals_sound(ob);
    let (floats, cex) = floats_sound(ob);
    let class = if !reals {
        Class::Unsound
    } else if floats {
        Class::SoundEverywhere
    } else {
        Class::RealsOnly
    };
    VerifyResult {
        name: ob.name.clone(),
        reals_sound: reals,
        floats_sound: floats,
        class,
        counterexample: cex,
    }
}

// ── the rule set (obligations mirroring the egg rewrites) ────────────────────
fn ob(name: &str, nvars: usize, lhs: Sym, rhs: Sym) -> Obligation {
    Obligation { name: name.to_string(), nvars, lhs, rhs }
}

pub fn scalar_obligations() -> Vec<Obligation> {
    vec![
        // identities / commutativity
        ob("add-comm", 2, add(v(0), v(1)), add(v(1), v(0))),
        ob("mul-comm", 2, mul(v(0), v(1)), mul(v(1), v(0))),
        ob("add-zero", 1, add(v(0), c(0.0)), v(0)),
        ob("mul-one", 1, mul(v(0), c(1.0)), v(0)),
        ob("mul-zero", 1, mul(v(0), c(0.0)), c(0.0)),
        ob("neg-twice", 1, neg(neg(v(0))), v(0)),
        // reassociation & distribution (the fast-math family)
        ob("add-assoc", 3, add(add(v(0), v(1)), v(2)), add(v(0), add(v(1), v(2)))),
        ob("mul-assoc", 3, mul(mul(v(0), v(1)), v(2)), mul(v(0), mul(v(1), v(2)))),
        ob("distribute", 3, mul(v(0), add(v(1), v(2))), add(mul(v(0), v(1)), mul(v(0), v(2)))),
    ]
}

/// A deliberately incorrect rule — the verifier must catch it (Phase 4.9).
pub fn bad_obligation() -> Obligation {
    // claims subtraction is commutative: a - b == b - a  (false)
    ob("BAD:sub-comm", 2, sub(v(0), v(1)), sub(v(1), v(0)))
}

// ── bounded tensor-rule verification: matmul associativity ───────────────────
// Unroll (A·B)·C vs A·(B·C) into scalar Syms for concrete dims and verify every
// output element. Contraction dim k=1 has no summation → associative even over
// floats; k≥2 reassociates a sum → unsound over IEEE-754.
type Mat = Vec<Vec<Sym>>;

fn mat_vars(rows: usize, cols: usize, base: &mut usize) -> Mat {
    (0..rows)
        .map(|_| (0..cols).map(|_| { let i = *base; *base += 1; v(i) }).collect())
        .collect()
}

fn matmul_sym(a: &Mat, b: &Mat) -> Mat {
    let (m, k, n) = (a.len(), b.len(), b[0].len());
    (0..m)
        .map(|i| {
            (0..n)
                .map(|j| {
                    // left-associated sum: (((a0*b0)+a1*b1)+...)
                    let mut acc = mul(a[i][0].clone(), b[0][j].clone());
                    for p in 1..k {
                        acc = add(acc, mul(a[i][p].clone(), b[p][j].clone()));
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

pub fn verify_matmul_assoc(m: usize, k: usize, n: usize, p: usize) -> VerifyResult {
    let mut base = 0usize;
    let a = mat_vars(m, k, &mut base);
    let b = mat_vars(k, n, &mut base);
    let cc = mat_vars(n, p, &mut base);
    let nvars = base;

    let left = matmul_sym(&matmul_sym(&a, &b), &cc); // (A·B)·C
    let right = matmul_sym(&a, &matmul_sym(&b, &cc)); // A·(B·C)

    let name = format!("matmul-assoc[{m}x{k}x{n}x{p}]");
    let (mut reals, mut floats, mut cex) = (true, true, None);
    for i in 0..m {
        for j in 0..p {
            let o = ob(&name, nvars, left[i][j].clone(), right[i][j].clone());
            let r = verify(&o);
            reals &= r.reals_sound;
            floats &= r.floats_sound;
            if !r.floats_sound && cex.is_none() {
                cex = r.counterexample;
            }
        }
    }
    let class = if !reals {
        Class::Unsound
    } else if floats {
        Class::SoundEverywhere
    } else {
        Class::RealsOnly
    };
    VerifyResult { name, reals_sound: reals, floats_sound: floats, class, counterexample: cex }
}

/// Names of rules that are sound over reals but not IEEE-754 — the optimizer
/// should only enable these under an explicit fast-math opt-in.
pub fn fast_math_rule_names() -> Vec<String> {
    scalar_obligations()
        .iter()
        .map(verify)
        .filter(|r| r.class == Class::RealsOnly)
        .map(|r| r.name)
        .collect()
}

// ── soundness report ─────────────────────────────────────────────────────────
pub fn soundness_report() -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "minijax rewrite-rule soundness report  (Z3: Real vs IEEE-754)").unwrap();
    writeln!(s, "{}", "=".repeat(70)).unwrap();
    writeln!(s, "{:<22} {:>6} {:>7}  {}", "rule", "reals", "float", "classification").unwrap();
    writeln!(s, "{}", "-".repeat(70)).unwrap();

    let mut results: Vec<VerifyResult> = scalar_obligations().iter().map(verify).collect();
    for &k in &[1usize, 2, 3] {
        results.push(verify_matmul_assoc(1, k, 1, 1));
    }
    results.push(verify(&bad_obligation()));

    for r in &results {
        writeln!(
            s,
            "{:<22} {:>6} {:>7}  {}",
            r.name,
            if r.reals_sound { "ok" } else { "FAIL" },
            if r.floats_sound { "ok" } else { "FAIL" },
            r.class.label(),
        )
        .unwrap();
    }

    writeln!(s, "{}", "-".repeat(70)).unwrap();
    writeln!(s, "\nfloat counterexamples (finite inputs where LHS != RHS bit-for-bit):").unwrap();
    for r in &results {
        if let Some(cex) = &r.counterexample {
            let vals: Vec<String> = cex.iter().map(|x| format!("{x:.17}")).collect();
            writeln!(s, "  {:<22} vars = [{}]", r.name, vals.join(", ")).unwrap();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_sound_everywhere() {
        for name in ["add-comm", "mul-comm", "mul-one", "neg-twice"] {
            let o = scalar_obligations().into_iter().find(|o| o.name == name).unwrap();
            assert_eq!(verify(&o).class, Class::SoundEverywhere, "{name}");
        }
    }

    #[test]
    fn reassociation_is_reals_only() {
        for name in ["add-assoc", "mul-assoc", "distribute"] {
            let o = scalar_obligations().into_iter().find(|o| o.name == name).unwrap();
            let r = verify(&o);
            assert!(r.reals_sound, "{name} should be sound over reals");
            assert!(!r.floats_sound, "{name} should be unsound over floats");
            assert_eq!(r.class, Class::RealsOnly);
            assert!(r.counterexample.is_some(), "{name} needs a float counterexample");
        }
    }

    #[test]
    fn signed_zero_breaks_add_zero_mul_zero() {
        // x + 0 → x  and  x * 0 → 0  are unsound under IEEE-754 (signed zero).
        for name in ["add-zero", "mul-zero"] {
            let o = scalar_obligations().into_iter().find(|o| o.name == name).unwrap();
            let r = verify(&o);
            assert!(r.reals_sound);
            assert_eq!(r.class, Class::RealsOnly, "{name}");
        }
    }

    #[test]
    fn verifier_catches_a_bad_rule() {
        let r = verify(&bad_obligation());
        assert!(!r.reals_sound, "bad rule must fail over reals");
        assert_eq!(r.class, Class::Unsound);
    }

    #[test]
    fn matmul_assoc_k1_sound_k2_fastmath() {
        // k=1: no summation, associative even over floats.
        assert_eq!(verify_matmul_assoc(1, 1, 1, 1).class, Class::SoundEverywhere);
        // k=2,3: reassociates a sum → unsound over IEEE-754.
        let r2 = verify_matmul_assoc(2, 2, 2, 2);
        assert!(r2.reals_sound && !r2.floats_sound);
        assert_eq!(verify_matmul_assoc(1, 3, 1, 1).class, Class::RealsOnly);
    }
}
