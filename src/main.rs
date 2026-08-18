mod ir;
mod interp;
mod autodiff;
mod opt;
mod verify;
mod vm;
mod jit;
mod nn;
mod fuzz;
mod viz;
mod cli;

fn main() {
    println!("minijax — differentiable array compiler");
    println!("Run `cargo test` to verify the IR and interpreter.");
}
