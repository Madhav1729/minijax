#include "minijax/jit.hpp"

#include <Eigen/Dense>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/TargetSelect.h"

#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <cstdlib>

namespace minijax {


extern "C" void mj_matmul(const double* a, const double* b, double* out,
                           size_t m, size_t k, size_t n) {
    using RowMajMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMajMat> ea(a, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    Eigen::Map<const RowMajMat> eb(b, static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    Eigen::Map<RowMajMat> eout(out, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
    eout.noalias() = ea * eb;
}

namespace jit {

namespace {


double mj_exp(double x) { return std::exp(x); }
double mj_log(double x) { return std::log(x); }
double mj_tanh(double x) { return std::tanh(x); }
double mj_sqrt(double x) { return std::sqrt(x); }
double mj_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

using EntryFn = void (*)(double**, double**);

size_t numel_of_shape(const std::vector<size_t>& shape) {
    return shape.empty()
               ? 1
               : std::accumulate(shape.begin(), shape.end(), size_t{1},
                                  [](size_t a, size_t b) { return a * b; });
}


void emit_for(llvm::IRBuilder<>& b, llvm::Function* fn, llvm::Value* count,
               const std::function<void(llvm::Value* iv)>& body) {
    llvm::LLVMContext& lc = b.getContext();
    llvm::Type* i64 = llvm::Type::getInt64Ty(lc);

    llvm::AllocaInst* iv = b.CreateAlloca(i64, nullptr, "iv");
    b.CreateStore(llvm::ConstantInt::get(i64, 0), iv);
    llvm::BasicBlock* cond = llvm::BasicBlock::Create(lc, "for.cond", fn);
    llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(lc, "for.body", fn);
    llvm::BasicBlock* end = llvm::BasicBlock::Create(lc, "for.end", fn);
    b.CreateBr(cond);
    b.SetInsertPoint(cond);
    llvm::Value* cur = b.CreateLoad(i64, iv);
    b.CreateCondBr(b.CreateICmpULT(cur, count), bodybb, end);
    b.SetInsertPoint(bodybb);
    body(cur);
    b.CreateStore(b.CreateAdd(b.CreateLoad(i64, iv), llvm::ConstantInt::get(i64, 1)), iv);
    b.CreateBr(cond);
    b.SetInsertPoint(end);
}

struct Codegen {
    llvm::IRBuilder<>& b;
    llvm::Module& module;
    llvm::Function& entry;
    llvm::Value* inputs_arg;
    llvm::Value* bufs_arg;
    const Graph& g;

    llvm::Type* dbl;
    llvm::Type* ptr;
    llvm::Type* i64;

    std::map<std::string, llvm::FunctionCallee> wrappers;
    llvm::FunctionCallee matmul_callee;

    Codegen(llvm::IRBuilder<>& builder, llvm::Module& m, const Graph& graph)
        : b(builder), module(m), entry(*builder.GetInsertBlock()->getParent()),
          inputs_arg(entry.arg_begin()), bufs_arg(entry.arg_begin() + 1),
          g(graph),
          dbl(llvm::Type::getDoubleTy(builder.getContext())),
          ptr(llvm::PointerType::getUnqual(builder.getContext())),
          i64(llvm::Type::getInt64Ty(builder.getContext())) {
        entry.arg_begin()->setName("inputs");
        (entry.arg_begin() + 1)->setName("bufs");
        matmul_callee = module.getOrInsertFunction(
            "mj_matmul",
            llvm::FunctionType::get(llvm::Type::getVoidTy(b.getContext()),
                                    {ptr, ptr, ptr, i64, i64, i64}, false));
    }

    llvm::Value* c_u64(uint64_t v) { return llvm::ConstantInt::get(i64, v); }


    llvm::Value* node_buf(NodeId id) {
        llvm::Value* slot_addr =
            b.CreateGEP(ptr, bufs_arg, c_u64(id));
        return b.CreateLoad(ptr, slot_addr);
    }

    llvm::Value* elem(llvm::Value* base, llvm::Value* idx) {
        return b.CreateGEP(dbl, base, idx);
    }

    llvm::Value* load_dbl(llvm::Value* base, llvm::Value* idx) {
        return b.CreateLoad(dbl, elem(base, idx));
    }

    void store_dbl(llvm::Value* base, llvm::Value* idx, llvm::Value* v) {
        b.CreateStore(v, elem(base, idx));
    }

    void copy_in(llvm::Value* dst, size_t slot, uint64_t nbytes) {

        llvm::Value* slot_addr =
            b.CreateGEP(ptr, inputs_arg, c_u64(slot));
        llvm::Value* src = b.CreateLoad(ptr, slot_addr);
        b.CreateMemCpy(dst, llvm::MaybeAlign(8), src, llvm::MaybeAlign(8), nbytes);
    }

    llvm::FunctionCallee wrapper(const char* name) {
        auto it = wrappers.find(name);
        if (it != wrappers.end()) return it->second;
        llvm::FunctionCallee c =
            module.getOrInsertFunction(name, llvm::FunctionType::get(dbl, {dbl}, false));
        wrappers.emplace(name, c);
        return c;
    }

    llvm::FunctionCallee fabs_decl() {
        return module.getOrInsertFunction("mj_fabs",
                                           llvm::FunctionType::get(dbl, {dbl}, false));
    }


    void unary_call(NodeId out_id, NodeId in_id, const char* fn_name) {
        llvm::Value* out = node_buf(out_id);
        llvm::Value* in = node_buf(in_id);
        uint64_t n = numel_of_shape(g.shape_of(out_id));
        emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
            llvm::Value* x = load_dbl(in, k);
            store_dbl(out, k, b.CreateCall(wrapper(fn_name), {x}));
        });
    }

    void binary_arith(OpKind op, NodeId out_id, NodeId ia, NodeId ib) {
        llvm::Value* out = node_buf(out_id);
        llvm::Value* a = node_buf(ia);
        llvm::Value* bb_ = node_buf(ib);
        uint64_t n = numel_of_shape(g.shape_of(out_id));
        emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
            llvm::Value* x = load_dbl(a, k);
            llvm::Value* y = load_dbl(bb_, k);
            llvm::Value* r;
            switch (op) {
                case OpKind::Add: r = b.CreateFAdd(x, y); break;
                case OpKind::Sub: r = b.CreateFSub(x, y); break;
                case OpKind::Mul: r = b.CreateFMul(x, y); break;
                case OpKind::Div: r = b.CreateFDiv(x, y); break;
                default: throw std::logic_error("unreachable binary op");
            }
            store_dbl(out, k, r);
        });
    }

    void relu_or_step(NodeId out_id, NodeId in_id, bool step_result) {
        llvm::Value* out = node_buf(out_id);
        llvm::Value* in = node_buf(in_id);
        uint64_t n = numel_of_shape(g.shape_of(out_id));
        llvm::Value* zero = llvm::ConstantFP::get(dbl, 0.0);
        llvm::Value* one = llvm::ConstantFP::get(dbl, 1.0);
        emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
            llvm::Value* x = load_dbl(in, k);
            llvm::Value* pos = b.CreateFCmpOGT(x, zero);

            llvm::Value* if_true = step_result ? one : x;
            store_dbl(out, k, b.CreateSelect(pos, if_true, zero));
        });
    }

    void sum_full(NodeId out_id, NodeId in_id) {
        llvm::Value* out = node_buf(out_id);
        llvm::Value* in = node_buf(in_id);
        uint64_t n = numel_of_shape(g.shape_of(in_id));
        llvm::AllocaInst* acc = b.CreateAlloca(dbl, nullptr, "acc");
        b.CreateStore(llvm::ConstantFP::get(dbl, 0.0), acc);
        emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
            llvm::Value* cur = b.CreateLoad(dbl, acc);
            b.CreateStore(b.CreateFAdd(cur, load_dbl(in, k)), acc);
        });
        store_dbl(out, c_u64(0), b.CreateLoad(dbl, acc));
    }

    void sum_axis(NodeId out_id, NodeId in_id, size_t axis) {
        const auto& s = g.shape_of(in_id);
        uint64_t outer = 1, mid = s[axis], inner = 1;
        for (size_t d = 0; d < axis; ++d) outer *= s[d];
        for (size_t d = axis + 1; d < s.size(); ++d) inner *= s[d];
        llvm::Value* out = node_buf(out_id);
        llvm::Value* in = node_buf(in_id);
        emit_for(b, &entry, c_u64(outer), [&](llvm::Value* o) {
            emit_for(b, &entry, c_u64(inner), [&](llvm::Value* ii) {
                llvm::AllocaInst* acc = b.CreateAlloca(dbl, nullptr, "axacc");
                b.CreateStore(llvm::ConstantFP::get(dbl, 0.0), acc);
                emit_for(b, &entry, c_u64(mid), [&](llvm::Value* m) {

                    llvm::Value* om = b.CreateMul(o, c_u64(mid));
                    llvm::Value* base = b.CreateAdd(om, m);
                    base = b.CreateMul(base, c_u64(inner));
                    base = b.CreateAdd(base, ii);
                    llvm::Value* cur = b.CreateLoad(dbl, acc);
                    b.CreateStore(b.CreateFAdd(cur, load_dbl(in, base)), acc);
                });
                llvm::Value* oidx = b.CreateAdd(b.CreateMul(o, c_u64(inner)), ii);
                store_dbl(out, oidx, b.CreateLoad(dbl, acc));
            });
        });
    }

    void transpose(NodeId out_id, NodeId in_id) {
        const auto& s = g.shape_of(in_id);
        uint64_t R = s[0], C = s[1];
        llvm::Value* out = node_buf(out_id);
        llvm::Value* in = node_buf(in_id);
        emit_for(b, &entry, c_u64(R), [&](llvm::Value* r) {
            emit_for(b, &entry, c_u64(C), [&](llvm::Value* c) {
                llvm::Value* v = load_dbl(in, b.CreateAdd(b.CreateMul(r, c_u64(C)), c));
                store_dbl(out, b.CreateAdd(b.CreateMul(c, c_u64(R)), r), v);
            });
        });
    }


    void broadcast(NodeId out_id, NodeId in_id) {
        const auto& t = g.shape_of(out_id);
        const auto& s = g.shape_of(in_id);
        if (s.size() > t.size()) {
            throw std::invalid_argument("jit: broadcast source rank exceeds target rank");
        }
        std::vector<uint64_t> table;
        size_t rt = t.size(), rs = s.size();
        for (size_t k = 0; k < rt; ++k) {
            uint64_t s_dim = (k >= rt - rs) ? s[k - (rt - rs)] : 1;
            if (s_dim == t[k]) {
                uint64_t inner = 1, stride = 1;
                for (size_t d = k + 1; d < rt; ++d) inner *= t[d];
                for (size_t d = k - (rt - rs) + 1; d < rs; ++d) stride *= s[d];
                table.push_back({inner});
                table.push_back({t[k]});
                table.push_back({stride});
            } else if (s_dim != 1) {
                throw std::invalid_argument("jit: incompatible broadcast shapes");
            }
        }
        uint64_t n = numel_of_shape(t);
        llvm::Value* out = node_buf(out_id);
        llvm::Value* in = node_buf(in_id);

        if (table.empty()) {

            emit_for(b, &entry, c_u64(n), [&](llvm::Value* f) {
                store_dbl(out, f, load_dbl(in, c_u64(0)));
            });
            return;
        }
        auto* arr_ty = llvm::ArrayType::get(i64, table.size());
        auto* gv = new llvm::GlobalVariable(module, arr_ty, true,
                                             llvm::GlobalValue::PrivateLinkage,
                                             llvm::ConstantDataArray::get(b.getContext(), table),
                                             "bcast_tab");
        llvm::Value* k_count = c_u64(table.size() / 3);
        emit_for(b, &entry, c_u64(n), [&](llvm::Value* f) {
            llvm::AllocaInst* acc = b.CreateAlloca(i64, nullptr, "srcidx");
            b.CreateStore(c_u64(0), acc);
            emit_for(b, &entry, k_count, [&](llvm::Value* j) {
                llvm::Value* j3 = b.CreateMul(j, c_u64(3));
                auto cell = [&](uint64_t off) {
                    return b.CreateLoad(i64, b.CreateGEP(
                        arr_ty, gv, {c_u64(0), b.CreateAdd(j3, c_u64(off))}));
                };
                llvm::Value* inner_k = cell(0);
                llvm::Value* len_k = cell(1);
                llvm::Value* stride_k = cell(2);
                llvm::Value* cur = b.CreateLoad(i64, acc);
                llvm::Value* q = b.CreateUDiv(f, inner_k);
                q = b.CreateURem(q, len_k);
                b.CreateStore(b.CreateAdd(cur, b.CreateMul(q, stride_k)), acc);
            });
            store_dbl(out, f, load_dbl(in, b.CreateLoad(i64, acc)));
        });
    }

    void emit_node(NodeId i) {
        const Node& nd = g.node(i);
        uint64_t n = numel_of_shape(nd.shape);
        llvm::Value* out = node_buf(i);

        switch (nd.op) {
            case OpKind::Const: {
                emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
                    store_dbl(out, k, llvm::ConstantFP::get(dbl, nd.const_value));
                });
                break;
            }
            case OpKind::Input:
                copy_in(out, nd.input_slot, n * sizeof(double));
                break;
            case OpKind::Add:
            case OpKind::Sub:
            case OpKind::Mul:
            case OpKind::Div:
                binary_arith(nd.op, i, nd.inputs[0], nd.inputs[1]);
                break;
            case OpKind::Neg: {
                llvm::Value* in = node_buf(nd.inputs[0]);
                emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
                    store_dbl(out, k, b.CreateFNeg(load_dbl(in, k)));
                });
                break;
            }
            case OpKind::MatMul: {
                const auto& sa = g.shape_of(nd.inputs[0]);
                const auto& sb = g.shape_of(nd.inputs[1]);
                b.CreateCall(matmul_callee,
                             {node_buf(nd.inputs[0]), node_buf(nd.inputs[1]), out,
                              c_u64(sa[0]), c_u64(sa[1]), c_u64(sb[1])});
                break;
            }
            case OpKind::Relu:
                relu_or_step(i, nd.inputs[0], false);
                break;
            case OpKind::Step:
                relu_or_step(i, nd.inputs[0], true);
                break;
            case OpKind::Tanh:
                unary_call(i, nd.inputs[0], "mj_tanh");
                break;
            case OpKind::Sigmoid:
                unary_call(i, nd.inputs[0], "mj_sigmoid");
                break;
            case OpKind::Exp:
                unary_call(i, nd.inputs[0], "mj_exp");
                break;
            case OpKind::Log:
                unary_call(i, nd.inputs[0], "mj_log");
                break;
            case OpKind::Sqrt:
                unary_call(i, nd.inputs[0], "mj_sqrt");
                break;
            case OpKind::Abs: {
                llvm::Value* in = node_buf(nd.inputs[0]);
                emit_for(b, &entry, c_u64(n), [&](llvm::Value* k) {
                    store_dbl(out, k,
                              b.CreateCall(fabs_decl(), {load_dbl(in, k)}));
                });
                break;
            }
            case OpKind::Sum:
                sum_full(i, nd.inputs[0]);
                break;
            case OpKind::SumAxis:
                sum_axis(i, nd.inputs[0], nd.axis);
                break;
            case OpKind::Transpose:
                transpose(i, nd.inputs[0]);
                break;
            case OpKind::Broadcast:
                broadcast(i, nd.inputs[0]);
                break;
            case OpKind::Reshape:


                b.CreateMemCpy(out, llvm::MaybeAlign(8), node_buf(nd.inputs[0]),
                               llvm::MaybeAlign(8), n * sizeof(double));
                break;
            case OpKind::Softmax:
            case OpKind::CrossEntropy:
            case OpKind::Conv2d:
            case OpKind::BatchNorm:
            case OpKind::Im2Col:
            case OpKind::Col2Im:
            case OpKind::MaxPool:
            case OpKind::MaxPoolBackward:
                throw std::runtime_error(
                    std::string("jit: op not yet implemented (Phase 9 subset not covered by JIT): ") +
                    op_kind_name(nd.op));
        }
    }
};

[[noreturn]] void throw_llvm_error(llvm::Error e) {
    std::string msg;
    llvm::handleAllErrors(std::move(e),
                          [&msg](llvm::ErrorInfoBase& ei) { msg = ei.message(); });
    throw std::runtime_error("jit: " + msg);
}

std::unique_ptr<llvm::orc::LLJIT> make_lljit() {


    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    llvm::orc::LLJITBuilder builder;
    static const char* noopt = std::getenv("MINIJAX_JIT_NOOPT");
    if (!noopt) {


        builder.setCompileFunctionCreator(
            [](llvm::orc::JITTargetMachineBuilder jtmb)
                -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
                jtmb.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
                auto tm = jtmb.createTargetMachine();
                if (!tm) return tm.takeError();
                return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(*tm));
            });
    }
    auto jit = builder.create();
    if (!jit) throw_llvm_error(jit.takeError());
    return std::move(*jit);
}

}


struct JitProgram::Impl {
    std::unique_ptr<llvm::orc::LLJIT> the_jit;
    EntryFn entry = nullptr;
};

JitProgram::JitProgram(const Graph& g) {


    num_nodes_ = g.size();
    shapes_.reserve(num_nodes_);
    for (NodeId i = 0; i < num_nodes_; ++i) shapes_.push_back(g.shape_of(i));
    input_slots_ = g.inputs();

    bufs_.resize(num_nodes_);
    buf_ptrs_.resize(num_nodes_);
    for (NodeId i = 0; i < num_nodes_; ++i) {
        bufs_[i].assign(numel_of_shape(shapes_[i]), 0.0);
        buf_ptrs_[i] = bufs_[i].data();
    }
    input_ptrs_.assign(input_slots_.size(), nullptr);

    impl_ = std::make_unique<Impl>();
    impl_->the_jit = make_lljit();


    llvm::orc::MangleAndInterner mangle(impl_->the_jit->getExecutionSession(),
                                         impl_->the_jit->getDataLayout());
    llvm::orc::SymbolMap syms;
    auto def = [](void* p) {
        return llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr::fromPtr(p),
                                             llvm::JITSymbolFlags::Exported);
    };
    syms[mangle("mj_matmul")] = def(reinterpret_cast<void*>(&mj_matmul));
    syms[mangle("mj_exp")] = def(reinterpret_cast<void*>(&mj_exp));
    syms[mangle("mj_log")] = def(reinterpret_cast<void*>(&mj_log));
    syms[mangle("mj_tanh")] = def(reinterpret_cast<void*>(&mj_tanh));
    syms[mangle("mj_sqrt")] = def(reinterpret_cast<void*>(&mj_sqrt));
    syms[mangle("mj_sigmoid")] = def(reinterpret_cast<void*>(&mj_sigmoid));
    syms[mangle("mj_fabs")] = def(reinterpret_cast<void*>(static_cast<double (*)(double)>(&std::fabs)));
    if (llvm::Error e = impl_->the_jit->getMainJITDylib().define(
            llvm::orc::absoluteSymbols(std::move(syms)))) {
        throw_llvm_error(std::move(e));
    }


    auto tmctx = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("minijax_graph", *tmctx);
    {
        llvm::IRBuilder<> b(*tmctx);
        llvm::FunctionType* sig = llvm::FunctionType::get(
            llvm::Type::getVoidTy(*tmctx),
            {llvm::PointerType::getUnqual(*tmctx), llvm::PointerType::getUnqual(*tmctx)},
            false);
        llvm::Function* entry_fn = llvm::Function::Create(
            sig, llvm::Function::ExternalLinkage, "minijax_entry", module.get());
        llvm::BasicBlock* entry_bb =
            llvm::BasicBlock::Create(*tmctx, "entry", entry_fn);
        b.SetInsertPoint(entry_bb);

        Codegen cg(b, *module, g);
        for (NodeId i = 0; i < g.size(); ++i) cg.emit_node(i);
        b.CreateRetVoid();

        if (const char* d = std::getenv("MINIJAX_JIT_DUMP"); d && d[0] == '1') {
            module->print(llvm::errs(), nullptr);
        }

        llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(tmctx));
        if (llvm::Error e = impl_->the_jit->addIRModule(std::move(tsm))) {
            throw_llvm_error(std::move(e));
        }
    }

    auto addr = impl_->the_jit->lookup("minijax_entry");
    if (!addr) throw_llvm_error(addr.takeError());
    impl_->entry = addr->toPtr<EntryFn>();
}

JitProgram::~JitProgram() = default;
JitProgram::JitProgram(JitProgram&&) noexcept = default;
JitProgram& JitProgram::operator=(JitProgram&&) noexcept = default;

void JitProgram::execute(const std::vector<Tensor>& inputs) {
    if (inputs.size() != input_slots_.size()) {
        throw std::invalid_argument("run_jit: expected " +
                                     std::to_string(input_slots_.size()) +
                                     " inputs, got " + std::to_string(inputs.size()));
    }
    for (size_t s = 0; s < input_slots_.size(); ++s) {
        NodeId nid = input_slots_[s];
        if (inputs[s].shape() != shapes_[nid]) {
            throw std::invalid_argument("run_jit: input slot " + std::to_string(s) +
                                         " shape mismatch");
        }
        if (inputs[s].numel() != bufs_[nid].size()) {
            throw std::invalid_argument("run_jit: input slot " + std::to_string(s) +
                                         " element count mismatch");
        }
        input_ptrs_[s] = const_cast<double*>(inputs[s].data().data());
    }
    impl_->entry(input_ptrs_.data(), buf_ptrs_.data());
}

Tensor JitProgram::value(NodeId id) const {
    if (id >= num_nodes_) throw std::invalid_argument("run_jit: node id out of range");
    return Tensor::from_vec(shapes_[id], bufs_[id]);
}

Tensor run_jit(const Graph& g, const std::vector<Tensor>& inputs, NodeId output) {
    JitProgram p(g);
    p.execute(inputs);
    return p.value(output);
}

}
}
