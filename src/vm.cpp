#include "minijax/vm.hpp"
#include "minijax/memplan.hpp"
#include <stdexcept>
#include <unordered_map>
#include <cstring>

namespace minijax {

bool Instr::operator==(const Instr& o) const {
    return op == o.op && dst == o.dst && src1 == o.src1 && src2 == o.src2 &&
           const_value == o.const_value && input_slot == o.input_slot &&
           shape == o.shape && axis == o.axis;
}

bool Program::operator==(const Program& o) const {
    return instrs == o.instrs && num_regs == o.num_regs && output_reg == o.output_reg;
}

namespace {

VMOp op_kind_to_vmop(OpKind k) {
    switch (k) {
        case OpKind::Add: return VMOp::Add;
        case OpKind::Sub: return VMOp::Sub;
        case OpKind::Mul: return VMOp::Mul;
        case OpKind::Div: return VMOp::Div;
        case OpKind::Neg: return VMOp::Neg;
        case OpKind::MatMul: return VMOp::MatMul;
        case OpKind::Relu: return VMOp::Relu;
        case OpKind::Step: return VMOp::Step;
        case OpKind::Tanh: return VMOp::Tanh;
        case OpKind::Sigmoid: return VMOp::Sigmoid;
        case OpKind::Exp: return VMOp::Exp;
        case OpKind::Log: return VMOp::Log;
        case OpKind::Sqrt: return VMOp::Sqrt;
        case OpKind::Abs: return VMOp::Abs;
        case OpKind::Sum: return VMOp::Sum;
        case OpKind::SumAxis: return VMOp::SumAxis;
        case OpKind::Transpose: return VMOp::Transpose;
        case OpKind::Broadcast: return VMOp::Broadcast;
        case OpKind::Reshape: return VMOp::Reshape;
        default:
            throw std::runtime_error(std::string("vm::compile: op not yet implemented (Phase 9): ") +
                                      op_kind_name(k));
    }
}

}

Program compile(const Graph& g, NodeId output) {
    Program p;
    p.num_regs = static_cast<uint32_t>(g.size());
    p.output_reg = static_cast<uint32_t>(output);
    p.instrs.reserve(g.size());

    for (NodeId i = 0; i < g.size(); ++i) {
        const Node& n = g.node(i);
        Instr instr;
        instr.dst = static_cast<uint32_t>(i);

        if (n.op == OpKind::Input) {
            instr.op = VMOp::LoadInput;
            instr.input_slot = static_cast<uint32_t>(n.input_slot);
        } else if (n.op == OpKind::Const) {
            instr.op = VMOp::LoadConst;
            instr.const_value = n.const_value;
            instr.shape = n.shape;
        } else {
            instr.op = op_kind_to_vmop(n.op);
            if (!n.inputs.empty()) instr.src1 = static_cast<uint32_t>(n.inputs[0]);
            if (n.inputs.size() > 1) instr.src2 = static_cast<uint32_t>(n.inputs[1]);
            if (n.op == OpKind::SumAxis) instr.axis = n.axis;
            if (n.op == OpKind::Broadcast || n.op == OpKind::Reshape) instr.shape = n.target_shape;
        }
        p.instrs.push_back(std::move(instr));
    }
    return p;
}

Program compact(const Program& p) {
    size_t n = p.instrs.size();


    std::vector<StepAccess> steps;
    steps.reserve(n);
    for (const Instr& instr : p.instrs) {
        StepAccess sa;
        sa.dst = instr.dst;
        bool has_src1 = instr.op != VMOp::LoadInput && instr.op != VMOp::LoadConst;
        bool has_src2 = instr.op == VMOp::Add || instr.op == VMOp::Sub || instr.op == VMOp::Mul ||
                         instr.op == VMOp::Div || instr.op == VMOp::MatMul;
        if (has_src1) sa.reads.push_back(instr.src1);
        if (has_src2) sa.reads.push_back(instr.src2);
        steps.push_back(std::move(sa));
    }
    std::vector<size_t> last_use = compute_last_use(p.num_regs, steps);
    last_use[p.output_reg] = SIZE_MAX;


    std::vector<uint32_t> virt_to_phys(p.num_regs, UINT32_MAX);
    std::vector<uint32_t> free_list;


    std::unordered_map<uint32_t, uint32_t> phys_owner;
    uint32_t next_phys = 0;

    Program out;
    out.output_reg = p.output_reg;
    out.instrs.reserve(n);

    for (size_t i = 0; i < n; ++i) {


        for (auto it = phys_owner.begin(); it != phys_owner.end();) {
            if (last_use[it->second] < i) {
                free_list.push_back(it->first);
                it = phys_owner.erase(it);
            } else {
                ++it;
            }
        }

        const Instr& instr = p.instrs[i];
        Instr new_instr = instr;

        bool has_src1 = instr.op != VMOp::LoadInput && instr.op != VMOp::LoadConst;
        bool has_src2 = instr.op == VMOp::Add || instr.op == VMOp::Sub || instr.op == VMOp::Mul ||
                         instr.op == VMOp::Div || instr.op == VMOp::MatMul;
        if (has_src1) new_instr.src1 = virt_to_phys[instr.src1];
        if (has_src2) new_instr.src2 = virt_to_phys[instr.src2];

        uint32_t phys;
        if (!free_list.empty()) {
            phys = free_list.back();
            free_list.pop_back();
        } else {
            phys = next_phys++;
        }
        virt_to_phys[instr.dst] = phys;
        phys_owner[phys] = instr.dst;
        new_instr.dst = phys;

        out.instrs.push_back(new_instr);
    }

    out.num_regs = next_phys;
    out.output_reg = virt_to_phys[p.output_reg];
    return out;
}

Tensor run_vm(const Program& p, const std::vector<Tensor>& inputs) {
    std::vector<std::optional<Tensor>> regs(p.num_regs);

    auto get = [&](uint32_t r) -> const Tensor& {
        if (!regs[r].has_value()) throw std::runtime_error("run_vm: read from uninitialized register");
        return *regs[r];
    };

    for (const auto& instr : p.instrs) {
        switch (instr.op) {
            case VMOp::LoadInput:
                if (instr.input_slot >= inputs.size())
                    throw std::invalid_argument("run_vm: input slot out of range");
                regs[instr.dst] = inputs[instr.input_slot];
                break;
            case VMOp::LoadConst:
                regs[instr.dst] = Tensor::full(instr.shape, instr.const_value);
                break;
            case VMOp::Add: regs[instr.dst] = get(instr.src1) + get(instr.src2); break;
            case VMOp::Sub: regs[instr.dst] = get(instr.src1) - get(instr.src2); break;
            case VMOp::Mul: regs[instr.dst] = get(instr.src1) * get(instr.src2); break;
            case VMOp::Div: regs[instr.dst] = get(instr.src1) / get(instr.src2); break;
            case VMOp::Neg: regs[instr.dst] = -get(instr.src1); break;
            case VMOp::MatMul: regs[instr.dst] = Tensor::matmul(get(instr.src1), get(instr.src2)); break;
            case VMOp::Relu: regs[instr.dst] = get(instr.src1).relu(); break;
            case VMOp::Step: regs[instr.dst] = get(instr.src1).step(); break;
            case VMOp::Tanh: regs[instr.dst] = get(instr.src1).tanh(); break;
            case VMOp::Sigmoid: regs[instr.dst] = get(instr.src1).sigmoid(); break;
            case VMOp::Exp: regs[instr.dst] = get(instr.src1).exp(); break;
            case VMOp::Log: regs[instr.dst] = get(instr.src1).log(); break;
            case VMOp::Sqrt: regs[instr.dst] = get(instr.src1).sqrt(); break;
            case VMOp::Abs: regs[instr.dst] = get(instr.src1).abs(); break;
            case VMOp::Sum: regs[instr.dst] = get(instr.src1).sum(); break;
            case VMOp::SumAxis: regs[instr.dst] = get(instr.src1).sum_axis(instr.axis); break;
            case VMOp::Transpose: regs[instr.dst] = get(instr.src1).transpose(); break;
            case VMOp::Broadcast: regs[instr.dst] = get(instr.src1).broadcast_to(instr.shape); break;
            case VMOp::Reshape: regs[instr.dst] = get(instr.src1).reshape(instr.shape); break;
        }
    }
    return get(p.output_reg);
}


namespace {

void put_u8(std::vector<uint8_t>& buf, uint8_t v) { buf.push_back(v); }
void put_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_f64(std::vector<uint8_t>& buf, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    put_u64(buf, bits);
}

uint8_t read_u8(const std::vector<uint8_t>& buf, size_t& pos) {
    if (pos >= buf.size()) throw std::runtime_error("deserialize: unexpected end of buffer");
    return buf[pos++];
}
uint32_t read_u32(const std::vector<uint8_t>& buf, size_t& pos) {
    if (pos + 4 > buf.size()) throw std::runtime_error("deserialize: unexpected end of buffer");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(buf[pos + i]) << (8 * i);
    pos += 4;
    return v;
}
uint64_t read_u64(const std::vector<uint8_t>& buf, size_t& pos) {
    if (pos + 8 > buf.size()) throw std::runtime_error("deserialize: unexpected end of buffer");
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(buf[pos + i]) << (8 * i);
    pos += 8;
    return v;
}
double read_f64(const std::vector<uint8_t>& buf, size_t& pos) {
    uint64_t bits = read_u64(buf, pos);
    double v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

}

std::vector<uint8_t> serialize(const Program& p) {
    std::vector<uint8_t> buf;
    put_u32(buf, p.num_regs);
    put_u32(buf, p.output_reg);
    put_u32(buf, static_cast<uint32_t>(p.instrs.size()));
    for (const auto& instr : p.instrs) {
        put_u8(buf, static_cast<uint8_t>(instr.op));
        put_u32(buf, instr.dst);
        put_u32(buf, instr.src1);
        put_u32(buf, instr.src2);
        put_f64(buf, instr.const_value);
        put_u32(buf, instr.input_slot);
        put_u64(buf, instr.axis);
        put_u32(buf, static_cast<uint32_t>(instr.shape.size()));
        for (size_t d : instr.shape) put_u64(buf, static_cast<uint64_t>(d));
    }
    return buf;
}

Program deserialize(const std::vector<uint8_t>& bytes) {
    Program p;
    size_t pos = 0;
    p.num_regs = read_u32(bytes, pos);
    p.output_reg = read_u32(bytes, pos);
    uint32_t count = read_u32(bytes, pos);
    p.instrs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Instr instr;
        instr.op = static_cast<VMOp>(read_u8(bytes, pos));
        instr.dst = read_u32(bytes, pos);
        instr.src1 = read_u32(bytes, pos);
        instr.src2 = read_u32(bytes, pos);
        instr.const_value = read_f64(bytes, pos);
        instr.input_slot = read_u32(bytes, pos);
        instr.axis = static_cast<size_t>(read_u64(bytes, pos));
        uint32_t shape_len = read_u32(bytes, pos);
        instr.shape.reserve(shape_len);
        for (uint32_t j = 0; j < shape_len; ++j) instr.shape.push_back(static_cast<size_t>(read_u64(bytes, pos)));
        p.instrs.push_back(std::move(instr));
    }
    return p;
}

}
