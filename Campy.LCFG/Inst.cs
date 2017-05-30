using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using Campy.CIL;
using Swigged.LLVM;
using System.Linq;
using Campy.Graphs;

namespace Campy.LCFG
{
    public class Inst
    {
        public BuilderRef Builder { get { return Block.Builder; } }

        public ContextRef LLVMContext { get; set; }

        public CIL_Inst Instruction { get; private set; }

        public List<Value> LLVMInstructions { get; private set; }

        public LLVMCFG.Vertex Block { get; set; }

        public override string ToString() { return Instruction.ToString(); }

        public Inst(CIL_Inst i) { Instruction = i; }

        public Mono.Cecil.Cil.OpCode OpCode { get { return Instruction.OpCode; } }

        public object Operand { get { return Instruction.Operand; } }

        // Required instruction sequencing so we can translate groups of instructions.
        public virtual Inst Next { get; set; }

        public virtual void ComputeStackLevel(ref int level_after)
        {
        }

        public virtual Inst Convert(State state) { return null; }

        static public Inst Wrap(CIL_Inst i)
        {
            Mono.Cecil.Cil.OpCode op = i.OpCode;
            switch (op.Code)
            {
                case Mono.Cecil.Cil.Code.Add:
                    return new i_add(i);
                case Mono.Cecil.Cil.Code.Add_Ovf:
                    return new i_add_ovf(i);
                case Mono.Cecil.Cil.Code.Add_Ovf_Un:
                    return new i_add_ovf_un(i);
                case Mono.Cecil.Cil.Code.And:
                    return new i_and(i);
                case Mono.Cecil.Cil.Code.Arglist:
                    return new i_arglist(i);
                case Mono.Cecil.Cil.Code.Beq:
                    return new i_beq(i);
                case Mono.Cecil.Cil.Code.Beq_S:
                    return new i_beq(i);
                case Mono.Cecil.Cil.Code.Bge:
                    return new i_bge(i);
                case Mono.Cecil.Cil.Code.Bge_S:
                    return new i_bge_s(i);
                case Mono.Cecil.Cil.Code.Bge_Un:
                    return new i_bge_un(i);
                case Mono.Cecil.Cil.Code.Bge_Un_S:
                    return new i_bge_un_s(i);
                case Mono.Cecil.Cil.Code.Bgt:
                    return new i_bgt(i);
                case Mono.Cecil.Cil.Code.Bgt_S:
                    return new i_bgt_s(i);
                case Mono.Cecil.Cil.Code.Bgt_Un:
                    return new i_bgt_un(i);
                case Mono.Cecil.Cil.Code.Bgt_Un_S:
                    return new i_bgt_un_s(i);
                case Mono.Cecil.Cil.Code.Ble:
                    return new i_ble(i);
                case Mono.Cecil.Cil.Code.Ble_S:
                    return new i_ble_s(i);
                case Mono.Cecil.Cil.Code.Ble_Un:
                    return new i_ble_un(i);
                case Mono.Cecil.Cil.Code.Ble_Un_S:
                    return new i_ble_un_s(i);
                case Mono.Cecil.Cil.Code.Blt:
                    return new i_blt(i);
                case Mono.Cecil.Cil.Code.Blt_S:
                    return new i_blt_s(i);
                case Mono.Cecil.Cil.Code.Blt_Un:
                    return new i_blt_un(i);
                case Mono.Cecil.Cil.Code.Blt_Un_S:
                    return new i_blt_un_s(i);
                case Mono.Cecil.Cil.Code.Bne_Un:
                    return new i_bne_un(i);
                case Mono.Cecil.Cil.Code.Bne_Un_S:
                    return new i_bne_un_s(i);
                case Mono.Cecil.Cil.Code.Box:
                    return new i_box(i);
                case Mono.Cecil.Cil.Code.Br:
                    return new i_br(i);
                case Mono.Cecil.Cil.Code.Br_S:
                    return new i_br_s(i);
                case Mono.Cecil.Cil.Code.Break:
                    return new i_break(i);
                case Mono.Cecil.Cil.Code.Brfalse:
                    return new i_brfalse(i);
                case Mono.Cecil.Cil.Code.Brfalse_S:
                    return new i_brfalse_s(i);
                // Missing brnull
                // Missing brzero
                case Mono.Cecil.Cil.Code.Brtrue:
                    return new i_brtrue(i);
                case Mono.Cecil.Cil.Code.Brtrue_S:
                    return new i_brtrue_s(i);
                case Mono.Cecil.Cil.Code.Call:
                    return new i_call(i);
                case Mono.Cecil.Cil.Code.Calli:
                    return new i_calli(i);
                case Mono.Cecil.Cil.Code.Callvirt:
                    return new i_callvirt(i);
                case Mono.Cecil.Cil.Code.Castclass:
                    return new i_castclass(i);
                case Mono.Cecil.Cil.Code.Ceq:
                    return new i_ceq(i);
                case Mono.Cecil.Cil.Code.Cgt:
                    return new i_cgt(i);
                case Mono.Cecil.Cil.Code.Cgt_Un:
                    return new i_cgt_un(i);
                case Mono.Cecil.Cil.Code.Ckfinite:
                    return new i_ckfinite(i);
                case Mono.Cecil.Cil.Code.Clt:
                    return new i_clt(i);
                case Mono.Cecil.Cil.Code.Clt_Un:
                    return new i_clt_un(i);
                case Mono.Cecil.Cil.Code.Constrained:
                    return new i_constrained(i);
                case Mono.Cecil.Cil.Code.Conv_I1:
                    return new i_conv_i1(i);
                case Mono.Cecil.Cil.Code.Conv_I2:
                    return new i_conv_i2(i);
                case Mono.Cecil.Cil.Code.Conv_I4:
                    return new i_conv_i4(i);
                case Mono.Cecil.Cil.Code.Conv_I8:
                    return new i_conv_i8(i);
                case Mono.Cecil.Cil.Code.Conv_I:
                    return new i_conv_i(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I1:
                    return new i_conv_ovf_i1(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I1_Un:
                    return new i_conv_ovf_i1_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I2:
                    return new i_conv_ovf_i2(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I2_Un:
                    return new i_conv_ovf_i2_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I4:
                    return new i_conv_ovf_i4(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I4_Un:
                    return new i_conv_ovf_i4_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I8:
                    return new i_conv_ovf_i8(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I8_Un:
                    return new i_conv_ovf_i8_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I:
                    return new i_conv_ovf_i(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I_Un:
                    return new i_conv_ovf_i_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U1:
                    return new i_conv_ovf_u1(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U1_Un:
                    return new i_conv_ovf_u1_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U2:
                    return new i_conv_ovf_u2(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U2_Un:
                    return new i_conv_ovf_u2_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U4:
                    return new i_conv_ovf_u4(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U4_Un:
                    return new i_conv_ovf_u4_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U8:
                    return new i_conv_ovf_u8(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U8_Un:
                    return new i_conv_ovf_u8_un(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U:
                    return new i_conv_ovf_u(i);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U_Un:
                    return new i_conv_ovf_u_un(i);
                case Mono.Cecil.Cil.Code.Conv_R4:
                    return new i_conv_r4(i);
                case Mono.Cecil.Cil.Code.Conv_R8:
                    return new i_conv_r8(i);
                case Mono.Cecil.Cil.Code.Conv_R_Un:
                    return new i_conv_r_un(i);
                case Mono.Cecil.Cil.Code.Conv_U1:
                    return new i_conv_u1(i);
                case Mono.Cecil.Cil.Code.Conv_U2:
                    return new i_conv_u2(i);
                case Mono.Cecil.Cil.Code.Conv_U4:
                    return new i_conv_u4(i);
                case Mono.Cecil.Cil.Code.Conv_U8:
                    return new i_conv_u8(i);
                case Mono.Cecil.Cil.Code.Conv_U:
                    return new i_conv_u(i);
                case Mono.Cecil.Cil.Code.Cpblk:
                    return new i_cpblk(i);
                case Mono.Cecil.Cil.Code.Cpobj:
                    return new i_cpobj(i);
                case Mono.Cecil.Cil.Code.Div:
                    return new i_div(i);
                case Mono.Cecil.Cil.Code.Div_Un:
                    return new i_div_un(i);
                case Mono.Cecil.Cil.Code.Dup:
                    return new i_dup(i);
                case Mono.Cecil.Cil.Code.Endfilter:
                    return new i_endfilter(i);
                case Mono.Cecil.Cil.Code.Endfinally:
                    return new i_endfinally(i);
                case Mono.Cecil.Cil.Code.Initblk:
                    return new i_initblk(i);
                case Mono.Cecil.Cil.Code.Initobj:
                    return new i_initobj(i);
                case Mono.Cecil.Cil.Code.Isinst:
                    return new i_isinst(i);
                case Mono.Cecil.Cil.Code.Jmp:
                    return new i_jmp(i);
                case Mono.Cecil.Cil.Code.Ldarg:
                    return new i_ldarg(i);
                case Mono.Cecil.Cil.Code.Ldarg_0:
                    return new i_ldarg_0(i);
                case Mono.Cecil.Cil.Code.Ldarg_1:
                    return new i_ldarg_1(i);
                case Mono.Cecil.Cil.Code.Ldarg_2:
                    return new i_ldarg_2(i);
                case Mono.Cecil.Cil.Code.Ldarg_3:
                    return new i_ldarg_3(i);
                case Mono.Cecil.Cil.Code.Ldarg_S:
                    return new i_ldarg_s(i);
                case Mono.Cecil.Cil.Code.Ldarga:
                    return new i_ldarga(i);
                case Mono.Cecil.Cil.Code.Ldarga_S:
                    return new i_ldarga_s(i);
                case Mono.Cecil.Cil.Code.Ldc_I4:
                    return new i_ldc_i4(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_0:
                    return new i_ldc_i4_0(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_1:
                    return new i_ldc_i4_1(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_2:
                    return new i_ldc_i4_2(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_3:
                    return new i_ldc_i4_3(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_4:
                    return new i_ldc_i4_4(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_5:
                    return new i_ldc_i4_5(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_6:
                    return new i_ldc_i4_6(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_7:
                    return new i_ldc_i4_7(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_8:
                    return new i_ldc_i4_8(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_M1:
                    return new i_ldc_i4_m1(i);
                case Mono.Cecil.Cil.Code.Ldc_I4_S:
                    return new i_ldc_i4_s(i);
                case Mono.Cecil.Cil.Code.Ldc_I8:
                    return new i_ldc_i8(i);
                case Mono.Cecil.Cil.Code.Ldc_R4:
                    return new i_ldc_r4(i);
                case Mono.Cecil.Cil.Code.Ldc_R8:
                    return new i_ldc_r8(i);
                case Mono.Cecil.Cil.Code.Ldelem_Any:
                    return new i_ldelem_any(i);
                case Mono.Cecil.Cil.Code.Ldelem_I1:
                    return new i_ldelem_i1(i);
                case Mono.Cecil.Cil.Code.Ldelem_I2:
                    return new i_ldelem_i2(i);
                case Mono.Cecil.Cil.Code.Ldelem_I4:
                    return new i_ldelem_i4(i);
                case Mono.Cecil.Cil.Code.Ldelem_I8:
                    return new i_ldelem_i8(i);
                case Mono.Cecil.Cil.Code.Ldelem_I:
                    return new i_ldelem_i(i);
                case Mono.Cecil.Cil.Code.Ldelem_R4:
                    return new i_ldelem_r4(i);
                case Mono.Cecil.Cil.Code.Ldelem_R8:
                    return new i_ldelem_r8(i);
                case Mono.Cecil.Cil.Code.Ldelem_Ref:
                    return new i_ldelem_ref(i);
                case Mono.Cecil.Cil.Code.Ldelem_U1:
                    return new i_ldelem_u1(i);
                case Mono.Cecil.Cil.Code.Ldelem_U2:
                    return new i_ldelem_u2(i);
                case Mono.Cecil.Cil.Code.Ldelem_U4:
                    return new i_ldelem_u4(i);
                case Mono.Cecil.Cil.Code.Ldelema:
                    return new i_ldelema(i);
                case Mono.Cecil.Cil.Code.Ldfld:
                    return new i_ldfld(i);
                case Mono.Cecil.Cil.Code.Ldflda:
                    return new i_ldflda(i);
                case Mono.Cecil.Cil.Code.Ldftn:
                    return new i_ldftn(i);
                case Mono.Cecil.Cil.Code.Ldind_I1:
                    return new i_ldind_i1(i);
                case Mono.Cecil.Cil.Code.Ldind_I2:
                    return new i_ldind_i2(i);
                case Mono.Cecil.Cil.Code.Ldind_I4:
                    return new i_ldind_i4(i);
                case Mono.Cecil.Cil.Code.Ldind_I8:
                    return new i_ldind_i8(i);
                case Mono.Cecil.Cil.Code.Ldind_I:
                    return new i_ldind_i(i);
                case Mono.Cecil.Cil.Code.Ldind_R4:
                    return new i_ldind_r4(i);
                case Mono.Cecil.Cil.Code.Ldind_R8:
                    return new i_ldind_r8(i);
                case Mono.Cecil.Cil.Code.Ldind_Ref:
                    return new i_ldind_ref(i);
                case Mono.Cecil.Cil.Code.Ldind_U1:
                    return new i_ldind_u1(i);
                case Mono.Cecil.Cil.Code.Ldind_U2:
                    return new i_ldind_u2(i);
                case Mono.Cecil.Cil.Code.Ldind_U4:
                    return new i_ldind_u4(i);
                case Mono.Cecil.Cil.Code.Ldlen:
                    return new i_ldlen(i);
                case Mono.Cecil.Cil.Code.Ldloc:
                    return new i_ldloc(i);
                case Mono.Cecil.Cil.Code.Ldloc_0:
                    return new i_ldloc_0(i);
                case Mono.Cecil.Cil.Code.Ldloc_1:
                    return new i_ldloc_1(i);
                case Mono.Cecil.Cil.Code.Ldloc_2:
                    return new i_ldloc_2(i);
                case Mono.Cecil.Cil.Code.Ldloc_3:
                    return new i_ldloc_3(i);
                case Mono.Cecil.Cil.Code.Ldloc_S:
                    return new i_ldloc_s(i);
                case Mono.Cecil.Cil.Code.Ldloca:
                    return new i_ldloca(i);
                case Mono.Cecil.Cil.Code.Ldloca_S:
                    return new i_ldloca_s(i);
                case Mono.Cecil.Cil.Code.Ldnull:
                    return new i_ldnull(i);
                case Mono.Cecil.Cil.Code.Ldobj:
                    return new i_ldobj(i);
                case Mono.Cecil.Cil.Code.Ldsfld:
                    return new i_ldsfld(i);
                case Mono.Cecil.Cil.Code.Ldsflda:
                    return new i_ldsflda(i);
                case Mono.Cecil.Cil.Code.Ldstr:
                    return new i_ldstr(i);
                case Mono.Cecil.Cil.Code.Ldtoken:
                    return new i_ldtoken(i);
                case Mono.Cecil.Cil.Code.Ldvirtftn:
                    return new i_ldvirtftn(i);
                case Mono.Cecil.Cil.Code.Leave:
                    return new i_leave(i);
                case Mono.Cecil.Cil.Code.Leave_S:
                    return new i_leave_s(i);
                case Mono.Cecil.Cil.Code.Localloc:
                    return new i_localloc(i);
                case Mono.Cecil.Cil.Code.Mkrefany:
                    return new i_mkrefany(i);
                case Mono.Cecil.Cil.Code.Mul:
                    return new i_mul(i);
                case Mono.Cecil.Cil.Code.Mul_Ovf:
                    return new i_mul_ovf(i);
                case Mono.Cecil.Cil.Code.Mul_Ovf_Un:
                    return new i_mul_ovf_un(i);
                case Mono.Cecil.Cil.Code.Neg:
                    return new i_neg(i);
                case Mono.Cecil.Cil.Code.Newarr:
                    return new i_newarr(i);
                case Mono.Cecil.Cil.Code.Newobj:
                    return new i_newobj(i);
                case Mono.Cecil.Cil.Code.No:
                    return new i_no(i);
                case Mono.Cecil.Cil.Code.Nop:
                    return new i_nop(i);
                case Mono.Cecil.Cil.Code.Not:
                    return new i_not(i);
                case Mono.Cecil.Cil.Code.Or:
                    return new i_or(i);
                case Mono.Cecil.Cil.Code.Pop:
                    return new i_pop(i);
                case Mono.Cecil.Cil.Code.Readonly:
                    return new i_readonly(i);
                case Mono.Cecil.Cil.Code.Refanytype:
                    return new i_refanytype(i);
                case Mono.Cecil.Cil.Code.Refanyval:
                    return new i_refanyval(i);
                case Mono.Cecil.Cil.Code.Rem:
                    return new i_rem(i);
                case Mono.Cecil.Cil.Code.Rem_Un:
                    return new i_rem_un(i);
                case Mono.Cecil.Cil.Code.Ret:
                    return new i_ret(i);
                case Mono.Cecil.Cil.Code.Rethrow:
                    return new i_rethrow(i);
                case Mono.Cecil.Cil.Code.Shl:
                    return new i_shl(i);
                case Mono.Cecil.Cil.Code.Shr:
                    return new i_shr(i);
                case Mono.Cecil.Cil.Code.Shr_Un:
                    return new i_shr_un(i);
                case Mono.Cecil.Cil.Code.Sizeof:
                    return new i_sizeof(i);
                case Mono.Cecil.Cil.Code.Starg:
                    return new i_starg(i);
                case Mono.Cecil.Cil.Code.Starg_S:
                    return new i_starg_s(i);
                case Mono.Cecil.Cil.Code.Stelem_Any:
                    return new i_stelem_any(i);
                case Mono.Cecil.Cil.Code.Stelem_I1:
                    return new i_stelem_i1(i);
                case Mono.Cecil.Cil.Code.Stelem_I2:
                    return new i_stelem_i2(i);
                case Mono.Cecil.Cil.Code.Stelem_I4:
                    return new i_stelem_i4(i);
                case Mono.Cecil.Cil.Code.Stelem_I8:
                    return new i_stelem_i8(i);
                case Mono.Cecil.Cil.Code.Stelem_I:
                    return new i_stelem_i(i);
                case Mono.Cecil.Cil.Code.Stelem_R4:
                    return new i_stelem_r4(i);
                case Mono.Cecil.Cil.Code.Stelem_R8:
                    return new i_stelem_r8(i);
                case Mono.Cecil.Cil.Code.Stelem_Ref:
                    return new i_stelem_ref(i);
                case Mono.Cecil.Cil.Code.Stfld:
                    return new i_stfld(i);
                case Mono.Cecil.Cil.Code.Stind_I1:
                    return new i_stind_i1(i);
                case Mono.Cecil.Cil.Code.Stind_I2:
                    return new i_stind_i2(i);
                case Mono.Cecil.Cil.Code.Stind_I4:
                    return new i_stind_i4(i);
                case Mono.Cecil.Cil.Code.Stind_I8:
                    return new i_stind_i8(i);
                case Mono.Cecil.Cil.Code.Stind_I:
                    return new i_stind_i(i);
                case Mono.Cecil.Cil.Code.Stind_R4:
                    return new i_stind_r4(i);
                case Mono.Cecil.Cil.Code.Stind_R8:
                    return new i_stind_r8(i);
                case Mono.Cecil.Cil.Code.Stind_Ref:
                    return new i_stind_ref(i);
                case Mono.Cecil.Cil.Code.Stloc:
                    return new i_stloc(i);
                case Mono.Cecil.Cil.Code.Stloc_0:
                    return new i_stloc_0(i);
                case Mono.Cecil.Cil.Code.Stloc_1:
                    return new i_stloc_1(i);
                case Mono.Cecil.Cil.Code.Stloc_2:
                    return new i_stloc_2(i);
                case Mono.Cecil.Cil.Code.Stloc_3:
                    return new i_stloc_3(i);
                case Mono.Cecil.Cil.Code.Stloc_S:
                    return new i_stloc_s(i);
                case Mono.Cecil.Cil.Code.Stobj:
                    return new i_stobj(i);
                case Mono.Cecil.Cil.Code.Stsfld:
                    return new i_stsfld(i);
                case Mono.Cecil.Cil.Code.Sub:
                    return new i_sub(i);
                case Mono.Cecil.Cil.Code.Sub_Ovf:
                    return new i_sub_ovf(i);
                case Mono.Cecil.Cil.Code.Sub_Ovf_Un:
                    return new i_sub_ovf_un(i);
                case Mono.Cecil.Cil.Code.Switch:
                    return new i_switch(i);
                case Mono.Cecil.Cil.Code.Tail:
                    return new i_tail(i);
                case Mono.Cecil.Cil.Code.Throw:
                    return new i_throw(i);
                case Mono.Cecil.Cil.Code.Unaligned:
                    return new i_unaligned(i);
                case Mono.Cecil.Cil.Code.Unbox:
                    return new i_unbox(i);
                case Mono.Cecil.Cil.Code.Unbox_Any:
                    return new i_unbox_any(i);
                case Mono.Cecil.Cil.Code.Volatile:
                    return new i_volatile(i);
                case Mono.Cecil.Cil.Code.Xor:
                    return new i_xor(i);
                default:
                    throw new Exception("Unknown instruction type " + i);
            }
        }

        private State _state_in;
        public State StateIn
        {
            get { return _state_in; }
            set { _state_in = value; }
        }

        private State _state_out;
        public State StateOut
        {
            get { return _state_out; }
            set { _state_out = value; }
        }

        public UInt32 TargetPointerSizeInBits = 64;
    }

    public class BinaryOpInst : Inst
    {
        public BinaryOpInst(CIL_Inst i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(State state)
        {
            var rhs = state._stack.Pop();
            var lhs = state._stack.Pop();
            var result = binaryOp(this.GetType(), lhs, rhs);
            state._stack.Push(result);
            return Next;
        }

        class BinaryInstTable
        {
            public System.Type Op;
            public Swigged.LLVM.Opcode Opcode;
            public bool IsOverflow;
            public bool IsUnsigned;

            public BinaryInstTable(System.Type ao, Swigged.LLVM.Opcode oc, bool aIsOverflow, bool aIsUnsigned)
            {
                Op = ao;
                Opcode = oc;
                IsOverflow = aIsOverflow;
                IsUnsigned = aIsUnsigned;
            }

            // Default constructor for invalid cases
            public BinaryInstTable()
            {
            }
        }

        static List<BinaryInstTable> IntMap = new List<BinaryInstTable>()
        {
            new BinaryInstTable(typeof(i_add), Opcode.Add, false, false), // ADD
            new BinaryInstTable(typeof(i_add_ovf), Opcode.Add, true, false), // ADD_OVF
            new BinaryInstTable(typeof(i_add_ovf_un), Opcode.Add, true, true), // ADD_OVF_UN
            new BinaryInstTable(typeof(i_and), Opcode.And, false, false), // AND
            new BinaryInstTable(typeof(i_div), Opcode.SDiv, false, false), // DIV
            new BinaryInstTable(typeof(i_div_un), Opcode.UDiv, false, true), // DIV_UN
            new BinaryInstTable(typeof(i_mul), Opcode.Mul, false, false), // MUL
            new BinaryInstTable(typeof(i_mul_ovf), Opcode.Mul, true, false), // MUL_OVF
            new BinaryInstTable(typeof(i_mul_ovf_un), Opcode.Mul, true, true), // MUL_OVF_UN
            new BinaryInstTable(typeof(i_or), Opcode.Or, false, false), // OR
            new BinaryInstTable(typeof(i_rem), Opcode.SRem, false, false), // REM
            new BinaryInstTable(typeof(i_rem_un), Opcode.SRem, false, true), // REM_UN
            new BinaryInstTable(typeof(i_sub), Opcode.Sub, false, false), // SUB
            new BinaryInstTable(typeof(i_sub_ovf), Opcode.Sub, true, false), // SUB_OVF
            new BinaryInstTable(typeof(i_sub_ovf_un), Opcode.Sub, true, true), // SUB_OVF_UN
            new BinaryInstTable(typeof(i_xor), Opcode.Xor, false, false) // XOR
        };

        static List<BinaryInstTable> FloatMap = new List<BinaryInstTable>()
        {
            new BinaryInstTable(typeof(i_add), Opcode.FAdd, false, false), // ADD
            new BinaryInstTable(), // ADD_OVF (invalid)
            new BinaryInstTable(), // ADD_OVF_UN (invalid)
            new BinaryInstTable(), // AND (invalid)
            new BinaryInstTable(typeof(i_div), Opcode.FDiv, false, false), // DIV
            new BinaryInstTable(), // DIV_UN (invalid)
            new BinaryInstTable(typeof(i_mul), Opcode.FMul, false, false), // MUL
            new BinaryInstTable(), // MUL_OVF (invalid)
            new BinaryInstTable(), // MUL_OVF_UN (invalid)
            new BinaryInstTable(), // OR (invalid)
            new BinaryInstTable(typeof(i_rem), Opcode.FRem, false, false), // REM
            new BinaryInstTable(), // REM_UN (invalid)
            new BinaryInstTable(typeof(i_sub), Opcode.FSub, false, false), // SUB
            new BinaryInstTable(), // SUB_OVF (invalid)
            new BinaryInstTable(), // SUB_OVF_UN (invalid)
            new BinaryInstTable(), // XOR (invalid)
        };


        Type binaryOpType(System.Type Opcode, Type Type1, Type Type2)
        {
            // Roughly follows ECMA-355, Table III.2.
            // If both types are floats, the result is the larger float type.
            if (Type1.isFloatingPointTy() && Type2.isFloatingPointTy())
            {
                UInt32 Size1a = Type1.getPrimitiveSizeInBits();
                UInt32 Size2a = Type2.getPrimitiveSizeInBits();
                return Size1a >= Size2a ? Type1 : Type2;
            }

            bool Type1IsInt = Type1.isIntegerTy();
            bool Type2IsInt = Type2.isIntegerTy();
            bool Type1IsPtr = Type1.isPointerTy();
            bool Type2IsPtr = Type2.isPointerTy();

            UInt32 Size1 =
                Type1IsPtr ? TargetPointerSizeInBits : Type1.getPrimitiveSizeInBits();
            UInt32 Size2 =
                Type2IsPtr ? TargetPointerSizeInBits : Type2.getPrimitiveSizeInBits();

            // If both types are integers, sizes must match, or one of the sizes must be
            // native int and the other must be smaller.
            if (Type1IsInt && Type2IsInt)
            {
                if (Size1 == Size2)
                {
                    return Type1;
                }
                if ((Size1 == TargetPointerSizeInBits) && (Size1 > Size2))
                {
                    return Type1;
                }
                if ((Size2 == TargetPointerSizeInBits) && (Size2 > Size1))
                {
                    return Type2;
                }
            }
            else
            {
                bool Type1IsUnmanagedPointer = GcInfo.isUnmanagedPointer(Type1);
                bool Type2IsUnmanagedPointer = GcInfo.isUnmanagedPointer(Type2);
                bool IsStrictlyAdd = (Opcode == typeof(i_add));
                bool IsAdd = IsStrictlyAdd || (Opcode == typeof(i_add_ovf)) ||
                             (Opcode == typeof(i_add_ovf_un));
                bool IsStrictlySub = (Opcode == typeof(i_sub));
                bool IsSub = IsStrictlySub || (Opcode == typeof(i_sub_ovf)) ||
                             (Opcode == typeof(i_sub_ovf_un));
                bool IsStrictlyAddOrSub = IsStrictlyAdd || IsStrictlySub;
                bool IsAddOrSub = IsAdd || IsSub;

                // If we see a mixture of int and unmanaged pointer, the result
                // is generally a native int, with a few special cases where we
                // preserve pointer-ness.
                if (Type1IsUnmanagedPointer || Type2IsUnmanagedPointer)
                {
                    // ptr +/- int = ptr
                    if (IsAddOrSub && Type1IsUnmanagedPointer && Type2IsInt &&
                        (Size1 >= Size2))
                    {
                        return Type1;
                    }
                    // int + ptr = ptr
                    if (IsAdd && Type1IsInt && Type2IsUnmanagedPointer && (Size2 >= Size1))
                    {
                        return Type2;
                    }
                    // Otherwise type result as native int as long as there's no truncation
                    // going on.
                    if ((Size1 <= TargetPointerSizeInBits) &&
                        (Size2 <= TargetPointerSizeInBits))
                    {
                        return new Type(Type.getIntNTy(LLVMContext,
                            TargetPointerSizeInBits));
                    }
                }
                else if (GcInfo.isGcPointer(Type1))
                {
                    if (IsSub && GcInfo.isGcPointer(Type2))
                    {
                        // The difference of two managed pointers is a native int.
                        return new Type(Type.getIntNTy(LLVMContext,
                            TargetPointerSizeInBits));
                    }
                    else if (IsStrictlyAddOrSub && Type2IsInt && (Size1 >= Size2))
                    {
                        // Special case for just strict add and sub: if Type1 is a managed
                        // pointer and Type2 is an integer, the result is Type1. We see the
                        // add case in some internal uses in reader base. We see the sub case
                        // in some IL stubs.
                        return Type1;
                    }
                }
            }

            // All other combinations are invalid.
            return null;
        }

        // Handle pointer + int by emitting a flattened LLVM GEP.
        Value genPointerAdd(Value Arg1, Value Arg2)
        {
            // Assume 1 is base and 2 is offset
            Value BasePtr = Arg1;
            Value Offset = Arg2;

            // Reconsider based on types.
            bool Arg1IsPointer = Arg1.T.isPointerTy();
            bool Arg2IsPointer = Arg2.T.isPointerTy();
            Debug.Assert(Arg1IsPointer || Arg2IsPointer);

            // Bail if both args are already pointer types.
            if (Arg1IsPointer && Arg2IsPointer)
            {
                return null;
            }

            // Swap base and offset if we got it wrong.
            if (Arg2IsPointer)
            {
                BasePtr = Arg2;
                Offset = Arg1;
            }

            // Bail if offset is not integral.
            Type OffsetTy = Offset.T;
            if (!OffsetTy.isIntegerTy())
            {
                return null;
            }

            // Build an LLVM GEP for the resulting address.
            // For now we "flatten" to byte offsets.
            Type CharPtrTy = new Type(Type.getInt8PtrTy(
                LLVMContext,
                BasePtr.T.getPointerAddressSpace()));
            Value BasePtrCast = new Value(LLVM.BuildBitCast(Builder, BasePtr.V, CharPtrTy.T, ""));
            Value ResultPtr = new Value(LLVM.BuildInBoundsGEP(Builder, BasePtrCast.V, new ValueRef[] {Offset.V}, ""));
            return ResultPtr;
        }

        // Handle pointer - int by emitting a flattened LLVM GEP.
        Value genPointerSub(Value Arg1, Value Arg2)
        {

            // Assume 1 is base and 2 is offset
            Value BasePtr = Arg1;
            Value Offset = Arg2;

            // Reconsider based on types.
            bool Arg1IsPointer = Arg1.T.isPointerTy();
            bool Arg2IsPointer = Arg2.T.isPointerTy();
            Debug.Assert(Arg1IsPointer);

            // Bail if both args are already pointer types.
            if (Arg1IsPointer && Arg2IsPointer)
            {
                return null;
            }

            // Bail if offset is not integral.
            Type OffsetTy = Offset.T;
            if (!OffsetTy.isIntegerTy())
            {
                return null;
            }

            // Build an LLVM GEP for the resulting address.
            // For now we "flatten" to byte offsets.
            Type CharPtrTy = new Type(Type.getInt8PtrTy(
                LLVMContext, BasePtr.T.getPointerAddressSpace()));
            Value BasePtrCast = new Value(LLVM.BuildBitCast(Builder, BasePtr.V, CharPtrTy.T, ""));
            Value NegOffset = new Value(LLVM.BuildNeg(Builder, Offset.V, ""));
            Value ResultPtr = new Value(LLVM.BuildGEP(Builder, BasePtrCast.V, new ValueRef[] { NegOffset.V }, ""));
            return ResultPtr;
        }

        // This method only handles basic arithmetic conversions for use in
        // binary operations.
        public Value convert(Type Ty, Value Node, bool SourceIsSigned)
        {
            Type SourceTy = Node.T;
            Value Result = null;

            if (Ty == SourceTy)
            {
                Result = Node;
            }
            else if (SourceTy.isIntegerTy() && Ty.isIntegerTy())
            {
                Result = new Value(LLVM.BuildIntCast(Builder, Node.V, Ty.T, ""));//SourceIsSigned);
            }
            else if (SourceTy.isFloatingPointTy() && Ty.isFloatingPointTy())
            {
                Result = new Value(LLVM.BuildFPCast(Builder, Node.V, Ty.T, ""));
            }
            else if (SourceTy.isPointerTy() && Ty.isIntegerTy())
            {
                Result = new Value(LLVM.BuildPtrToInt(Builder, Node.V, Ty.T, ""));
            }
            else
            {
                Debug.Assert(false);
            }

            return Result;
        }

        Value binaryOp(System.Type Opcode, Value Arg1, Value Arg2)
        {
            Type Type1 = Arg1.T;
            Type Type2 = Arg2.T;
            Type ResultType = binaryOpType(Opcode, Type1, Type2);
            Type ArithType = ResultType;

            // If the result is a pointer, see if we have simple
            // pointer + int op...
            if (ResultType.isPointerTy())
            {
                if (Opcode == typeof(i_add))
                {
                    Value PtrAdd = genPointerAdd(Arg1, Arg2);
                    if (PtrAdd != null)
                    {
                        return PtrAdd;
                    }
                }
                else if (Opcode == typeof(i_add_ovf_un))
                {
                    Value PtrSub = genPointerSub(Arg1, Arg2);
                    if (PtrSub != null)
                    {
                        return PtrSub;
                    }
                }
                else if (Opcode == typeof(i_sub_ovf_un))
                { 
                    // Arithmetic with overflow must use an appropriately-sized integer to
                    // perform the arithmetic, then convert the result back to the pointer
                    // type.
                    ArithType = new Type(Type.getIntNTy(LLVMContext, TargetPointerSizeInBits));
                }
            }

            Debug.Assert(ArithType == ResultType || ResultType.isPointerTy());

            bool IsFloat = ResultType.isFloatingPointTy();
            List<BinaryInstTable> Triple = IsFloat ? FloatMap : IntMap;

            bool IsOverflow = Triple.Where(trip => Opcode == trip.Op).Select(trip => trip.IsOverflow).First();
            bool IsUnsigned = Triple.Where(trip => Opcode == trip.Op).Select(trip => trip.IsUnsigned).First();

            if (Type1 != ArithType)
            {
                Arg1 = convert(ArithType, Arg1, !IsUnsigned);
            }

            if (Type2 != ArithType)
            {
                Arg2 = convert(ArithType, Arg2, !IsUnsigned);
            }

            Value Result;
            //if (IsFloat && Opcode == typeof(i_rem))
            //{
            //    // FRem must be lowered to a JIT helper call to avoid undefined symbols
            //    // during emit.
            //    //
            //    // TODO: it may be possible to delay this lowering by updating the JIT
            //    // APIs to allow the definition of a target library (via TargeLibraryInfo).
            //    CorInfoHelpFunc Helper = CORINFO_HELP_UNDEF;
            //    if (ResultType.isFloatTy())
            //    {
            //        Helper = CORINFO_HELP_FLTREM;
            //    }
            //    else if (ResultType.isDoubleTy())
            //    {
            //        Helper = CORINFO_HELP_DBLREM;
            //    }
            //    else
            //    {
            //        llvm_unreachable("Bad floating point type!");
            //    }

            //    const bool MayThrow = false;
            //    Result = (Value)callHelperImpl(Helper, MayThrow, ResultType, Arg1, Arg2)
            //    .getInstruction();
            //}
            //else
            //if (IsOverflow)
            //{
            //    // Call the appropriate intrinsic.  Its result is a pair of the arithmetic
            //    // result and a bool indicating whether the operation overflows.
            //    Value Intrinsic = Intrinsic::getDeclaration(
            //        JitContext.CurrentModule, Triple[Opcode].Op.Intrinsic, ArithType);
            //    Value[] Args = new Value[] { Arg1, Arg2 };
            //    const bool MayThrow = false;
            //    Value Pair = makeCall(Intrinsic, MayThrow, Args).getInstruction();

            //    // Extract the bool and raise an overflow exception if set.
            //    Value OvfBool = new Value(LLVM.BuildExtractValue(Builder, Pair.V, 1, "Ovf"));
            //    genConditionalThrow(OvfBool, CORINFO_HELP_OVERFLOW, "ThrowOverflow");

            //    // Extract the result.
            //    Result = new Value(LLVM.BuildExtractValue(Builder, Pair.V, 0, ""));
            //}
            //else
            {
                // Create a simple binary operation.
                BinaryInstTable OpI = Triple.Find(t => t.Op == Opcode);

                if (Opcode == typeof(i_div) ||
                    Opcode == typeof(i_div_un) ||
                    Opcode == typeof(i_rem) ||
                    Opcode == typeof(i_rem_un))
                {
                    // Integer divide and remainder throw a DivideByZeroException
                    // if the divisor is zero
                    if (UseExplicitZeroDivideChecks)
                    {
                        Value IsZero = new Value(LLVM.BuildIsNull(Builder, Arg2.V, ""));
                        //genConditionalThrow(IsZero, CORINFO_HELP_THROWDIVZERO, "ThrowDivideByZero");
                    }
                    else
                    {
                        // This configuration isn't really supported.  To support it we'd
                        // need to annotate the divide we're about to generate as possibly
                        // throwing an exception (that would be raised from a machine trap).
                    }
                }

                Result = new Value(LLVM.BuildBinOp(Builder, OpI.Opcode, Arg1.V, Arg2.V, ""));
            }

            if (ResultType != ArithType)
            {
                Debug.Assert(ResultType.isPointerTy());
                Debug.Assert(ArithType.isIntegerTy());

                Result = new Value(LLVM.BuildIntToPtr(Builder, Result.V, ResultType.T, ""));
            }

            return Result;
        }

        public bool UseExplicitZeroDivideChecks { get; set; }
    }

    public class LoadArgInst : Inst
    {
        public LoadArgInst(CIL_Inst i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(State state)
        {
            int _arg = (this.Instruction as CIL.LoadArgInst)._arg;
            Value value = state._arguments[_arg];
            state._stack.Push(value);
            return Next;
        }
    }

    public class LDCInstI4 : Inst
    {
        public LDCInstI4(CIL_Inst i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(State state)
        {
            Int32 arg = (this.Instruction as CIL.LDCInst4)._arg;
            Value value = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)arg, true));
            state._stack.Push(value);
            return Next;
        }
    }

    public class LDCInstI8 : Inst
    {
        public LDCInstI8(CIL_Inst i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(State state)
        {
            var arg = (this.Instruction as CIL.LDCInst8)._arg;
            Value value = new Value(LLVM.ConstInt(LLVM.Int64Type(), (ulong)arg, true));
            state._stack.Push(value);
            return Next;
        }
    }

    public class LdLoc : Inst
    {
        public LdLoc(CIL_Inst i) : base(i)
        {
        }

        public override Inst Convert(State state)
        {
            var arg = (this.Instruction as CIL.LdLoc)._arg;
            Value v = state._locals[arg];
            state._stack.Push(v);
            return Next;
        }
    }

    public class StLoc : Inst
    {
        public StLoc(CIL_Inst i) : base(i)
        {
        }

        public override Inst Convert(State state)
        {
            Value v = state._stack.Pop();
            var arg = (this.Instruction as CIL.StLoc)._arg;
            state._locals[arg] = v;
            return Next;
        }
    }

    public class CompareInst : Inst
    {
        public CompareInst(CIL_Inst i) : base(i)
        {
        }

        public enum PredicateType
        {
            eq,
            ne,
            gt,
            lt,
            ge,
            le,
        };

        public Swigged.LLVM.IntPredicate[] _int_pred = new Swigged.LLVM.IntPredicate[]
        {
            Swigged.LLVM.IntPredicate.IntEQ,
            Swigged.LLVM.IntPredicate.IntNE,
            Swigged.LLVM.IntPredicate.IntSGT,
            Swigged.LLVM.IntPredicate.IntSLT,
            Swigged.LLVM.IntPredicate.IntSGE,
            Swigged.LLVM.IntPredicate.IntSLE,
        };

        public Swigged.LLVM.IntPredicate[] _uint_pred = new Swigged.LLVM.IntPredicate[]
        {
            Swigged.LLVM.IntPredicate.IntEQ,
            Swigged.LLVM.IntPredicate.IntNE,
            Swigged.LLVM.IntPredicate.IntUGT,
            Swigged.LLVM.IntPredicate.IntULT,
            Swigged.LLVM.IntPredicate.IntUGE,
            Swigged.LLVM.IntPredicate.IntULE,
        };

        public virtual PredicateType Predicate { get; set; }

        public override Inst Convert(State state)
        {
            Value v2 = state._stack.Pop();
            Value v1 = state._stack.Pop();
            // TODO Undoubtably, this will be much more complicated than my initial stab.
            Type t1 = v1.T;
            Type t2 = v2.T;
            ValueRef cmp = default(ValueRef);
            // Deal with various combinations of types.
            if (t1.isIntegerTy() && t2.isIntegerTy())
            {
                var op = _int_pred[(int)Predicate];
                cmp = LLVM.BuildICmp(Builder, op, v1.V, v2.V, "");
                if (Next == null) return null;
                var t = Next.GetType();
                if (t == typeof(i_brfalse))
                {
                    // Push, Pop, branch -> combine
                }
                else if (t == typeof(i_brfalse_s))
                {
                    // Push, Pop, branch -> combine
                }
                else if (t == typeof(i_brtrue))
                {
                    // Push, Pop, branch -> combine
                }
                else if (t == typeof(i_brtrue_s))
                {
                    // Push, Pop, branch -> combine
                } else
                {
                    // Set up for push of 0/1.
                    var ret = LLVM.BuildZExt(Builder, cmp, LLVM.Int32Type(), "");
                    state._stack.Push(new Value(ret, LLVM.Int32Type()));
                }
            }
            return Next;
        }
    }

    public class i_add : BinaryOpInst
    {
        public i_add(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_add_ovf : BinaryOpInst
        {
        public i_add_ovf(CIL_Inst i)
            : base(i)
        {
        }
        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_add_ovf_un : BinaryOpInst
        {
        public i_add_ovf_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_and : BinaryOpInst
        {
        public i_and(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_arglist : Inst
    {
        public i_arglist(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_beq : Inst
    {
        public i_beq(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_beq_s : CompareInst
    {
        public i_beq_s(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.eq;
        }
    }

    public class i_bge : CompareInst
    {
        public i_bge(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.ne;
        }
    }

    public class i_bge_un : Inst
    {
        public i_bge_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_bge_un_s : Inst
    {
        public i_bge_un_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_bge_s : CompareInst
    {
        public i_bge_s(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.ge;
        }
    }

    public class i_bgt : Inst
    {
        public i_bgt(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_bgt_s : CompareInst
    {
        public i_bgt_s(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.gt;
        }
    }

    public class i_bgt_un : Inst
    {
        public i_bgt_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_bgt_un_s : Inst
    {
        public i_bgt_un_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ble : Inst
    {
        public i_ble(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ble_s : CompareInst
    {
        public i_ble_s(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.le;
        }
    }

    public class i_ble_un : Inst
    {
        public i_ble_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ble_un_s : Inst
    {
        public i_ble_un_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_blt : Inst
    {
        public i_blt(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_blt_s : CompareInst
    {
        public i_blt_s(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.lt;
        }
    }

    public class i_blt_un : Inst
    {
        public i_blt_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_blt_un_s : Inst
    {
        public i_blt_un_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_bne_un : Inst
    {
        public i_bne_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_bne_un_s : CompareInst
    {
        public i_bne_un_s(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.ne;
        }
    }

    public class i_box : Inst
    {
        public i_box(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_br : Inst
    {
        public i_br(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_br_s : Inst
    {
        public i_br_s(CIL_Inst i)
            : base(i)
        {
        }

        public override Inst Convert(State state)
        {
            GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>.Edge edge = Block._Successors[0];
            int succ = edge.To;
            var s = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ)];
            var br = LLVM.BuildBr(Builder, s.BasicBlock);
            return Next;
        }
    }

    public class i_brfalse : Inst
    {
        public i_brfalse(CIL_Inst i)
            : base(i)
        {
        }

        public override Inst Convert(State state)
        {
            var v = state._stack.Pop();
            GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>.Edge edge1 = Block._Successors[0];
            GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>.Edge edge2 = Block._Successors[1];
            int succ1 = edge1.To;
            int succ2 = edge1.To;
            var s1 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
            var s2 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
            LLVM.BuildCondBr(Builder, v.V, s1.BasicBlock, s2.BasicBlock);
            return Next;
        }
    }

    public class i_break : Inst
    {
        public i_break(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_brfalse_s : Inst
    {
        public i_brfalse_s(CIL_Inst i)
            : base(i)
        {
        }

        public override Inst Convert(State state)
        {
            var v = state._stack.Pop();
            GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>.Edge edge1 = Block._Successors[0];
            GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>.Edge edge2 = Block._Successors[1];
            int succ1 = edge1.To;
            int succ2 = edge2.To;
            var s1 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
            var s2 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ2)];
            // We need to compare the value popped with 0/1.
            var v2 = LLVM.ConstInt(LLVM.Int32Type(), 1, false);
            var v3 = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, v.V, v2, "");
            LLVM.BuildCondBr(Builder, v3, s1.BasicBlock, s2.BasicBlock);
            return Next;
        }
    }

    public class i_brtrue : Inst
    {
        public i_brtrue(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_brtrue_s : Inst
    {
        public i_brtrue_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_call : Inst
    {
        public i_call(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_calli : Inst
    {
        public i_calli(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_callvirt : Inst
    {
        public i_callvirt(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_castclass : Inst
    {
        public i_castclass(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ceq : CompareInst
    {
        public i_ceq(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.eq;
        }
    }

    public class i_cgt : CompareInst
    {
        public i_cgt(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.gt;
        }
    }

    public class i_cgt_un : CompareInst
    {
        public i_cgt_un(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.gt;
        }
    }

    public class i_ckfinite : Inst
    {
        public i_ckfinite(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_clt : CompareInst
    {
        public i_clt(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.lt;
        }
    }

    public class i_clt_un : CompareInst
    {
        public i_clt_un(CIL_Inst i)
            : base(i)
        {
            Predicate = PredicateType.lt;
        }
    }

    public class i_constrained : Inst
    {
        public i_constrained(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_i1 : Inst
    {
        public i_conv_i1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_i2 : Inst
    {
        public i_conv_i2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_i4 : Inst
    {
        public i_conv_i4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_i8 : Inst
    {
        public i_conv_i8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_i : Inst
    {
        public i_conv_i(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i1 : Inst
    {
        public i_conv_ovf_i1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i1_un : Inst
    {
        public i_conv_ovf_i1_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i2 : Inst
    {
        public i_conv_ovf_i2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i2_un : Inst
    {
        public i_conv_ovf_i2_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i4 : Inst
    {
        public i_conv_ovf_i4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i4_un : Inst
    {
        public i_conv_ovf_i4_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i8 : Inst
    {
        public i_conv_ovf_i8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i8_un : Inst
    {
        public i_conv_ovf_i8_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i : Inst
    {
        public i_conv_ovf_i(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_i_un : Inst
    {
        public i_conv_ovf_i_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u1 : Inst
    {
        public i_conv_ovf_u1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u1_un : Inst
    {
        public i_conv_ovf_u1_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u2 : Inst
    {
        public i_conv_ovf_u2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u2_un : Inst
    {
        public i_conv_ovf_u2_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u4 : Inst
    {
        public i_conv_ovf_u4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u4_un : Inst
    {
        public i_conv_ovf_u4_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u8 : Inst
    {
        public i_conv_ovf_u8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u8_un : Inst
    {
        public i_conv_ovf_u8_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u : Inst
    {
        public i_conv_ovf_u(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_ovf_u_un : Inst
    {
        public i_conv_ovf_u_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_r4 : Inst
    {
        public i_conv_r4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_r8 : Inst
    {
        public i_conv_r8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_r_un : Inst
    {
        public i_conv_r_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_u1 : Inst
    {
        public i_conv_u1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_u2 : Inst
    {
        public i_conv_u2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_u4 : Inst
    {
        public i_conv_u4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_u8 : Inst
    {
        public i_conv_u8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_conv_u : Inst
    {
        public i_conv_u(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_cpblk : Inst
    {
        public i_cpblk(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_cpobj : Inst
    {
        public i_cpobj(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_div : BinaryOpInst
        {
        public i_div(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_div_un : BinaryOpInst
        {
        public i_div_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_dup : Inst
    {
        public i_dup(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_endfilter : Inst
    {
        public i_endfilter(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_endfinally : Inst
    {
        public i_endfinally(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_initblk : Inst
    {
        public i_initblk(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_initobj : Inst
    {
        public i_initobj(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_isinst : Inst
    {
        public i_isinst(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_jmp : Inst
    {
        public i_jmp(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarg : LoadArgInst
    {

        public i_ldarg(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarg_0 : LoadArgInst
    {
        public i_ldarg_0(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarg_1 : LoadArgInst
    {
        public i_ldarg_1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarg_2 : LoadArgInst
    {
        public i_ldarg_2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarg_3 : LoadArgInst
    {
        public i_ldarg_3(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarg_s : LoadArgInst
    {
        public i_ldarg_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarga : LoadArgInst
    {
        public i_ldarga(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldarga_s : LoadArgInst
    {
        public i_ldarga_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldc_i4 : LDCInstI4
    {
        public i_ldc_i4(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_0 : LDCInstI4
    {
        public i_ldc_i4_0(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_1 : LDCInstI4
    {
        public i_ldc_i4_1(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_2 : LDCInstI4
    {
        public i_ldc_i4_2(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_3 : LDCInstI4
    {
        public i_ldc_i4_3(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_4 : LDCInstI4
    {
        public i_ldc_i4_4(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_5 : LDCInstI4
    {
        public i_ldc_i4_5(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_6 : LDCInstI4
    {
        public i_ldc_i4_6(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_7 : LDCInstI4
    {
        public i_ldc_i4_7(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_8 : LDCInstI4
    {
        public i_ldc_i4_8(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_m1 : LDCInstI4
    {
        public i_ldc_i4_m1(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i4_s : LDCInstI4
    {
        public i_ldc_i4_s(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_i8 : LDCInstI8
    {
        public i_ldc_i8(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldc_r4 : Inst
    {
        public i_ldc_r4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldc_r8 : Inst
    {
        public i_ldc_r8(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldelem_any : Inst
    {
        public i_ldelem_any(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_i1 : Inst
    {
        public i_ldelem_i1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_i2 : Inst
    {
        public i_ldelem_i2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_i4 : Inst
    {
        public i_ldelem_i4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_i8 : Inst
    {
        public i_ldelem_i8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_i : Inst
    {
        public i_ldelem_i(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_r4 : Inst
    {
        public i_ldelem_r4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_r8 : Inst
    {
        public i_ldelem_r8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_ref : Inst
    {
        public i_ldelem_ref(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_u1 : Inst
    {
        public i_ldelem_u1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_u2 : Inst
    {
        public i_ldelem_u2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelem_u4 : Inst
    {
        public i_ldelem_u4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldelema : Inst
    {
        public i_ldelema(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldfld : Inst
    {
        public i_ldfld(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldflda : Inst
    {
        public i_ldflda(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldftn : Inst
    {
        public i_ldftn(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_i1 : Inst
    {
        public i_ldind_i1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_i2 : Inst
    {
        public i_ldind_i2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_i4 : Inst
    {
        public i_ldind_i4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_i8 : Inst
    {
        public i_ldind_i8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_i : Inst
    {
        public i_ldind_i(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_r4 : Inst
    {
        public i_ldind_r4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_r8 : Inst
    {
        public i_ldind_r8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_ref : Inst
    {
        public i_ldind_ref(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_u1 : Inst
    {
        public i_ldind_u1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_u2 : Inst
    {
        public i_ldind_u2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldind_u4 : Inst
    {
        public i_ldind_u4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldlen : Inst
    {
        public i_ldlen(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldloc : LdLoc
    {
        public i_ldloc(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldloc_0 : LdLoc
    {
        public i_ldloc_0(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldloc_1 : LdLoc
    {
        public i_ldloc_1(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldloc_2 : LdLoc
    {
        public i_ldloc_2(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldloc_3 : LdLoc
    {
        public i_ldloc_3(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldloc_s : LdLoc
    {
        public i_ldloc_s(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldloca : LdLoc
    {
        public i_ldloca(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldloca_s : LdLoc
    {
        public i_ldloca_s(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_ldnull : Inst
    {
        public i_ldnull(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldobj : Inst
    {
        public i_ldobj(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldsfld : Inst
    {
        public i_ldsfld(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldsflda : Inst
    {
        public i_ldsflda(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldstr : Inst
    {
        public i_ldstr(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldtoken : Inst
    {
        public i_ldtoken(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ldvirtftn : Inst
    {
        public i_ldvirtftn(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_leave : Inst
    {
        public i_leave(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_leave_s : Inst
    {
        public i_leave_s(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_localloc : Inst
    {
        public i_localloc(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_mkrefany : Inst
    {
        public i_mkrefany(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_mul : BinaryOpInst
        {
        public i_mul(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_mul_ovf : BinaryOpInst
        {
        public i_mul_ovf(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_mul_ovf_un : BinaryOpInst
        {
        public i_mul_ovf_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_neg : Inst
    {
        public i_neg(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_newarr : Inst
    {
        public i_newarr(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_newobj : Inst
    {
        public i_newobj(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_no : Inst
    {
        public i_no(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_nop : Inst
    {
        public i_nop(CIL_Inst i)
            : base(i)
        {
        }
        public override Inst Convert(State state)
        {
            return Next;
        }
    }

    public class i_not : Inst
    {
        public i_not(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_or : BinaryOpInst
        {
        public i_or(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_pop : Inst
    {
        public i_pop(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_readonly : Inst
    {
        public i_readonly(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_refanytype : Inst
    {
        public i_refanytype(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_refanyval : Inst
    {
        public i_refanyval(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_rem : BinaryOpInst
        {
        public i_rem(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_rem_un : BinaryOpInst
        {
        public i_rem_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_ret : Inst
    {
        public i_ret(CIL_Inst i)
            : base(i)
        {
        }

        public override Inst Convert(State state)
        {
            // There are really two different stacks here:
            // one for the called method, and the other for the caller of the method.
            // When returning, the stack of the method is pretty much unchanged.
            // In fact the top of stack often contains the return value from the method.
            // Back in the caller, the stack is popped of all arguments to the callee.
            // And, the return value is pushed on the top of stack.
            // This is handled by the call instruction.
            var v = state._stack.Pop();
            var i = LLVM.BuildRet(Builder, v.V);
            return Next;
        }
    }

    public class i_rethrow : Inst
    {
        public i_rethrow(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_shl : Inst
    {
        public i_shl(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_shr : Inst
    {
        public i_shr(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_shr_un : Inst
    {
        public i_shr_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_sizeof : Inst
    {
        public i_sizeof(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_starg : Inst
    {
        int _arg;

        public i_starg(CIL_Inst i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_starg_s : Inst
    {
        int _arg;

        public i_starg_s(CIL_Inst i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stelem_any : Inst
    {
        public i_stelem_any(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_i1 : Inst
    {
        public i_stelem_i1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_i2 : Inst
    {
        public i_stelem_i2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_i4 : Inst
    {
        public i_stelem_i4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_i8 : Inst
    {
        public i_stelem_i8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_i : Inst
    {
        public i_stelem_i(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_r4 : Inst
    {
        public i_stelem_r4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_r8 : Inst
    {
        public i_stelem_r8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stelem_ref : Inst
    {
        public i_stelem_ref(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stfld : Inst
    {
        public i_stfld(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_i1 : Inst
    {
        public i_stind_i1(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_i2 : Inst
    {
        public i_stind_i2(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_i4 : Inst
    {
        public i_stind_i4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_i8 : Inst
    {
        public i_stind_i8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_i : Inst
    {
        public i_stind_i(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_r4 : Inst
    {
        public i_stind_r4(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_r8 : Inst
    {
        public i_stind_r8(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stind_ref : Inst
    {
        public i_stind_ref(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stloc : StLoc
    {
        public i_stloc(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_stloc_0 : StLoc
    {
        public i_stloc_0(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_stloc_1 : StLoc
    {
        public i_stloc_1(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_stloc_2 : StLoc
    {
        public i_stloc_2(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_stloc_3 : StLoc
    {
        public i_stloc_3(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_stloc_s : StLoc
    {
        public i_stloc_s(CIL_Inst i) : base(i)
        {
        }
    }

    public class i_stobj : Inst
    {
        public i_stobj(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_stsfld : Inst
    {
        public i_stsfld(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_sub : BinaryOpInst
        {
        public i_sub(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_sub_ovf : BinaryOpInst
        {
        public i_sub_ovf(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_sub_ovf_un : BinaryOpInst
        {
        public i_sub_ovf_un(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_switch : Inst
    {
        public i_switch(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_tail : Inst
    {
        public i_tail(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_throw : Inst
    {
        public i_throw(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_unaligned : Inst
    {
        public i_unaligned(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_unbox : Inst
    {
        public i_unbox(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_unbox_any : Inst
    {
        public i_unbox_any(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_volatile : Inst
    {
        public i_volatile(CIL_Inst i)
            : base(i)
        {
        }
    }

    public class i_xor : BinaryOpInst
        {
        public i_xor(CIL_Inst i)
            : base(i)
        {
        }
    }
}
