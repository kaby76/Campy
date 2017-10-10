using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Campy.Graphs;
using Campy.Utils;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Swigged.LLVM;

namespace Campy.Compiler
{
    /// <summary>
    /// Wrapper for CIL instructions that are implemented using Mono.Cecil.Cil.
    /// This class adds basic block graph structure on top of these instructions. There
    /// is no semantics encoded in the wrapper.
    /// </summary>
    public class Inst
    {
        // Required for Mono to bb conversion.
        public Mono.Cecil.Cil.Instruction Instruction { get; private set; }
        public static List<Inst> CallInstructions { get; private set; } = new List<Inst>();
        public override string ToString() { return Instruction.ToString(); }
        public Mono.Cecil.Cil.OpCode OpCode { get { return Instruction.OpCode; } }
        public object Operand { get { return Instruction.Operand; } }


        // Required for LLVM conversion.
        public BuilderRef Builder { get { return Block.Builder; } }
        public ContextRef LLVMContext { get; set; }
        public List<Value> LLVMInstructions { get; private set; }
        public CFG.Vertex Block { get; set; }
        // Required instruction sequencing so we can translate groups of instructions.
        public virtual Inst Next { get; set; }

        public virtual void ComputeStackLevel(ref int level_after)
        {
            throw new Exception("Must have an implementation for ComputeStackLevel! The instruction is: "
                + this.ToString());
        }

        public virtual Inst Convert(Converter converter, State state)
        {
            throw new Exception("Must have an implementation for Convert! The instruction is: "
                                + this.ToString());
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


        public Inst(Mono.Cecil.Cil.Instruction i)
        {
            Instruction = i;
            if (i.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Call)
            {
                Inst.CallInstructions.Add(this);
            }
        }
        static public Inst Wrap(Mono.Cecil.Cil.Instruction i)
        {
            // Wrap instruction with semantics, def/use/kill properties.
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
    }

    public class BinaryOpInst : Inst
    {
        public BinaryOpInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var lhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(lhs);

            var result = binaryOp(this.GetType(), lhs, rhs);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(result);

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
                        return new Type(Type.getIntNTy(LLVM.GetModuleContext(Converter.global_llvm_module),
                            TargetPointerSizeInBits));
                    }
                }
                else if (GcInfo.isGcPointer(Type1))
                {
                    if (IsSub && GcInfo.isGcPointer(Type2))
                    {
                        // The difference of two managed pointers is a native int.
                        return new Type(Type.getIntNTy(LLVM.GetModuleContext(Converter.global_llvm_module),
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
                LLVM.GetModuleContext(Converter.global_llvm_module),
                BasePtr.T.getPointerAddressSpace()));
            Value BasePtrCast = new Value(LLVM.BuildBitCast(Builder, BasePtr.V, CharPtrTy.IntermediateType, ""));
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
                LLVM.GetModuleContext(Converter.global_llvm_module), BasePtr.T.getPointerAddressSpace()));
            Value BasePtrCast = new Value(LLVM.BuildBitCast(Builder, BasePtr.V, CharPtrTy.IntermediateType, ""));
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
                Result = new Value(LLVM.BuildIntCast(Builder, Node.V, Ty.IntermediateType, ""));//SourceIsSigned);
            }
            else if (SourceTy.isFloatingPointTy() && Ty.isFloatingPointTy())
            {
                Result = new Value(LLVM.BuildFPCast(Builder, Node.V, Ty.IntermediateType, ""));
            }
            else if (SourceTy.isPointerTy() && Ty.isIntegerTy())
            {
                Result = new Value(LLVM.BuildPtrToInt(Builder, Node.V, Ty.IntermediateType, ""));
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
                    ArithType = new Type(Type.getIntNTy(LLVM.GetModuleContext(Converter.global_llvm_module), TargetPointerSizeInBits));
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

                Result = new Value(LLVM.BuildIntToPtr(Builder, Result.V, ResultType.IntermediateType, ""));
            }

            return Result;
        }

        public bool UseExplicitZeroDivideChecks { get; set; }
    }

    /// <summary>
    /// The LoadArgInst is a class for representing Load Arg instructions. The purpose to
    /// provide a representation of the arg operand of the instruction.
    /// </summary>
    public class LoadArgInst : Inst
    {
        public int _arg;

        public LoadArgInst(Mono.Cecil.Cil.Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value value = state._arguments[_arg];
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value.ToString());

            state._stack.Push(value);
            return Next;
        }
    }

    /// <summary>
    /// The LDCInstI4 and LDCInstI8 are classes for representing load constant instructions. The constant
    /// of the instruction is encoded here.
    /// </summary>
    public class LDCInstI4 : Inst
    {
        public Int32 _arg;

        public LDCInstI4(Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value value = new Value(LLVM.ConstInt(LLVM.Int32Type(), (ulong)_arg, true));
            state._stack.Push(value);
            return Next;
        }
    }

    public class LDCInstI8 : Inst
    {
        public Int64 _arg;

        public LDCInstI8(Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value value = new Value(LLVM.ConstInt(LLVM.Int64Type(), (ulong)_arg, true));
            state._stack.Push(value);
            return Next;
        }
    }

    public class LDCInstR4 : Inst
    {
        public double _arg;

        public LDCInstR4(Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value value = new Value(LLVM.ConstReal(LLVM.FloatType(), _arg));
            state._stack.Push(value);
            return Next;
        }
    }

    public class LDCInstR8 : Inst
    {
        public double _arg;

        public LDCInstR8(Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value value = new Value(LLVM.ConstReal(LLVM.DoubleType(), _arg));
            state._stack.Push(value);
            return Next;
        }
    }

    /// <summary>
    /// The LdLoc is a class for representing load local instructions.
    /// </summary>
    public class LdLoc : Inst
    {
        public int _arg;

        public LdLoc(Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var bb = this.Block;
            var mn = bb.ExpectedCalleeSignature.FullName;
            if (mn == "System.Int32 Campy.Index::op_Implicit(Campy.Index)")
            {
                //threadId
                var tidx = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.tid.x"];
                var tidy = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.tid.y"];
                var tidz = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.tid.z"];

                //blockIdx
                var ctaidx = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.ctaid.x"];
                var ctaidy = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.ctaid.y"];
                var ctaidz = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.ctaid.z"];

                //blockDim
                var ntidx = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.ntid.x"];
                var ntidy = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.ntid.y"];
                var ntidz = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.ntid.z"];

                //gridDim
                var nctaidx = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.nctaid.x"];
                var nctaidy = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.nctaid.y"];
                var nctaidz = Converter.built_in_functions["llvm.nvvm.read.ptx.sreg.nctaid.z"];

                var v_tidx = LLVM.BuildCall(bb.Builder, tidx, new ValueRef[] { }, "tidx");
                var v_tidy = LLVM.BuildCall(bb.Builder, tidy, new ValueRef[] { }, "tidy");
                var v_ntidx = LLVM.BuildCall(bb.Builder, ntidx, new ValueRef[] { }, "ntidx");
                var v_ntidy = LLVM.BuildCall(bb.Builder, ntidy, new ValueRef[] { }, "ntidy");
                var v_ctaidx = LLVM.BuildCall(bb.Builder, ctaidx, new ValueRef[] { }, "ctaidx");
                var v_ctaidy = LLVM.BuildCall(bb.Builder, ctaidy, new ValueRef[] { }, "ctaidx");
                var v_nctaidx = LLVM.BuildCall(bb.Builder, nctaidx, new ValueRef[] { }, "nctaidx");

                //int i = (threadIdx.x
                //         + blockDim.x * blockIdx.x
                //         + blockDim.x * gridDim.x * blockDim.y * blockIdx.y
                //         + blockDim.x * gridDim.x * threadIdx.y);

                var t1 = v_tidx;

                var t2 = LLVM.BuildMul(bb.Builder, v_ntidx, v_ctaidx, "");

                var t3 = LLVM.BuildMul(bb.Builder, v_ntidx, v_nctaidx, "");
                t3 = LLVM.BuildMul(bb.Builder, t3, v_ntidy, "");
                t3 = LLVM.BuildMul(bb.Builder, t3, v_ctaidy, "");

                var t4 = LLVM.BuildMul(bb.Builder, v_ntidx, v_nctaidx, "");
                t4 = LLVM.BuildMul(bb.Builder, t4, v_tidy, "");

                var sum = LLVM.BuildAdd(bb.Builder, t1, t2, "");
                sum = LLVM.BuildAdd(bb.Builder, sum, t3, "");
                sum = LLVM.BuildAdd(bb.Builder, sum, t4, "");

                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("load " + new Value(sum));
                state._stack.Push(new Value(sum));
            }
            else
            {
                Value v = state._locals[_arg];
                state._stack.Push(v);
            }
            return Next;
        }
    }

    /// <summary>
    /// The StLoc is a class for representing store local instructions.
    /// </summary>
    public class StLoc : Inst
    {
        public int _arg;

        public StLoc(Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value v = state._stack.Pop();
            state._locals[_arg] = v;
            return Next;
        }
    }


    public class CompareInst : Inst
    {
        public CompareInst(Mono.Cecil.Cil.Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after -= 1;
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
        public virtual bool IsSigned { get; set; }

        public override Inst Convert(Converter converter, State state)
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
                IntPredicate op;
                if (IsSigned) op = _int_pred[(int)Predicate];
                else op = _uint_pred[(int)Predicate];

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
                }
                else
                {
                    // Set up for push of 0/1.
                    var return_type = new Type(typeof(bool));
                    var ret_llvm = LLVM.BuildZExt(Builder, cmp, return_type.IntermediateType, "");
                    var ret = new Value(ret_llvm, return_type);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(ret);

                    state._stack.Push(ret);
                }
            }
            return Next;
        }
    }

    public class CompareAndBranchInst : Inst
    {
        public CompareAndBranchInst(Mono.Cecil.Cil.Instruction i) : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after -= 2;
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
        public virtual bool IsSigned { get; set; }

        public override Inst Convert(Converter converter, State state)
        {
            Value v2 = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v2);

            Value v1 = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v1);

            // TODO Undoubtably, this will be much more complicated than my initial stab.
            Type t1 = v1.T;
            Type t2 = v2.T;
            ValueRef cmp = default(ValueRef);
            // Deal with various combinations of types.
            if (t1.isIntegerTy() && t2.isIntegerTy())
            {
                IntPredicate op;
                if (IsSigned) op = _int_pred[(int)Predicate];
                else op = _uint_pred[(int)Predicate];

                cmp = LLVM.BuildICmp(Builder, op, v1.V, v2.V, "");

                GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge1 = Block._Successors[0];
                GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge2 = Block._Successors[1];
                int succ1 = edge1.To;
                int succ2 = edge2.To;
                var s1 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
                var s2 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ2)];
                LLVM.BuildCondBr(Builder, cmp, s1.BasicBlock, s2.BasicBlock);
                return Next;
            }
            throw new Exception("Unhandled compare and branch.");
        }
    }


    public class ConvertInst : Inst
    {
        protected Type _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        Value convert_full(Value src)
        {
            TypeRef stype = LLVM.TypeOf(src.V);
            TypeRef dtype = _dst.IntermediateType;

            if (stype != dtype)
            {
                bool ext = false;

                /* Extend */
                if (dtype == LLVM.Int64Type()
                    && (stype == LLVM.Int32Type() || stype == LLVM.Int16Type() || stype == LLVM.Int8Type()))
                    ext = true;
                else if (dtype == LLVM.Int32Type()
                    && (stype == LLVM.Int16Type() || stype == LLVM.Int8Type()))
                    ext = true;
                else if (dtype == LLVM.Int16Type()
                    && (stype == LLVM.Int8Type()))
                    ext = true;

                if (ext)
                    return new Value(
                        _dst.is_unsigned
                        ? LLVM.BuildZExt(Builder, src.V, dtype, "")
                        : LLVM.BuildSExt(Builder, src.V, dtype, ""));

                if (dtype == LLVM.DoubleType() && stype == LLVM.FloatType())
                    return new Value(LLVM.BuildFPExt(Builder, src.V, dtype, ""));

                /* Trunc */
                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.Int32Type() || dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type()))
                    return new Value(LLVM.BuildTrunc(Builder, src.V, dtype, ""));
                if (stype == LLVM.Int32Type()
                    && (dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type()))
                    return new Value(LLVM.BuildTrunc(Builder, src.V, dtype, ""));
                if (stype == LLVM.Int16Type()
                    && dtype == LLVM.Int8Type())
                    return new Value(LLVM.BuildTrunc(Builder, src.V, dtype, ""));
                if (stype == LLVM.DoubleType()
                    && dtype == LLVM.FloatType())
                    return new Value(LLVM.BuildFPTrunc(Builder, src.V, dtype, ""));

                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.FloatType()))
                    return new Value(LLVM.BuildSIToFP(Builder, src.V, dtype, ""));
                if (stype == LLVM.Int32Type()
                    && (dtype == LLVM.FloatType()))
                    return new Value(LLVM.BuildSIToFP(Builder, src.V, dtype, ""));
                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.DoubleType()))
                    return new Value(LLVM.BuildSIToFP(Builder, src.V, dtype, ""));
                if (stype == LLVM.Int32Type()
                    && (dtype == LLVM.DoubleType()))
                    return new Value(LLVM.BuildSIToFP(Builder, src.V, dtype, ""));

                //if (LLVM.GetTypeKind(stype) == LLVM.PointerTypeKind && LLVM.GetTypeKind(dtype) == LLVMPointerTypeKind)
                //    return LLVM.BuildBitCast(Builder, src, dtype, "");
                //if (LLVM.GetTypeKind(dtype) == LLVM.PointerTypeKind)
                //    return LLVM.BuildIntToPtr(Builder, src, dtype, "");
                //if (LLVM.GetTypeKind(stype) == LLVM.PointerTypeKind)
                //    return LLVM.BuildPtrToInt(Builder, src, dtype, "");

                //if (mono_arch_is_soft_float())
                //{
                //    if (stype == LLVM.Int32Type() && dtype == LLVM.FloatType())
                //        return LLVM.BuildBitCast(Builder, src, dtype, "");
                //    if (stype == LLVM.Int32Type() && dtype == LLVM.DoubleType())
                //        return LLVM.BuildBitCast(Builder, LLVM.BuildZExt(Builder, src, LLVM.Int64Type(), ""), dtype, "");
                //}

                //if (LLVM.GetTypeKind(stype) == LLVM.VectorTypeKind && LLVM.GetTypeKind(dtype) == LLVMVectorTypeKind)
                //    return LLVM.BuildBitCast(Builder, src, dtype, "");

                //                LLVM.DumpValue(src);
                //                LLVM.DumpValue(LLVM.ConstNull(dtype.T));
                return new Value(default(ValueRef));
            }
            else
            {
                return src;
            }
        }

        public ConvertInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // No change in stack level.
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value s = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(s.ToString());

            Value d = convert_full(s);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(d.ToString());

            state._stack.Push(d);
            return Next;
        }
    }

    public class ConvertOvfInst : ConvertInst
    {
        public ConvertOvfInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _check_overflow = true;
        }
    }

    public class ConvertOvfUnsInst : ConvertInst
    {
        public ConvertOvfUnsInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _check_overflow = true;
            _from_unsigned = true;
        }
    }

    public class ConvertUnsInst : ConvertInst
    {
        public ConvertUnsInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _from_unsigned = true;
        }
    }

    public class ConvertLoadElement : Inst
    {
        protected Type _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertLoadElement(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            Value a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var load = a.V;
            load = LLVM.BuildLoad(Builder, load, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            // Load array base.
            ValueRef extract_value = LLVM.BuildExtractValue(Builder, load, 0, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(extract_value));

            // Now add in index to pointer.
            ValueRef[] indexes = new ValueRef[1];
            indexes[0] = i.V;
            ValueRef gep = LLVM.BuildInBoundsGEP(Builder, extract_value, indexes, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(gep));

            load = LLVM.BuildLoad(Builder, gep, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            if (_dst != null &&_dst.IntermediateType != LLVM.TypeOf(load))
            {
                load = LLVM.BuildIntCast(Builder, load, _dst.IntermediateType, "");
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new Value(load));
            }
            else if (_dst == null)
            {
                var t_v = LLVM.TypeOf(load);
                TypeRef t_to;
                // Type information for instruction obtuse. 
                // Use LLVM type and set stack type.
                if (t_v == LLVM.Int8Type() || t_v == LLVM.Int16Type())
                {
                    load = LLVM.BuildIntCast(Builder, load, LLVM.Int32Type(), "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(load));
                }
                else
                    t_to = t_v;
                //var op = this.Operand;
                //var tt = op.GetType();
            }

            state._stack.Push(new Value(load));
            return Next;
        }
    }

    public class ConvertStoreElement : Inst
    {
        protected Type _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertStoreElement(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after = level_after - 3;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());

            Value i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            Value a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var load = a.V;
            load = LLVM.BuildLoad(Builder, load, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            ValueRef extract_value = LLVM.BuildExtractValue(Builder, load, 0, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(extract_value));

            // Now add in index to pointer.
            ValueRef[] indexes = new ValueRef[1];
            indexes[0] = i.V;
            ValueRef gep = LLVM.BuildInBoundsGEP(Builder, extract_value, indexes, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(gep));

            var value = v.V;
            if (_dst != null && _dst.VerificationType.ToTypeRef() != v.T.IntermediateType)
            {
                value = LLVM.BuildIntCast(Builder, value, _dst.VerificationType.ToTypeRef(), "");
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new Value(value));
            }
            else if (_dst == null)
            {
                var t_v = LLVM.TypeOf(value);
                var t_d = LLVM.TypeOf(gep);
                var t_e = LLVM.GetElementType(t_d);
                if (t_v != t_e)
                {
                    value = LLVM.BuildIntCast(Builder, value, t_e, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(value));
                }
            }

            // Store.
            var store = LLVM.BuildStore(Builder, value, gep);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(store));

            return Next;
        }
    }


    public class ConvertLoadElementA : Inst
    {
        public ConvertLoadElementA(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            Value a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var load = a.V;
            load = LLVM.BuildLoad(Builder, load, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            // Load array base.
            ValueRef extract_value = LLVM.BuildExtractValue(Builder, load, 0, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(extract_value));

            // Now add in index to pointer.
            ValueRef[] indexes = new ValueRef[1];
            indexes[0] = i.V;
            ValueRef gep = LLVM.BuildInBoundsGEP(Builder, extract_value, indexes, "");
            var result = new Value(gep);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(result);

            state._stack.Push(result);
            return Next;
        }
    }

    public class ConvertLoadField : Inst
    {
        public ConvertLoadField(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // Stack level remains unchanged through instruction.
        }

        public override Inst Convert(Converter converter, State state)
        {
            {
                Value v = state._stack.Pop();
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(v);

                TypeRef tr = LLVM.TypeOf(v.V);
                bool isPtr = v.T.isPointerTy();
                bool isArr = v.T.isArrayTy();
                bool isSt = v.T.isStructTy();

                if (isPtr)
                {
                    uint offset = 0;
                    object yy = this.Instruction.Operand;
                    FieldReference field = yy as Mono.Cecil.FieldReference;
                    if (yy == null) throw new Exception("Cannot convert.");

                    // The instruction may be generic, even if the method
                    // is an instance. Convert field to generic instance type reference
                    // if it is a generic, in the context of this basic block.

                    TypeReference declaring_type_tr = field.DeclaringType;
                    TypeDefinition declaring_type = declaring_type_tr.Resolve();

                    if (!declaring_type.IsGenericInstance && declaring_type.HasGenericParameters)
                    {
                        // This is a red flag. We need to come up with a generic instance for type.
                        declaring_type_tr = this.Block.ExpectedCalleeSignature.DeclaringType;
                    }
                    
                    // need to take into account padding fields. Unfortunately,
                    // LLVM does not name elements in a struct/class. So, we must
                    // compute padding and adjust.
                    int size = 0;
                    foreach (var f in declaring_type.Fields)
                    {
                        var attr = f.Attributes;
                        if ((attr & FieldAttributes.Static) != 0)
                            continue;

                        int field_size;
                        int alignment;
                        var array_or_class = (f.FieldType.IsArray || !f.FieldType.IsValueType);
                        if (array_or_class)
                        {
                            field_size = Buffers.SizeOf(typeof(IntPtr));
                            alignment = Buffers.Alignment(typeof(IntPtr));
                        }
                        else
                        {
                            var ft = f.FieldType.ToSystemType();
                            field_size = Buffers.SizeOf(ft);
                            alignment = Buffers.Alignment(ft);
                        }
                        int padding = Buffers.Padding(size, alignment);
                        size = size + padding + field_size;
                        if (padding != 0)
                        {
                            // Add in bytes to effect padding.
                            for (int j = 0; j < padding; ++j)
                                offset++;
                        }

                        if (f.Name == field.Name)
                            break;
                        offset++;
                    }

                    var tt = LLVM.TypeOf(v.V);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(LLVM.PrintTypeToString(tt));

                    var addr = LLVM.BuildStructGEP(Builder, v.V, offset, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(addr));

                    var load = LLVM.BuildLoad(Builder, addr, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(load));


                    //var you = Converter.FromGenericParameterToTypeReference(field.FieldType,
                    //    declaring_type_tr as GenericInstanceType);
                    //// Add extra load for pointer types like objects and arrays.
                    //var array_or_classyou  = (you.IsArray || !you.IsValueType);
                    //if (array_or_classyou)
                    //{
                    //    load = LLVM.BuildLoad(Builder, load, "");
                    //    if (Campy.Utils.Options.IsOn("jit_trace"))
                    //        System.Console.WriteLine(new Value(load));
                    //}

                    bool xInt = LLVM.GetTypeKind(tt) == TypeKind.IntegerTypeKind;
                    bool xP = LLVM.GetTypeKind(tt) == TypeKind.PointerTypeKind;
                    bool xA = LLVM.GetTypeKind(tt) == TypeKind.ArrayTypeKind;

                    // If load result is a pointer, then cast it to proper type.
                    // This is because I had to avoid recursive data types in classes
                    // as LLVM cannot handle these at all. So, all pointer types
                    // were defined as void* in the LLVM field.

                    var load_value = new Value(load);
                    bool isPtrLoad = load_value.T.isPointerTy();
                    //if (isPtrLoad)
                    //{
                    //    var mono_field_type = field.FieldType;
                    //    TypeRef type = Converter.ToTypeRef(
                    //        mono_field_type,
                    //        Block.OpsFromOriginal);
                    //    load = LLVM.BuildBitCast(Builder,
                    //        load, type, "");
                    //    if (Campy.Utils.Options.IsOn("jit_trace"))
                    //        System.Console.WriteLine(new Value(load));
                    //}

                    state._stack.Push(new Value(load));
                }
                else
                {
                    uint offset = 0;
                    var yy = this.Instruction.Operand;
                    var field = yy as Mono.Cecil.FieldReference;
                    if (yy == null) throw new Exception("Cannot convert.");
                    var declaring_type_tr = field.DeclaringType;
                    var declaring_type = declaring_type_tr.Resolve();

                    // need to take into account padding fields. Unfortunately,
                    // LLVM does not name elements in a struct/class. So, we must
                    // compute padding and adjust.
                    int size = 0;
                    foreach (var f in declaring_type.Fields)
                    {
                        var attr = f.Attributes;
                        if ((attr & FieldAttributes.Static) != 0)
                            continue;

                        int field_size;
                        int alignment;
                        var ft = f.FieldType.ToSystemType();
                        var array_or_class = (f.FieldType.IsArray || !f.FieldType.IsValueType);
                        if (array_or_class)
                        {
                            field_size = Buffers.SizeOf(typeof(IntPtr));
                            alignment = Buffers.Alignment(typeof(IntPtr));
                        }
                        else
                        {
                            field_size = Buffers.SizeOf(ft);
                            alignment = Buffers.Alignment(ft);
                        }
                        int padding = Buffers.Padding(size, alignment);
                        size = size + padding + field_size;
                        if (padding != 0)
                        {
                            // Add in bytes to effect padding.
                            for (int j = 0; j < padding; ++j)
                                offset++;
                        }

                        if (f.Name == field.Name)
                            break;
                        offset++;
                    }

                    var tt = LLVM.TypeOf(v.V);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(LLVM.PrintTypeToString(tt));

                    var addr = LLVM.BuildExtractValue(Builder, v.V, offset, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(addr));

                    var load = LLVM.BuildLoad(Builder, addr, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(load));

                    bool xInt = LLVM.GetTypeKind(tt) == TypeKind.IntegerTypeKind;
                    bool xP = LLVM.GetTypeKind(tt) == TypeKind.PointerTypeKind;
                    bool xA = LLVM.GetTypeKind(tt) == TypeKind.ArrayTypeKind;

                    // If load result is a pointer, then cast it to proper type.
                    // This is because I had to avoid recursive data types in classes
                    // as LLVM cannot handle these at all. So, all pointer types
                    // were defined as void* in the LLVM field.

                    var load_value = new Value(load);
                    bool isPtrLoad = load_value.T.isPointerTy();
                    if (isPtrLoad)
                    {
                        var mono_field_type = field.FieldType;
                        TypeRef type = mono_field_type.ToTypeRef(Block.OpsFromOriginal);
                        load = LLVM.BuildBitCast(Builder,
                            load, type, "");
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new Value(load));
                    }

                    state._stack.Push(new Value(load));
                }

                return Next;
            }
        }
    }

    public class ConvertStoreField : Inst
    {
        public ConvertStoreField(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after = level_after - 2;
        }

        public override Inst Convert(Converter converter, State state)
        {
            {
                Value v = state._stack.Pop();
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(v);

                Value o = state._stack.Pop();
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(o);

                TypeRef tr = LLVM.TypeOf(o.V);
                bool isPtr = o.T.isPointerTy();
                bool isArr = o.T.isArrayTy();
                bool isSt = o.T.isStructTy();
                bool is_ptr = false;
                if (isPtr)
                {
                    uint offset = 0;
                    var yy = this.Instruction.Operand;
                    var field = yy as Mono.Cecil.FieldReference;
                    if (yy == null) throw new Exception("Cannot convert.");
                    var declaring_type_tr = field.DeclaringType;
                    var declaring_type = declaring_type_tr.Resolve();

                    // need to take into account padding fields. Unfortunately,
                    // LLVM does not name elements in a struct/class. So, we must
                    // compute padding and adjust.
                    int size = 0;
                    foreach (var f in declaring_type.Fields)
                    {
                        var attr = f.Attributes;
                        if ((attr & FieldAttributes.Static) != 0)
                            continue;

                        int field_size;
                        int alignment;
                        var array_or_class = (f.FieldType.IsArray || !f.FieldType.IsValueType);
                        if (array_or_class)
                        {
                            field_size = Buffers.SizeOf(typeof(IntPtr));
                            alignment = Buffers.Alignment(typeof(IntPtr));
                        }
                        else
                        {
                            var ft = f.FieldType.ToSystemType();
                            field_size = Buffers.SizeOf(ft);
                            alignment = Buffers.Alignment(ft);
                        }
                        int padding = Buffers.Padding(size, alignment);
                        size = size + padding + field_size;
                        if (padding != 0)
                        {
                            // Add in bytes to effect padding.
                            for (int j = 0; j < padding; ++j)
                                offset++;
                        }

                        if (f.Name == field.Name)
                        {
                            is_ptr = f.FieldType.IsArray || f.FieldType.IsPointer;
                            break;
                        }
                        offset++;
                    }

                    var addr = LLVM.BuildStructGEP(Builder, o.V, offset, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(addr));

                    // If value is a pointer, then cast it to void*.
                    // This is because I had to avoid recursive data types in classes
                    // as LLVM cannot handle these at all. So, all pointer types
                    // were defined as void* in the LLVM field.
                    var load = LLVM.BuildLoad(Builder, addr, "");

                    var load_value = new Value(load);
                    bool isPtrLoad = load_value.T.isPointerTy();
                    if (isPtrLoad)
                    {
                        var mono_field_type = field.FieldType;
                        TypeRef type = mono_field_type.ToTypeRef(Block.OpsFromOriginal);
                        addr = LLVM.BuildBitCast(Builder,
                            addr, type, "");
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new Value(addr));
                    }

                    var store = LLVM.BuildStore(Builder, v.V, addr);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new Value(store));
                }
                else
                {
                    throw new Exception("Value type ldfld not implemented!");
                }

                return Next;
            }
        }
    }

    public class ConvertLoadIndirect : Inst
    {
        protected Type _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertLoadIndirect(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // No change in depth of stack.
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("ConvertLoadIndirect into function " + v.ToString());
            TypeRef tr = LLVM.TypeOf(v.V);
            bool isPtr = v.T.isPointerTy();
            bool isArr = v.T.isArrayTy();
            bool isSt = v.T.isStructTy();
            TypeKind kind = LLVM.GetTypeKind(tr);
            bool isPtra = kind == TypeKind.PointerTypeKind;
            bool isArra = kind == TypeKind.ArrayTypeKind;
            bool isSta = kind == TypeKind.StructTypeKind;
            ValueRef[] indexes = new ValueRef[0];
            ValueRef load = v.V;
            ValueRef ll = LLVM.BuildInBoundsGEP(Builder, load, indexes, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(ll).ToString());
            var zz = LLVM.BuildLoad(Builder, ll, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("zz = " + new Value(zz).ToString());
            state._stack.Push(new Value(zz));
            return Next;
        }
    }

    public class ConvertStoreIndirect : Inst
    {
        public ConvertStoreIndirect(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after -= 2;
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value v = state._stack.Pop();
            Value a = state._stack.Pop();

            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("ConvertStoreIndirect into function v " + v.ToString());
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("ConvertLoadIndirect into function a " + a.ToString());

            TypeRef tr = LLVM.TypeOf(a.V);
            bool isPtr = a.T.isPointerTy();

            ValueRef[] indexes = new ValueRef[0];
            ValueRef load = a.V;
            ValueRef ll = LLVM.BuildInBoundsGEP(Builder, load, indexes, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(ll).ToString());
            var zz = LLVM.BuildStore(Builder, v.V, ll);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("Store = " + new Value(zz).ToString());

            return Next;
        }
    }



    public class i_add : BinaryOpInst
    {
        public i_add(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_add_ovf : BinaryOpInst
    {
        public i_add_ovf(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_add_ovf_un : BinaryOpInst
    {
        public i_add_ovf_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_and : BinaryOpInst
    {
        public i_and(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_arglist : Inst
    {
        public i_arglist(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_beq : CompareAndBranchInst
    {
        public i_beq(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.eq;
            IsSigned = true;
        }
    }

    public class i_beq_s : CompareAndBranchInst
    {
        public i_beq_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.eq;
            IsSigned = true;
        }
    }

    public class i_bge : CompareAndBranchInst
    {
        public i_bge(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = true;
        }
    }

    public class i_bge_un : CompareAndBranchInst
    {
        public i_bge_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = false;
        }
    }

    public class i_bge_un_s : CompareAndBranchInst
    {
        public i_bge_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = false;
        }
    }

    public class i_bge_s : CompareAndBranchInst
    {
        public i_bge_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = true;
        }
    }

    public class i_bgt : CompareAndBranchInst
    {
        public i_bgt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = true;
        }
    }

    public class i_bgt_s : CompareAndBranchInst
    {
        public i_bgt_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = true;
        }
    }

    public class i_bgt_un : CompareAndBranchInst
    {
        public i_bgt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = false;
        }
    }

    public class i_bgt_un_s : CompareAndBranchInst
    {
        public i_bgt_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = false;
        }
    }

    public class i_ble : CompareAndBranchInst
    {
        public i_ble(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
            IsSigned = true;
        }
    }

    public class i_ble_s : CompareAndBranchInst
    {
        public i_ble_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
        }
    }

    public class i_ble_un : CompareAndBranchInst
    {
        public i_ble_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
            IsSigned = false;
        }
    }

    public class i_ble_un_s : CompareAndBranchInst
    {
        public i_ble_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
            IsSigned = false;
        }
    }

    public class i_blt : CompareAndBranchInst
    {
        public i_blt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = true;
        }
    }

    public class i_blt_s : CompareAndBranchInst
    {
        public i_blt_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = true;
        }
    }

    public class i_blt_un : CompareAndBranchInst
    {
        public i_blt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = false;
        }
    }

    public class i_blt_un_s : CompareAndBranchInst
    {
        public i_blt_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = false;
        }
    }

    public class i_bne_un : CompareAndBranchInst
    {
        public i_bne_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ne;
            IsSigned = false;
        }
    }

    public class i_bne_un_s : CompareAndBranchInst
    {
        public i_bne_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ne;
            IsSigned = false;
        }
    }

    public class i_box : Inst
    {
        public i_box(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_br : Inst
    {
        public i_br(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // No change.
        }

        public override Inst Convert(Converter converter, State state)
        {
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge = Block._Successors[0];
            int succ = edge.To;
            var s = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ)];
            var br = LLVM.BuildBr(Builder, s.BasicBlock);
            return Next;
        }
    }

    public class i_br_s : Inst
    {
        public i_br_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // No change.
        }

        public override Inst Convert(Converter converter, State state)
        {
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge = Block._Successors[0];
            int succ = edge.To;
            var s = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ)];
            var br = LLVM.BuildBr(Builder, s.BasicBlock);
            return Next;
        }
    }

    public class i_brfalse : Inst
    {
        public i_brfalse(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var v = state._stack.Pop();
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge1 = Block._Successors[0];
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge2 = Block._Successors[1];
            int succ1 = edge1.To;
            int succ2 = edge2.To;
            var s1 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
            var s2 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ2)];
            LLVM.BuildCondBr(Builder, v.V, s1.BasicBlock, s2.BasicBlock);
            return Next;
        }
    }

    public class i_break : Inst
    {
        public i_break(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_brfalse_s : Inst
    {
        public i_brfalse_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var v = state._stack.Pop();
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge1 = Block._Successors[0];
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge2 = Block._Successors[1];
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
        public i_brtrue(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var v = state._stack.Pop();
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge1 = Block._Successors[0];
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge2 = Block._Successors[1];
            int succ1 = edge1.To;
            int succ2 = edge2.To;
            var s1 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
            var s2 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ2)];
            // We need to compare the value popped with 0/1.
            var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
            var v3 = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, v.V, v2, "");
            LLVM.BuildCondBr(Builder, v3, s1.BasicBlock, s2.BasicBlock);
            return Next;
        }
    }

    public class i_brtrue_s : Inst
    {
        public i_brtrue_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var v = state._stack.Pop();
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge1 = Block._Successors[0];
            GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge2 = Block._Successors[1];
            int succ1 = edge1.To;
            int succ2 = edge2.To;
            var s1 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ1)];
            var s2 = Block._Graph.VertexSpace[Block._Graph.NameSpace.BijectFromBasetype(succ2)];
            // We need to compare the value popped with 0/1.
            var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
            var v3 = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, v.V, v2, "");
            LLVM.BuildCondBr(Builder, v3, s1.BasicBlock, s2.BasicBlock);
            return Next;
        }
    }

    public class i_call : Inst
    {
        public i_call(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // Successor is fallthrough.
            int args = 0;
            int ret = 0;
            object method = this.Operand;
            if (method as Mono.Cecil.MethodReference != null)
            {
                Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
                {
                    if (mr.HasThis)
                        args++;
                    args += mr.Parameters.Count;
                    if (mr.MethodReturnType != null)
                    {
                        Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                        Mono.Cecil.TypeReference tr = rt.ReturnType;
                        // Get type, may contain modifiers.
                        if (tr.FullName.Contains(' '))
                        {
                            String[] sp = tr.FullName.Split(' ');
                            if (!sp[0].Equals("System.Void"))
                                ret++;
                        }
                        else
                        {
                            if (!tr.FullName.Equals("System.Void"))
                                ret++;
                        }
                    }
                }
            }
            level_after = level_after + ret - args;
        }

        public override Inst Convert(Converter converter, State state)
        {
            // Get function.
            var j = this;

            // Successor is fallthrough.
            int nargs = 0;
            int ret = 0;
            object method = j.Operand;
            if (method as Mono.Cecil.MethodReference != null)
            {
                Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
                CFG graph = (CFG)this.Block._Graph;
                if (!graph.MethodAvoid(mr.FullName))
                {
                    if (mr.HasThis)
                        nargs++;
                    nargs += mr.Parameters.Count;
                    if (mr.MethodReturnType != null)
                    {
                        Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                        Mono.Cecil.TypeReference tr = rt.ReturnType;
                        // Get type, may contain modifiers.
                        if (tr.FullName.Contains(' '))
                        {
                            String[] sp = tr.FullName.Split(' ');
                            if (!sp[0].Equals("System.Void"))
                                ret++;
                        }
                        else
                        {
                            if (!tr.FullName.Equals("System.Void"))
                                ret++;
                        }
                    }
                    var name = Converter.MethodName(mr);
                    // Find bb entry.
                    CFG.Vertex the_entry = this.Block._Graph.VertexNodes.Where(node
                        =>
                    {
                        GraphLinkedList<int, CFG.Vertex, CFG.Edge> g = j.Block._Graph;
                        int k = g.NameSpace.BijectFromBasetype(node.Name);
                        CFG.Vertex v = g.VertexSpace[k];
                        Converter c = converter;
                        if (v.IsEntry && Converter.MethodName(v.ExpectedCalleeSignature) == name && c.IsFullyInstantiatedNode(v))
                            return true;
                        else return false;
                    }).ToList().FirstOrDefault();

                    if (the_entry == null)
                    {
                        if (!Options.IsOn("continue_with_no_resolve"))
                            throw new Exception("unimplemented.");
                        else
                            return Next;
                    }
                    else
                    {
                        BuilderRef bu = this.Builder;
                        ValueRef fv = the_entry.MethodValueRef;
                        var t_fun = LLVM.TypeOf(fv);
                        var t_fun_con = LLVM.GetTypeContext(t_fun);
                        var context = LLVM.GetModuleContext(Converter.global_llvm_module);
                        if (t_fun_con != context) throw new Exception("not equal");
                        //LLVM.VerifyFunction(fv, VerifierFailureAction.PrintMessageAction);
                        ValueRef[] args = new ValueRef[nargs];
                        for (int k = nargs - 1; k >= 0; --k)
                            args[k] = state._stack.Pop().V;
                        if (ret > 0)
                        {
                            var call = LLVM.BuildCall(Builder, fv, args, name);
                            state._stack.Push(new Value(call));
                            if (Campy.Utils.Options.IsOn("jit_trace"))
                                System.Console.WriteLine(call.ToString());

                        }
                        else
                        {
                            var call = LLVM.BuildCall(Builder, fv, args, "");
                            if (Campy.Utils.Options.IsOn("jit_trace"))
                                System.Console.WriteLine(call.ToString());
                        }
                    }
                }
                else
                {
                    for (int k = nargs - 1; k >= 0; --k)
                        state._stack.Pop();

                    // Generate code to crash.


                    if (ret > 0)
                    {
                        state._stack.Push(new Value(LLVM.ConstInt(LLVM.Int32Type(), 0, false)));
                    }
                    else
                    {
                    }
                }
            }
            return Next;
        }
    }

    public class i_calli : Inst
    {
        public i_calli(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
        public override void ComputeStackLevel(ref int level_after)
        {
            // Successor is fallthrough.
            int args = 0;
            int ret = 0;
            args++; // The function is on the stack.
            object method = this.Operand;
            if (method as Mono.Cecil.CallSite != null)
            {
                Mono.Cecil.CallSite mr = method as Mono.Cecil.CallSite;
                if (mr.HasThis)
                    args++;
                args += mr.Parameters.Count;
                if (mr.MethodReturnType != null)
                {
                    Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                    Mono.Cecil.TypeReference tr = rt.ReturnType;
            // Get type, may contain modifiers.
                    if (tr.FullName.Contains(' '))
                    {
                        String[] sp = tr.FullName.Split(' ');
                        if (!sp[0].Equals("System.Void"))
                            ret++;
                    }
                    else
                    {
                        if (!tr.FullName.Equals("System.Void"))
                            ret++;
                    }
                }
            }
            level_after = level_after + ret - args;
        }
    }

    public class i_callvirt : Inst
    {
        public i_callvirt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // Successor is fallthrough.
            int args = 0;
            int ret = 0;
            object method = this.Operand;
            if (method as Mono.Cecil.MethodReference != null)
            {
                Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
                if (mr.HasThis)
                    args++;
                args += mr.Parameters.Count;
                if (mr.MethodReturnType != null)
                {
                    Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                    Mono.Cecil.TypeReference tr = rt.ReturnType;
                    // Get type, may contain modifiers.
                    if (tr.FullName.Contains(' '))
                    {
                        String[] sp = tr.FullName.Split(' ');
                        if (!sp[0].Equals("System.Void"))
                            ret++;
                    }
                    else
                    {
                        if (!tr.FullName.Equals("System.Void"))
                            ret++;
                    }
                }
            }
            level_after = level_after + ret - args;
        }
        public override Inst Convert(Converter converter, State state)
        {
            // Get function.
            var j = this;

            // Successor is fallthrough.
            int nargs = 0;
            int ret = 0;
            object method = j.Operand;
            if (method as Mono.Cecil.MethodReference != null)
            {
                Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
                if (mr.HasThis)
                    nargs++;
                nargs += mr.Parameters.Count;
                if (mr.MethodReturnType != null)
                {
                    Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                    Mono.Cecil.TypeReference tr = rt.ReturnType;
                    // Get type, may contain modifiers.
                    if (tr.FullName.Contains(' '))
                    {
                        String[] sp = tr.FullName.Split(' ');
                        if (!sp[0].Equals("System.Void"))
                            ret++;
                    }
                    else
                    {
                        if (!tr.FullName.Equals("System.Void"))
                            ret++;
                    }
                }
                var name = Converter.MethodName(mr);
                // Find bb entry.
                CFG.Vertex the_entry = this.Block._Graph.VertexNodes.Where(node
                    =>
                {
                    GraphLinkedList<int, CFG.Vertex, CFG.Edge> g = j.Block._Graph;
                    int k = g.NameSpace.BijectFromBasetype(node.Name);
                    CFG.Vertex v = g.VertexSpace[k];
                    Converter c = converter;
                    if (v.IsEntry && Converter.MethodName(v.ExpectedCalleeSignature) == name && c.IsFullyInstantiatedNode(v))
                        return true;
                    else return false;
                }).ToList().FirstOrDefault();

                if (the_entry != default(CFG.Vertex))
                {
                    BuilderRef bu = this.Builder;
                    ValueRef fv = the_entry.MethodValueRef;
                    var t_fun = LLVM.TypeOf(fv);
                    var t_fun_con = LLVM.GetTypeContext(t_fun);
                    var context = LLVM.GetModuleContext(Converter.global_llvm_module);
                    if (t_fun_con != context) throw new Exception("not equal");
                    //LLVM.VerifyFunction(fv, VerifierFailureAction.PrintMessageAction);
                    ValueRef[] args = new ValueRef[nargs];
                    for (int k = nargs - 1; k >= 0; --k)
                        args[k] = state._stack.Pop().V;
                    if (ret > 0)
                    {
                        var call = LLVM.BuildCall(Builder, fv, args, name);
                        state._stack.Push(new Value(call));
                    }
                    else
                    {
                        var call = LLVM.BuildCall(Builder, fv, args, "");
                    }
                }
            }
            return Next;
        }
    }

    public class i_castclass : Inst
    {
        public i_castclass(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ceq : CompareInst
    {
        public i_ceq(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.eq;
            IsSigned = true;
        }
    }

    public class i_cgt : CompareInst
    {
        public i_cgt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = true;
        }
    }

    public class i_cgt_un : CompareInst
    {
        public i_cgt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = false;
        }
    }

    public class i_ckfinite : Inst
    {
        public i_ckfinite(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_clt : CompareInst
    {
        public i_clt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = true;
        }
    }

    public class i_clt_un : CompareInst
    {
        public i_clt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = false;
        }
    }

    public class i_constrained : Inst
    {
        public i_constrained(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_conv_i1 : ConvertInst
    {
        public i_conv_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(sbyte));
        }
    }

    public class i_conv_i2 : ConvertInst
    {
        public i_conv_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(short));
        }
    }

    public class i_conv_i4 : ConvertInst
    {
        public i_conv_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_conv_i8 : ConvertInst
    {
        public i_conv_i8(Mono.Cecil.Cil.Instruction i)
                : base(i)
        {
            _dst = new Type(typeof(long));
        }
    }

    public class i_conv_i : ConvertInst
    {
        public i_conv_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_conv_ovf_i1 : ConvertOvfInst
    {
        public i_conv_ovf_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(sbyte));
        }
    }

    public class i_conv_ovf_i1_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_i1_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(sbyte));
        }
    }

    public class i_conv_ovf_i2 : ConvertOvfInst
    {
        public i_conv_ovf_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(short));
        }
    }

    public class i_conv_ovf_i2_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_i2_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(short));
        }
    }

    public class i_conv_ovf_i4 : ConvertOvfInst
    {
        public i_conv_ovf_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_conv_ovf_i4_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_i4_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_conv_ovf_i8 : ConvertOvfInst
    {
        public i_conv_ovf_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(long));
        }
    }

    public class i_conv_ovf_i8_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_i8_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(long));
        }
    }

    public class i_conv_ovf_i : ConvertOvfInst
    {
        public i_conv_ovf_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_conv_ovf_i_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_i_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_conv_ovf_u1 : ConvertOvfInst
    {
        public i_conv_ovf_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(byte));
        }
    }

    public class i_conv_ovf_u1_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_u1_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(byte));
        }
    }

    public class i_conv_ovf_u2 : ConvertOvfInst
    {
        public i_conv_ovf_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ushort));
        }
    }

    public class i_conv_ovf_u2_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_u2_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ushort));
        }
    }

    public class i_conv_ovf_u4 : ConvertOvfInst
    {
        public i_conv_ovf_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_conv_ovf_u4_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_u4_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_conv_ovf_u8 : ConvertOvfInst
    {
        public i_conv_ovf_u8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ulong));
        }
    }

    public class i_conv_ovf_u8_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_u8_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ulong));
        }
    }

    public class i_conv_ovf_u : ConvertOvfInst
    {
        public i_conv_ovf_u(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_conv_ovf_u_un : ConvertOvfUnsInst
    {
        public i_conv_ovf_u_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_conv_r4 : ConvertInst
    {
        public i_conv_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(float));
        }
    }

    public class i_conv_r8 : ConvertInst
    {
        public i_conv_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(double));
        }
    }

    public class i_conv_r_un : ConvertUnsInst
    {
        public i_conv_r_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(float));
        }
    }

    public class i_conv_u1 : ConvertInst
    {
        public i_conv_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(byte));
        }
    }

    public class i_conv_u2 : ConvertInst
    {
        public i_conv_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ushort));
        }
    }

    public class i_conv_u4 : ConvertInst
    {
        public i_conv_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_conv_u8 : ConvertInst
    {
        public i_conv_u8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ulong));
        }
    }

    public class i_conv_u : ConvertInst
    {
        public i_conv_u(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_cpblk : Inst
    {
        public i_cpblk(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_cpobj : Inst
    {
        public i_cpobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_div : BinaryOpInst
    {
        public i_div(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_div_un : BinaryOpInst
    {
        public i_div_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_dup : Inst
    {
        public i_dup(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            var rhs = state._stack.Pop();
            state._stack.Push(rhs);
            state._stack.Push(rhs);
            return Next;
        }

    }

    public class i_endfilter : Inst
    {
        public i_endfilter(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_endfinally : Inst
    {
        public i_endfinally(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_initblk : Inst
    {
        public i_initblk(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after -= 3;
    }
    }

    public class i_initobj : Inst
    {
        public i_initobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_isinst : Inst
    {
        public i_isinst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_jmp : Inst
    {
        public i_jmp(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldarg : LoadArgInst
    {
        public i_ldarg(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }
    }

    public class i_ldarg_0 : LoadArgInst
    {
        public i_ldarg_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 0;
        }
    }

    public class i_ldarg_1 : LoadArgInst
    {
        public i_ldarg_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 1;
        }
    }

    public class i_ldarg_2 : LoadArgInst
    {
        public i_ldarg_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 2;
        }
    }

    public class i_ldarg_3 : LoadArgInst
    {

        public i_ldarg_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 3;
        }
    }

    public class i_ldarg_s : LoadArgInst
    {
        public i_ldarg_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }
    }

    public class i_ldarga : LoadArgInst
    {
        public i_ldarga(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldarga_s : LoadArgInst
    {
        public i_ldarga_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldc_i4 : LDCInstI4
    {
        public i_ldc_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = default(int);
            object o = i.Operand;
            if (o != null)
            {
                // Fuck C# casting in the way of just getting
                // a plain ol' int.
                for (;;)
                {
                    bool success = false;
                    try
                    {
                        byte? o3 = (byte?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        sbyte? o3 = (sbyte?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        short? o3 = (short?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        ushort? o3 = (ushort?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        int? o3 = (int?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    throw new Exception("Cannot convert ldc_i4. Unknown type of operand. F... Mono.");
                }
            }
            _arg = arg;
        }
    }

    public class i_ldc_i4_0 : LDCInstI4
    {
        public i_ldc_i4_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldc_i4_1 : LDCInstI4
    {
        public i_ldc_i4_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_2 : LDCInstI4
    {
        public i_ldc_i4_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldc_i4_3 : LDCInstI4
    {
        public i_ldc_i4_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldc_i4_4 : LDCInstI4
    {
        public i_ldc_i4_4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 4;
            _arg = arg;
        }
    }

    public class i_ldc_i4_5 : LDCInstI4
    {
        public i_ldc_i4_5(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 5;
            _arg = arg;
        }
    }

    public class i_ldc_i4_6 : LDCInstI4
    {
        public i_ldc_i4_6(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 6;
            _arg = arg;
        }
    }

    public class i_ldc_i4_7 : LDCInstI4
    {
        public i_ldc_i4_7(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 7;
            _arg = arg;
        }
    }

    public class i_ldc_i4_8 : LDCInstI4
    {
        public i_ldc_i4_8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 8;
            _arg = arg;
        }
    }

    public class i_ldc_i4_m1 : LDCInstI4
    {
        public i_ldc_i4_m1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = -1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_s : LDCInstI4
    {
        public i_ldc_i4_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = default(int);
            object o = i.Operand;
            if (o != null)
            {
                // Fuck C# casting in the way of just getting
                // a plain ol' int.
                for (;;)
                {
                    bool success = false;
                    try
                    {
                        byte? o3 = (byte?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        sbyte? o3 = (sbyte?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        short? o3 = (short?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        ushort? o3 = (ushort?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        int? o3 = (int?)o;
                        arg = (int)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    throw new Exception("Cannot convert ldc_i4. Unknown type of operand. F... Mono.");
                }
            }
            _arg = arg;
        }
    }

    public class i_ldc_i8 : LDCInstI8
    {
        public i_ldc_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Int64 arg = default(Int64);
            object o = i.Operand;
            if (o != null)
            {
                // Fuck C# casting in the way of just getting
                // a plain ol' int.
                for (;;)
                {
                    bool success = false;
                    try
                    {
                        byte? o3 = (byte?)o;
                        arg = (Int64)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        sbyte? o3 = (sbyte?)o;
                        arg = (Int64)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        short? o3 = (short?)o;
                        arg = (Int64)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        ushort? o3 = (ushort?)o;
                        arg = (Int64)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        int? o3 = (int?)o;
                        arg = (Int64)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    throw new Exception("Cannot convert ldc_i4. Unknown type of operand. F... Mono.");
                }
            }
            _arg = arg;
        }
    }

    public class i_ldc_r4 : LDCInstR4
    {
        public i_ldc_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Single arg = default(Single);
            object o = i.Operand;
            if (o != null)
            {
                // Fuck C# casting in the way of just getting
                // a plain ol' int.
                for (;;)
                {
                    bool success = false;
                    try
                    {
                        byte? o3 = (byte?)o;
                        arg = (Single)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        sbyte? o3 = (sbyte?)o;
                        arg = (Single)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        short? o3 = (short?)o;
                        arg = (Single)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        ushort? o3 = (ushort?)o;
                        arg = (Single)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        int? o3 = (int?)o;
                        arg = (Single)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        Single? o3 = (Single?)o;
                        arg = (Single)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    throw new Exception("Cannot convert ldc_i4. Unknown type of operand. F... Mono.");
                }
            }
            _arg = arg;
        }
    }

    public class i_ldc_r8 : LDCInstR8
    {
        public i_ldc_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Double arg = default(Double);
            object o = i.Operand;
            if (o != null)
            {
                // Fuck C# casting in the way of just getting
                // a plain ol' int.
                for (;;)
                {
                    bool success = false;
                    try
                    {
                        byte? o3 = (byte?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        sbyte? o3 = (sbyte?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        short? o3 = (short?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        ushort? o3 = (ushort?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        int? o3 = (int?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        Single? o3 = (Single?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    try
                    {
                        Double? o3 = (Double?)o;
                        arg = (Double)o3;
                        success = true;
                    }
                    catch { }
                    if (success) break;
                    throw new Exception("Cannot convert ldc_i4. Unknown type of operand. F... Mono.");
                }
            }
            _arg = arg;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldelem_any : ConvertLoadElement
    {
        public i_ldelem_any(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldelem_i1 : ConvertLoadElement
    {
        public i_ldelem_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(sbyte));
        }
    }

    public class i_ldelem_i2 : ConvertLoadElement
    {
        public i_ldelem_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(short));
        }
    }

    public class i_ldelem_i4 : ConvertLoadElement
    {
        public i_ldelem_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_ldelem_i8 : ConvertLoadElement
    {
        public i_ldelem_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(long));
        }
    }

    public class i_ldelem_i : ConvertLoadElement
    {
        public i_ldelem_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldelem_r4 : ConvertLoadElement
    {
        public i_ldelem_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(float));
        }
    }

    public class i_ldelem_r8 : ConvertLoadElement
    {
        public i_ldelem_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(double));
        }
    }

    public class i_ldelem_ref : ConvertLoadElement
    {
        public i_ldelem_ref(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldelem_u1 : ConvertLoadElement
    {
        public i_ldelem_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(byte));
        }
    }

    public class i_ldelem_u2 : ConvertLoadElement
    {
        public i_ldelem_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(ushort));
        }
    }

    public class i_ldelem_u4 : ConvertLoadElement
    {
        public i_ldelem_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_ldelema : ConvertLoadElementA
    {
        public i_ldelema(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldfld : ConvertLoadField
    {
        public i_ldfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldflda : Inst
    {
        public i_ldflda(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldftn : Inst
    {
        public i_ldftn(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldind_i1 : ConvertLoadIndirect
    {
        public i_ldind_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(sbyte));
        }
    }

    public class i_ldind_i2 : ConvertLoadIndirect
    {
        public i_ldind_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(short));
        }
    }

    public class i_ldind_i4 : ConvertLoadIndirect
    {
        public i_ldind_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_ldind_i8 : ConvertLoadIndirect
    {
        public i_ldind_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(long));
        }
    }

    public class i_ldind_i : ConvertLoadIndirect
    {
        public i_ldind_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldind_r4 : ConvertLoadIndirect
    {
        public i_ldind_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(float));
        }
    }

    public class i_ldind_r8 : ConvertLoadIndirect
    {
        public i_ldind_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(double));
        }
    }

    public class i_ldind_ref : ConvertLoadIndirect
    {
        public i_ldind_ref(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldind_u1 : ConvertLoadIndirect
    {
        public i_ldind_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(byte));
        }
    }

    public class i_ldind_u2 : ConvertLoadIndirect
    {
        public i_ldind_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
             _dst = new Type(typeof(ushort));
        }
    }

    public class i_ldind_u4 : ConvertLoadIndirect
    {
        public i_ldind_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(uint));
        }
    }

    public class i_ldlen : Inst
    {
        public i_ldlen(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        // For array implementation, see https://www.codeproject.com/Articles/3467/Arrays-UNDOCUMENTED
        public override void ComputeStackLevel(ref int level_after)
        {
            // No effect change in stack size.
        }

        public override Inst Convert(Converter converter, State state)
        {
            Value v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v);

            var load = v.V;
            load = LLVM.BuildLoad(Builder, load, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            // Load len.
            load = LLVM.BuildExtractValue(Builder, load, 1, "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            load = LLVM.BuildTrunc(Builder, load, LLVM.Int32Type(), "");
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new Value(load));

            state._stack.Push(new Value(load));
            return Next;
        }
    }

    public class i_ldloc : LdLoc
    {
        public i_ldloc(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloc_0 : LdLoc
    {
        public i_ldloc_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldloc_1 : LdLoc
    {
        public i_ldloc_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldloc_2 : LdLoc
    {
        public i_ldloc_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldloc_3 : LdLoc
    {
        public i_ldloc_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldloc_s : LdLoc
    {
        public i_ldloc_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca : LdLoc
    {
        public i_ldloca(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca_s : LdLoc
    {
        public i_ldloca_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldnull : Inst
    {
        public i_ldnull(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldobj : Inst
    {
        public i_ldobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldsfld : Inst
    {
        public i_ldsfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldsflda : Inst
    {
        public i_ldsflda(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldstr : Inst
    {
        public i_ldstr(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldtoken : Inst
    {
        public i_ldtoken(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldvirtftn : Inst
    {
        public i_ldvirtftn(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_leave : Inst
    {
        public i_leave(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_leave_s : Inst
    {
        public i_leave_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_localloc : Inst
    {
        public i_localloc(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_mkrefany : Inst
    {
        public i_mkrefany(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_mul : BinaryOpInst
    {
        public i_mul(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_mul_ovf : BinaryOpInst
    {
        public i_mul_ovf(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_mul_ovf_un : BinaryOpInst
    {
        public i_mul_ovf_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_neg : Inst
    {
        public i_neg(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_newarr : Inst
    {
        public i_newarr(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
    // Successor is fallthrough.
        int args = 0;
        int ret = 0;
        object method = this.Operand;
        if (method as Mono.Cecil.MethodReference != null)
        {
            Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
            args += mr.Parameters.Count;
            if (mr.MethodReturnType != null)
            {
                Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                Mono.Cecil.TypeReference tr = rt.ReturnType;
        // Get type, may contain modifiers.
                if (tr.FullName.Contains(' '))
                {
                    String[] sp = tr.FullName.Split(' ');
                    if (!sp[0].Equals("System.Void"))
                        ret++;
                }
                else
                {
                    if (!tr.FullName.Equals("System.Void"))
                        ret++;
                }
            }
            ret++;
        }
        level_after = level_after + ret - args;
    }
    }

    public class i_newobj : Inst
    {
        public i_newobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    public override void ComputeStackLevel(ref int level_after)
    {
    // Successor is fallthrough.
        int args = 0;
        int ret = 0;
        object method = this.Operand;
        if (method as Mono.Cecil.MethodReference != null)
        {
            Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
            args += mr.Parameters.Count;
            if (mr.MethodReturnType != null)
            {
                Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                Mono.Cecil.TypeReference tr = rt.ReturnType;
        // Get type, may contain modifiers.
                if (tr.FullName.Contains(' '))
                {
                    String[] sp = tr.FullName.Split(' ');
                    if (!sp[0].Equals("System.Void"))
                        ret++;
                }
                else
                {
                    if (!tr.FullName.Equals("System.Void"))
                        ret++;
                }
            }
            ret++;
        }
        level_after = level_after + ret - args;
    }
    }

    public class i_no : Inst
    {
        public i_no(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_nop : Inst
    {
        public i_nop(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // No change.
        }

        public override Inst Convert(Converter converter, State state)
        {
            return Next;
        }
    }

    public class i_not : Inst
    {
        public i_not(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_or : BinaryOpInst
    {
        public i_or(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_pop : Inst
    {
        public i_pop(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_readonly : Inst
    {
        public i_readonly(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_refanytype : Inst
    {
        public i_refanytype(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_refanyval : Inst
    {
        public i_refanyval(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_rem : BinaryOpInst
    {
        public i_rem(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_rem_un : BinaryOpInst
    {
        public i_rem_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ret : Inst
    {
        public i_ret(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            // There are really two different stacks here:
            // one for the called method, and the other for the caller of the method.
            // When returning, the stack of the method is pretty much unchanged.
            // In fact the top of stack often contains the return value from the method.
            // Back in the caller, the stack is popped of all arguments to the callee.
            // And, the return value is pushed on the top of stack.
            // This is handled by the call instruction.
        }

        public override Inst Convert(Converter converter, State state)
        {
            // There are really two different stacks here:
            // one for the called method, and the other for the caller of the method.
            // When returning, the stack of the method is pretty much unchanged.
            // In fact the top of stack often contains the return value from the method.
            // Back in the caller, the stack is popped of all arguments to the callee.
            // And, the return value is pushed on the top of stack.
            // This is handled by the call instruction.
            var ret = this.Block.ExpectedCalleeSignature.ReturnType;
            var void_type = typeof(void).ToMonoTypeReference();

            if (ret == null || ret.FullName == void_type.FullName)
            {
                var i = LLVM.BuildRetVoid(Builder);
            }
            else
            {
                var v = state._stack.Pop();
                var i = LLVM.BuildRet(Builder, v.V);
                state._stack.Push(new Value(i));
            }
            return Next;
        }
    }

    public class i_rethrow : Inst
    {
        public i_rethrow(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_shl : Inst
    {
        public i_shl(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_shr : Inst
    {
        public i_shr(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_shr_un : Inst
    {
        public i_shr_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_sizeof : Inst
    {
        public i_sizeof(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }

        public override Inst Convert(Converter converter, State state)
        {
            object operand = this.Operand;
            System.Type t = operand.GetType();
            if (t.FullName == "Mono.Cecil.PointerType")
                state._stack.Push(new Value(LLVM.ConstInt(LLVM.Int32Type(), 8, false)));
            else
                throw new Exception("Unimplemented sizeof");
            return Next;
        }
    }

    public class i_starg : Inst
    {
        int _arg;

        public i_starg(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_starg_s : Inst
    {
        int _arg;

        public i_starg_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_stelem_any : ConvertStoreElement
    {
        public i_stelem_any(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stelem_i1 : ConvertStoreElement
    {
        public i_stelem_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(sbyte));
        }
    }

    public class i_stelem_i2 : ConvertStoreElement
    {
        public i_stelem_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(short));
        }
    }

    public class i_stelem_i4 : ConvertStoreElement
    {
        public i_stelem_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(int));
        }
    }

    public class i_stelem_i8 : ConvertStoreElement
    {
        public i_stelem_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(long));
        }
    }

    public class i_stelem_i : ConvertStoreElement
    {
        public i_stelem_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stelem_r4 : ConvertStoreElement
    {
        public i_stelem_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(float));
        }
    }

    public class i_stelem_r8 : ConvertStoreElement
    {
        public i_stelem_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new Type(typeof(double));
        }
    }

    public class i_stelem_ref : ConvertStoreElement
    {
        public i_stelem_ref(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stfld : ConvertStoreField
    {
        public i_stfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_i1 : ConvertStoreIndirect
    {
        public i_stind_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_i2 : ConvertStoreIndirect
    {
        public i_stind_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_i4 : ConvertStoreIndirect
    {
        public i_stind_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_i8 : ConvertStoreIndirect
    {
        public i_stind_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_i : ConvertStoreIndirect
    {
        public i_stind_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_r4 : ConvertStoreIndirect
    {
        public i_stind_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_r8 : ConvertStoreIndirect
    {
        public i_stind_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stind_ref : ConvertStoreIndirect
    {
        public i_stind_ref(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_stloc : StLoc
    {
        public i_stloc(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stloc_0 : StLoc
    {
        public i_stloc_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_stloc_1 : StLoc
    {
        public i_stloc_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_stloc_2 : StLoc
    {
        public i_stloc_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_stloc_3 : StLoc
    {
        public i_stloc_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_stloc_s : StLoc
    {
        public i_stloc_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stobj : Inst
    {
        public i_stobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after -= 2;
    }
    }

    public class i_stsfld : Inst
    {
        public i_stsfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_sub : BinaryOpInst
    {
        public i_sub(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_sub_ovf : BinaryOpInst
    {
        public i_sub_ovf(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_sub_ovf_un : BinaryOpInst
    {
        public i_sub_ovf_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_switch : Inst
    {
        public i_switch(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_tail : Inst
    {
        public i_tail(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_throw : Inst
    {
        public i_throw(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_unaligned : Inst
    {
        public i_unaligned(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_unbox : Inst
    {
        public i_unbox(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_unbox_any : Inst
    {
        public i_unbox_any(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_volatile : Inst
    {
        public i_volatile(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_xor : BinaryOpInst
    {
        public i_xor(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }
}
