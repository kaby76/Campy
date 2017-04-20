using System;
using System.Collections.Generic;
using System.Diagnostics;
using Campy.CIL;
using Mono.Cecil.Cil;
using Swigged.LLVM;
using System.Linq;

namespace Campy.LCFG
{
    public class Inst
    {
        protected readonly CIL_Inst _instruction;
        public CIL_Inst Instruction
        {
            get
            {
                return _instruction;
            }
        }

        protected List<ValueRef> _llvm_instructions;
        public List<ValueRef> LLVMInstructions
        {
            get { return this._llvm_instructions; }
        }

        private readonly LLVMCFG.Vertex _block;
        public LLVMCFG.Vertex Block
        {
            get
            {
                return _block;
            }
        }

        public override string ToString()
        {
            return _instruction.ToString();
        }

        public Inst(CIL_Inst i, LLVMCFG.Vertex b)
        {
            _instruction = i;
            _block = b;
        }

        public Mono.Cecil.Cil.OpCode OpCode
        {
            get
            {
                return _instruction.OpCode;
            }
        }

        public object Operand
        {
            get
            {
                return _instruction.Operand;
            }
        }
        public virtual void ComputeStackLevel(ref int level_after)
        {
        }

        public virtual void Convert(ref State state)
        {
        }

        static public Inst Wrap(CIL_Inst i, LLVMCFG.Vertex b)
        {
            Mono.Cecil.Cil.OpCode op = i.OpCode;
            switch (op.Code)
            {
                case Mono.Cecil.Cil.Code.Add:
                    return new i_add(i, b);
                case Mono.Cecil.Cil.Code.Add_Ovf:
                    return new i_add_ovf(i, b);
                case Mono.Cecil.Cil.Code.Add_Ovf_Un:
                    return new i_add_ovf_un(i, b);
                case Mono.Cecil.Cil.Code.And:
                    return new i_and(i, b);
                case Mono.Cecil.Cil.Code.Arglist:
                    return new i_arglist(i, b);
                case Mono.Cecil.Cil.Code.Beq:
                    return new i_beq(i, b);
                case Mono.Cecil.Cil.Code.Beq_S:
                    return new i_beq(i, b);
                case Mono.Cecil.Cil.Code.Bge:
                    return new i_bge(i, b);
                case Mono.Cecil.Cil.Code.Bge_S:
                    return new i_bge_s(i, b);
                case Mono.Cecil.Cil.Code.Bge_Un:
                    return new i_bge_un(i, b);
                case Mono.Cecil.Cil.Code.Bge_Un_S:
                    return new i_bge_un_s(i, b);
                case Mono.Cecil.Cil.Code.Bgt:
                    return new i_bgt(i, b);
                case Mono.Cecil.Cil.Code.Bgt_S:
                    return new i_bgt_s(i, b);
                case Mono.Cecil.Cil.Code.Bgt_Un:
                    return new i_bgt_un(i, b);
                case Mono.Cecil.Cil.Code.Bgt_Un_S:
                    return new i_bgt_un_s(i, b);
                case Mono.Cecil.Cil.Code.Ble:
                    return new i_ble(i, b);
                case Mono.Cecil.Cil.Code.Ble_S:
                    return new i_ble_s(i, b);
                case Mono.Cecil.Cil.Code.Ble_Un:
                    return new i_ble_un(i, b);
                case Mono.Cecil.Cil.Code.Ble_Un_S:
                    return new i_ble_un_s(i, b);
                case Mono.Cecil.Cil.Code.Blt:
                    return new i_blt(i, b);
                case Mono.Cecil.Cil.Code.Blt_S:
                    return new i_blt_s(i, b);
                case Mono.Cecil.Cil.Code.Blt_Un:
                    return new i_blt_un(i, b);
                case Mono.Cecil.Cil.Code.Blt_Un_S:
                    return new i_blt_un_s(i, b);
                case Mono.Cecil.Cil.Code.Bne_Un:
                    return new i_bne_un(i, b);
                case Mono.Cecil.Cil.Code.Bne_Un_S:
                    return new i_bne_un_s(i, b);
                case Mono.Cecil.Cil.Code.Box:
                    return new i_box(i, b);
                case Mono.Cecil.Cil.Code.Br:
                    return new i_br(i, b);
                case Mono.Cecil.Cil.Code.Br_S:
                    return new i_br_s(i, b);
                case Mono.Cecil.Cil.Code.Break:
                    return new i_break(i, b);
                case Mono.Cecil.Cil.Code.Brfalse:
                    return new i_brfalse(i, b);
                case Mono.Cecil.Cil.Code.Brfalse_S:
                    return new i_brfalse_s(i, b);
                // Missing brnull
                // Missing brzero
                case Mono.Cecil.Cil.Code.Brtrue:
                    return new i_brtrue(i, b);
                case Mono.Cecil.Cil.Code.Brtrue_S:
                    return new i_brtrue_s(i, b);
                case Mono.Cecil.Cil.Code.Call:
                    return new i_call(i, b);
                case Mono.Cecil.Cil.Code.Calli:
                    return new i_calli(i, b);
                case Mono.Cecil.Cil.Code.Callvirt:
                    return new i_callvirt(i, b);
                case Mono.Cecil.Cil.Code.Castclass:
                    return new i_castclass(i, b);
                case Mono.Cecil.Cil.Code.Ceq:
                    return new i_ceq(i, b);
                case Mono.Cecil.Cil.Code.Cgt:
                    return new i_cgt(i, b);
                case Mono.Cecil.Cil.Code.Cgt_Un:
                    return new i_cgt_un(i, b);
                case Mono.Cecil.Cil.Code.Ckfinite:
                    return new i_ckfinite(i, b);
                case Mono.Cecil.Cil.Code.Clt:
                    return new i_clt(i, b);
                case Mono.Cecil.Cil.Code.Clt_Un:
                    return new i_clt_un(i, b);
                case Mono.Cecil.Cil.Code.Constrained:
                    return new i_constrained(i, b);
                case Mono.Cecil.Cil.Code.Conv_I1:
                    return new i_conv_i1(i, b);
                case Mono.Cecil.Cil.Code.Conv_I2:
                    return new i_conv_i2(i, b);
                case Mono.Cecil.Cil.Code.Conv_I4:
                    return new i_conv_i4(i, b);
                case Mono.Cecil.Cil.Code.Conv_I8:
                    return new i_conv_i8(i, b);
                case Mono.Cecil.Cil.Code.Conv_I:
                    return new i_conv_i(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I1:
                    return new i_conv_ovf_i1(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I1_Un:
                    return new i_conv_ovf_i1_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I2:
                    return new i_conv_ovf_i2(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I2_Un:
                    return new i_conv_ovf_i2_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I4:
                    return new i_conv_ovf_i4(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I4_Un:
                    return new i_conv_ovf_i4_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I8:
                    return new i_conv_ovf_i8(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I8_Un:
                    return new i_conv_ovf_i8_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I:
                    return new i_conv_ovf_i(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_I_Un:
                    return new i_conv_ovf_i_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U1:
                    return new i_conv_ovf_u1(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U1_Un:
                    return new i_conv_ovf_u1_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U2:
                    return new i_conv_ovf_u2(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U2_Un:
                    return new i_conv_ovf_u2_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U4:
                    return new i_conv_ovf_u4(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U4_Un:
                    return new i_conv_ovf_u4_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U8:
                    return new i_conv_ovf_u8(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U8_Un:
                    return new i_conv_ovf_u8_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U:
                    return new i_conv_ovf_u(i, b);
                case Mono.Cecil.Cil.Code.Conv_Ovf_U_Un:
                    return new i_conv_ovf_u_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_R4:
                    return new i_conv_r4(i, b);
                case Mono.Cecil.Cil.Code.Conv_R8:
                    return new i_conv_r8(i, b);
                case Mono.Cecil.Cil.Code.Conv_R_Un:
                    return new i_conv_r_un(i, b);
                case Mono.Cecil.Cil.Code.Conv_U1:
                    return new i_conv_u1(i, b);
                case Mono.Cecil.Cil.Code.Conv_U2:
                    return new i_conv_u2(i, b);
                case Mono.Cecil.Cil.Code.Conv_U4:
                    return new i_conv_u4(i, b);
                case Mono.Cecil.Cil.Code.Conv_U8:
                    return new i_conv_u8(i, b);
                case Mono.Cecil.Cil.Code.Conv_U:
                    return new i_conv_u(i, b);
                case Mono.Cecil.Cil.Code.Cpblk:
                    return new i_cpblk(i, b);
                case Mono.Cecil.Cil.Code.Cpobj:
                    return new i_cpobj(i, b);
                case Mono.Cecil.Cil.Code.Div:
                    return new i_div(i, b);
                case Mono.Cecil.Cil.Code.Div_Un:
                    return new i_div_un(i, b);
                case Mono.Cecil.Cil.Code.Dup:
                    return new i_dup(i, b);
                case Mono.Cecil.Cil.Code.Endfilter:
                    return new i_endfilter(i, b);
                case Mono.Cecil.Cil.Code.Endfinally:
                    return new i_endfinally(i, b);
                case Mono.Cecil.Cil.Code.Initblk:
                    return new i_initblk(i, b);
                case Mono.Cecil.Cil.Code.Initobj:
                    return new i_initobj(i, b);
                case Mono.Cecil.Cil.Code.Isinst:
                    return new i_isinst(i, b);
                case Mono.Cecil.Cil.Code.Jmp:
                    return new i_jmp(i, b);
                case Mono.Cecil.Cil.Code.Ldarg:
                    return new i_ldarg(i, b);
                case Mono.Cecil.Cil.Code.Ldarg_0:
                    return new i_ldarg_0(i, b);
                case Mono.Cecil.Cil.Code.Ldarg_1:
                    return new i_ldarg_1(i, b);
                case Mono.Cecil.Cil.Code.Ldarg_2:
                    return new i_ldarg_2(i, b);
                case Mono.Cecil.Cil.Code.Ldarg_3:
                    return new i_ldarg_3(i, b);
                case Mono.Cecil.Cil.Code.Ldarg_S:
                    return new i_ldarg_s(i, b);
                case Mono.Cecil.Cil.Code.Ldarga:
                    return new i_ldarga(i, b);
                case Mono.Cecil.Cil.Code.Ldarga_S:
                    return new i_ldarga_s(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4:
                    return new i_ldc_i4(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_0:
                    return new i_ldc_i4_0(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_1:
                    return new i_ldc_i4_1(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_2:
                    return new i_ldc_i4_2(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_3:
                    return new i_ldc_i4_3(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_4:
                    return new i_ldc_i4_4(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_5:
                    return new i_ldc_i4_5(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_6:
                    return new i_ldc_i4_6(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_7:
                    return new i_ldc_i4_7(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_8:
                    return new i_ldc_i4_8(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_M1:
                    return new i_ldc_i4_m1(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I4_S:
                    return new i_ldc_i4_s(i, b);
                case Mono.Cecil.Cil.Code.Ldc_I8:
                    return new i_ldc_i8(i, b);
                case Mono.Cecil.Cil.Code.Ldc_R4:
                    return new i_ldc_r4(i, b);
                case Mono.Cecil.Cil.Code.Ldc_R8:
                    return new i_ldc_r8(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_Any:
                    return new i_ldelem_any(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_I1:
                    return new i_ldelem_i1(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_I2:
                    return new i_ldelem_i2(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_I4:
                    return new i_ldelem_i4(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_I8:
                    return new i_ldelem_i8(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_I:
                    return new i_ldelem_i(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_R4:
                    return new i_ldelem_r4(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_R8:
                    return new i_ldelem_r8(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_Ref:
                    return new i_ldelem_ref(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_U1:
                    return new i_ldelem_u1(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_U2:
                    return new i_ldelem_u2(i, b);
                case Mono.Cecil.Cil.Code.Ldelem_U4:
                    return new i_ldelem_u4(i, b);
                case Mono.Cecil.Cil.Code.Ldelema:
                    return new i_ldelema(i, b);
                case Mono.Cecil.Cil.Code.Ldfld:
                    return new i_ldfld(i, b);
                case Mono.Cecil.Cil.Code.Ldflda:
                    return new i_ldflda(i, b);
                case Mono.Cecil.Cil.Code.Ldftn:
                    return new i_ldftn(i, b);
                case Mono.Cecil.Cil.Code.Ldind_I1:
                    return new i_ldind_i1(i, b);
                case Mono.Cecil.Cil.Code.Ldind_I2:
                    return new i_ldind_i2(i, b);
                case Mono.Cecil.Cil.Code.Ldind_I4:
                    return new i_ldind_i4(i, b);
                case Mono.Cecil.Cil.Code.Ldind_I8:
                    return new i_ldind_i8(i, b);
                case Mono.Cecil.Cil.Code.Ldind_I:
                    return new i_ldind_i(i, b);
                case Mono.Cecil.Cil.Code.Ldind_R4:
                    return new i_ldind_r4(i, b);
                case Mono.Cecil.Cil.Code.Ldind_R8:
                    return new i_ldind_r8(i, b);
                case Mono.Cecil.Cil.Code.Ldind_Ref:
                    return new i_ldind_ref(i, b);
                case Mono.Cecil.Cil.Code.Ldind_U1:
                    return new i_ldind_u1(i, b);
                case Mono.Cecil.Cil.Code.Ldind_U2:
                    return new i_ldind_u2(i, b);
                case Mono.Cecil.Cil.Code.Ldind_U4:
                    return new i_ldind_u4(i, b);
                case Mono.Cecil.Cil.Code.Ldlen:
                    return new i_ldlen(i, b);
                case Mono.Cecil.Cil.Code.Ldloc:
                    return new i_ldloc(i, b);
                case Mono.Cecil.Cil.Code.Ldloc_0:
                    return new i_ldloc_0(i, b);
                case Mono.Cecil.Cil.Code.Ldloc_1:
                    return new i_ldloc_1(i, b);
                case Mono.Cecil.Cil.Code.Ldloc_2:
                    return new i_ldloc_2(i, b);
                case Mono.Cecil.Cil.Code.Ldloc_3:
                    return new i_ldloc_3(i, b);
                case Mono.Cecil.Cil.Code.Ldloc_S:
                    return new i_ldloc_s(i, b);
                case Mono.Cecil.Cil.Code.Ldloca:
                    return new i_ldloca(i, b);
                case Mono.Cecil.Cil.Code.Ldloca_S:
                    return new i_ldloca_s(i, b);
                case Mono.Cecil.Cil.Code.Ldnull:
                    return new i_ldnull(i, b);
                case Mono.Cecil.Cil.Code.Ldobj:
                    return new i_ldobj(i, b);
                case Mono.Cecil.Cil.Code.Ldsfld:
                    return new i_ldsfld(i, b);
                case Mono.Cecil.Cil.Code.Ldsflda:
                    return new i_ldsflda(i, b);
                case Mono.Cecil.Cil.Code.Ldstr:
                    return new i_ldstr(i, b);
                case Mono.Cecil.Cil.Code.Ldtoken:
                    return new i_ldtoken(i, b);
                case Mono.Cecil.Cil.Code.Ldvirtftn:
                    return new i_ldvirtftn(i, b);
                case Mono.Cecil.Cil.Code.Leave:
                    return new i_leave(i, b);
                case Mono.Cecil.Cil.Code.Leave_S:
                    return new i_leave_s(i, b);
                case Mono.Cecil.Cil.Code.Localloc:
                    return new i_localloc(i, b);
                case Mono.Cecil.Cil.Code.Mkrefany:
                    return new i_mkrefany(i, b);
                case Mono.Cecil.Cil.Code.Mul:
                    return new i_mul(i, b);
                case Mono.Cecil.Cil.Code.Mul_Ovf:
                    return new i_mul_ovf(i, b);
                case Mono.Cecil.Cil.Code.Mul_Ovf_Un:
                    return new i_mul_ovf_un(i, b);
                case Mono.Cecil.Cil.Code.Neg:
                    return new i_neg(i, b);
                case Mono.Cecil.Cil.Code.Newarr:
                    return new i_newarr(i, b);
                case Mono.Cecil.Cil.Code.Newobj:
                    return new i_newobj(i, b);
                case Mono.Cecil.Cil.Code.No:
                    return new i_no(i, b);
                case Mono.Cecil.Cil.Code.Nop:
                    return new i_nop(i, b);
                case Mono.Cecil.Cil.Code.Not:
                    return new i_not(i, b);
                case Mono.Cecil.Cil.Code.Or:
                    return new i_or(i, b);
                case Mono.Cecil.Cil.Code.Pop:
                    return new i_pop(i, b);
                case Mono.Cecil.Cil.Code.Readonly:
                    return new i_readonly(i, b);
                case Mono.Cecil.Cil.Code.Refanytype:
                    return new i_refanytype(i, b);
                case Mono.Cecil.Cil.Code.Refanyval:
                    return new i_refanyval(i, b);
                case Mono.Cecil.Cil.Code.Rem:
                    return new i_rem(i, b);
                case Mono.Cecil.Cil.Code.Rem_Un:
                    return new i_rem_un(i, b);
                case Mono.Cecil.Cil.Code.Ret:
                    return new i_ret(i, b);
                case Mono.Cecil.Cil.Code.Rethrow:
                    return new i_rethrow(i, b);
                case Mono.Cecil.Cil.Code.Shl:
                    return new i_shl(i, b);
                case Mono.Cecil.Cil.Code.Shr:
                    return new i_shr(i, b);
                case Mono.Cecil.Cil.Code.Shr_Un:
                    return new i_shr_un(i, b);
                case Mono.Cecil.Cil.Code.Sizeof:
                    return new i_sizeof(i, b);
                case Mono.Cecil.Cil.Code.Starg:
                    return new i_starg(i, b);
                case Mono.Cecil.Cil.Code.Starg_S:
                    return new i_starg_s(i, b);
                case Mono.Cecil.Cil.Code.Stelem_Any:
                    return new i_stelem_any(i, b);
                case Mono.Cecil.Cil.Code.Stelem_I1:
                    return new i_stelem_i1(i, b);
                case Mono.Cecil.Cil.Code.Stelem_I2:
                    return new i_stelem_i2(i, b);
                case Mono.Cecil.Cil.Code.Stelem_I4:
                    return new i_stelem_i4(i, b);
                case Mono.Cecil.Cil.Code.Stelem_I8:
                    return new i_stelem_i8(i, b);
                case Mono.Cecil.Cil.Code.Stelem_I:
                    return new i_stelem_i(i, b);
                case Mono.Cecil.Cil.Code.Stelem_R4:
                    return new i_stelem_r4(i, b);
                case Mono.Cecil.Cil.Code.Stelem_R8:
                    return new i_stelem_r8(i, b);
                case Mono.Cecil.Cil.Code.Stelem_Ref:
                    return new i_stelem_ref(i, b);
                case Mono.Cecil.Cil.Code.Stfld:
                    return new i_stfld(i, b);
                case Mono.Cecil.Cil.Code.Stind_I1:
                    return new i_stind_i1(i, b);
                case Mono.Cecil.Cil.Code.Stind_I2:
                    return new i_stind_i2(i, b);
                case Mono.Cecil.Cil.Code.Stind_I4:
                    return new i_stind_i4(i, b);
                case Mono.Cecil.Cil.Code.Stind_I8:
                    return new i_stind_i8(i, b);
                case Mono.Cecil.Cil.Code.Stind_I:
                    return new i_stind_i(i, b);
                case Mono.Cecil.Cil.Code.Stind_R4:
                    return new i_stind_r4(i, b);
                case Mono.Cecil.Cil.Code.Stind_R8:
                    return new i_stind_r8(i, b);
                case Mono.Cecil.Cil.Code.Stind_Ref:
                    return new i_stind_ref(i, b);
                case Mono.Cecil.Cil.Code.Stloc:
                    return new i_stloc(i, b);
                case Mono.Cecil.Cil.Code.Stloc_0:
                    return new i_stloc_0(i, b);
                case Mono.Cecil.Cil.Code.Stloc_1:
                    return new i_stloc_1(i, b);
                case Mono.Cecil.Cil.Code.Stloc_2:
                    return new i_stloc_2(i, b);
                case Mono.Cecil.Cil.Code.Stloc_3:
                    return new i_stloc_3(i, b);
                case Mono.Cecil.Cil.Code.Stloc_S:
                    return new i_stloc_s(i, b);
                case Mono.Cecil.Cil.Code.Stobj:
                    return new i_stobj(i, b);
                case Mono.Cecil.Cil.Code.Stsfld:
                    return new i_stsfld(i, b);
                case Mono.Cecil.Cil.Code.Sub:
                    return new i_sub(i, b);
                case Mono.Cecil.Cil.Code.Sub_Ovf:
                    return new i_sub_ovf(i, b);
                case Mono.Cecil.Cil.Code.Sub_Ovf_Un:
                    return new i_sub_ovf_un(i, b);
                case Mono.Cecil.Cil.Code.Switch:
                    return new i_switch(i, b);
                case Mono.Cecil.Cil.Code.Tail:
                    return new i_tail(i, b);
                case Mono.Cecil.Cil.Code.Throw:
                    return new i_throw(i, b);
                case Mono.Cecil.Cil.Code.Unaligned:
                    return new i_unaligned(i, b);
                case Mono.Cecil.Cil.Code.Unbox:
                    return new i_unbox(i, b);
                case Mono.Cecil.Cil.Code.Unbox_Any:
                    return new i_unbox_any(i, b);
                case Mono.Cecil.Cil.Code.Volatile:
                    return new i_volatile(i, b);
                case Mono.Cecil.Cil.Code.Xor:
                    return new i_xor(i, b);
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

    }

    public class BinaryOpInst : Inst
    {
        public BinaryOpInst(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }

        public override void Convert(ref State state)
        {
            var lhs = state._stack.Pop();
            var rhs = state._stack.Pop();
            ValueRef tmp = LLVM.BuildAdd(this.Block.Builder, lhs.V, rhs.V, "tmp");
            Value result = new Value(tmp);
            state._stack.Push(result);
            var list = new List<ValueRef>();
            list.Add(tmp);
            this._llvm_instructions = list;
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

        UInt32 TargetPointerSizeInBits = 64;

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

        public Swigged.LLVM.ContextRef LLVMContext { get; set; }

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

        //// Generate a call to the throw helper if the condition is met.
        //public void genConditionalThrow(Value Condition, CorInfoHelpFunc HelperId,
        //    Twine ThrowBlockName)
        //{
        //    Value Arg1 = null, Arg2 = null;
        //    Type ReturnType = new Type(Type.getVoidTy(LLVMContext));
        //    bool MayThrow = true;
        //    bool CallReturns = false;
        //    genConditionalHelperCall(Condition, HelperId, MayThrow, ReturnType, Arg1,
        //        Arg2, CallReturns, ThrowBlockName);
        //}

        //CallSite genConditionalHelperCall(
        //    Value Condition, CorInfoHelpFunc HelperId,
        //    bool MayThrow, Type ReturnType,
        //    Value Arg1, Value Arg2, bool CallReturns, Twine CallBlockName)
        //{
        //    // Create the call block and fill it in.
        //    Swigged.LLVM.BasicBlockRef CallBlock = createPointBlock(CallBlockName);

            

        //    IRBuilder<>::InsertPoint SavedInsertPoint = LLVMBuilder->saveIP();
        //    LLVMBuilder->SetInsertPoint(CallBlock);
        //    CallSite HelperCall =
        //        callHelperImpl(HelperId, MayThrow, ReturnType, Arg1, Arg2);

        //    if (!CallReturns) {
        //        HelperCall.setDoesNotReturn();
        //        LLVMBuilder->CreateUnreachable();
        //    }
        //    LLVMBuilder->restoreIP(SavedInsertPoint);

        //    // Splice it into the flow.
        //    insertConditionalPointBlock(Condition, CallBlock, CallReturns);

        //    // Return the the call.
        //    return HelperCall;
        //}

        public BuilderRef Builder { get; set; } = LLVM.CreateBuilder();

        public bool UseExplicitZeroDivideChecks { get; set; }
    }


    public class i_add : BinaryOpInst
    {
        public i_add(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_add_ovf : BinaryOpInst
        {
        public i_add_ovf(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
        public override void ComputeStackLevel(ref int level_after)
        {
            level_after--;
        }
    }

    public class i_add_ovf_un : BinaryOpInst
        {
        public i_add_ovf_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_and : BinaryOpInst
        {
        public i_and(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_arglist : Inst
    {
        public i_arglist(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_beq : Inst
    {
        public i_beq(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_beq_s : Inst
    {
        public i_beq_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge : Inst
    {
        public i_bge(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge_un : Inst
    {
        public i_bge_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge_un_s : Inst
    {
        public i_bge_un_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge_s : Inst
    {
        public i_bge_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt : Inst
    {
        public i_bgt(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt_s : Inst
    {
        public i_bgt_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt_un : Inst
    {
        public i_bgt_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt_un_s : Inst
    {
        public i_bgt_un_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble : Inst
    {
        public i_ble(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble_s : Inst
    {
        public i_ble_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble_un : Inst
    {
        public i_ble_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble_un_s : Inst
    {
        public i_ble_un_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt : Inst
    {
        public i_blt(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt_s : Inst
    {
        public i_blt_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt_un : Inst
    {
        public i_blt_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt_un_s : Inst
    {
        public i_blt_un_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bne_un : Inst
    {
        public i_bne_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bne_un_s : Inst
    {
        public i_bne_un_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_box : Inst
    {
        public i_box(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_br : Inst
    {
        public i_br(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_br_s : Inst
    {
        public i_br_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brfalse : Inst
    {
        public i_brfalse(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_break : Inst
    {
        public i_break(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brfalse_s : Inst
    {
        public i_brfalse_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brtrue : Inst
    {
        public i_brtrue(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brtrue_s : Inst
    {
        public i_brtrue_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_call : Inst
    {
        public i_call(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_calli : Inst
    {
        public i_calli(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_callvirt : Inst
    {
        public i_callvirt(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_castclass : Inst
    {
        public i_castclass(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ceq : Inst
    {
        public i_ceq(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cgt : Inst
    {
        public i_cgt(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cgt_un : Inst
    {
        public i_cgt_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ckfinite : Inst
    {
        public i_ckfinite(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_clt : Inst
    {
        public i_clt(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_clt_un : Inst
    {
        public i_clt_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_constrained : Inst
    {
        public i_constrained(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i1 : Inst
    {
        public i_conv_i1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i2 : Inst
    {
        public i_conv_i2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i4 : Inst
    {
        public i_conv_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i8 : Inst
    {
        public i_conv_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i : Inst
    {
        public i_conv_i(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i1 : Inst
    {
        public i_conv_ovf_i1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i1_un : Inst
    {
        public i_conv_ovf_i1_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i2 : Inst
    {
        public i_conv_ovf_i2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i2_un : Inst
    {
        public i_conv_ovf_i2_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i4 : Inst
    {
        public i_conv_ovf_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i4_un : Inst
    {
        public i_conv_ovf_i4_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i8 : Inst
    {
        public i_conv_ovf_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i8_un : Inst
    {
        public i_conv_ovf_i8_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i : Inst
    {
        public i_conv_ovf_i(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i_un : Inst
    {
        public i_conv_ovf_i_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u1 : Inst
    {
        public i_conv_ovf_u1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u1_un : Inst
    {
        public i_conv_ovf_u1_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u2 : Inst
    {
        public i_conv_ovf_u2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u2_un : Inst
    {
        public i_conv_ovf_u2_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u4 : Inst
    {
        public i_conv_ovf_u4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u4_un : Inst
    {
        public i_conv_ovf_u4_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u8 : Inst
    {
        public i_conv_ovf_u8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u8_un : Inst
    {
        public i_conv_ovf_u8_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u : Inst
    {
        public i_conv_ovf_u(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u_un : Inst
    {
        public i_conv_ovf_u_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_r4 : Inst
    {
        public i_conv_r4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_r8 : Inst
    {
        public i_conv_r8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_r_un : Inst
    {
        public i_conv_r_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u1 : Inst
    {
        public i_conv_u1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u2 : Inst
    {
        public i_conv_u2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u4 : Inst
    {
        public i_conv_u4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u8 : Inst
    {
        public i_conv_u8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u : Inst
    {
        public i_conv_u(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cpblk : Inst
    {
        public i_cpblk(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cpobj : Inst
    {
        public i_cpobj(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_div : BinaryOpInst
        {
        public i_div(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_div_un : BinaryOpInst
        {
        public i_div_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_dup : Inst
    {
        public i_dup(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_endfilter : Inst
    {
        public i_endfilter(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_endfinally : Inst
    {
        public i_endfinally(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_initblk : Inst
    {
        public i_initblk(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_initobj : Inst
    {
        public i_initobj(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_isinst : Inst
    {
        public i_isinst(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_jmp : Inst
    {
        public i_jmp(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldarg : Inst
    {
        int _arg;

        public i_ldarg(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }
    }

    public class i_ldarg_0 : Inst
    {
        int _arg = 0;

        public i_ldarg_0(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldarg_1 : Inst
    {
        int _arg = 1;

        public i_ldarg_1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldarg_2 : Inst
    {
        int _arg = 2;

        public i_ldarg_2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldarg_3 : Inst
    {
        int _arg = 3;

        public i_ldarg_3(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldarg_s : Inst
    {
        int _arg;

        public i_ldarg_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }
    }

    public class i_ldarga : Inst
    {
        int _arg;

        public i_ldarga(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldarga_s : Inst
    {
        int _arg;

        public i_ldarga_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldc_i4 : Inst
    {
        int _arg;

        public i_ldc_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
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

    public class i_ldc_i4_0 : Inst
    {
        int _arg;

        public i_ldc_i4_0(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldc_i4_1 : Inst
    {
        int _arg;

        public i_ldc_i4_1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_2 : Inst
    {
        int _arg;

        public i_ldc_i4_2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldc_i4_3 : Inst
    {
        int _arg;

        public i_ldc_i4_3(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldc_i4_4 : Inst
    {
        int _arg;

        public i_ldc_i4_4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 4;
            _arg = arg;
        }
    }

    public class i_ldc_i4_5 : Inst
    {
        int _arg;

        public i_ldc_i4_5(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 5;
            _arg = arg;
        }
    }

    public class i_ldc_i4_6 : Inst
    {
        int _arg;

        public i_ldc_i4_6(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 6;
            _arg = arg;
        }
    }

    public class i_ldc_i4_7 : Inst
    {
        int _arg;

        public i_ldc_i4_7(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 7;
            _arg = arg;
        }
    }

    public class i_ldc_i4_8 : Inst
    {
        int _arg;

        public i_ldc_i4_8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 8;
            _arg = arg;
        }
    }

    public class i_ldc_i4_m1 : Inst
    {
        int _arg;

        public i_ldc_i4_m1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = -1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_s : Inst
    {
        int _arg;

        public i_ldc_i4_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
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

    public class i_ldc_i8 : Inst
    {
        Int64 _arg;

        public i_ldc_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
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

    public class i_ldc_r4 : Inst
    {
        Single _arg;

        public i_ldc_r4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
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

    public class i_ldc_r8 : Inst
    {
        Double _arg;

        public i_ldc_r8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
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
    }

    public class i_ldelem_any : Inst
    {
        public i_ldelem_any(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_i1 : Inst
    {
        public i_ldelem_i1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_i2 : Inst
    {
        public i_ldelem_i2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_i4 : Inst
    {
        public i_ldelem_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_i8 : Inst
    {
        public i_ldelem_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_i : Inst
    {
        public i_ldelem_i(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_r4 : Inst
    {
        public i_ldelem_r4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_r8 : Inst
    {
        public i_ldelem_r8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_ref : Inst
    {
        public i_ldelem_ref(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_u1 : Inst
    {
        public i_ldelem_u1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_u2 : Inst
    {
        public i_ldelem_u2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelem_u4 : Inst
    {
        public i_ldelem_u4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldelema : Inst
    {
        public i_ldelema(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldfld : Inst
    {
        public i_ldfld(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldflda : Inst
    {
        public i_ldflda(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldftn : Inst
    {
        public i_ldftn(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i1 : Inst
    {
        public i_ldind_i1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i2 : Inst
    {
        public i_ldind_i2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i4 : Inst
    {
        public i_ldind_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i8 : Inst
    {
        public i_ldind_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i : Inst
    {
        public i_ldind_i(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_r4 : Inst
    {
        public i_ldind_r4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_r8 : Inst
    {
        public i_ldind_r8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_ref : Inst
    {
        public i_ldind_ref(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_u1 : Inst
    {
        public i_ldind_u1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_u2 : Inst
    {
        public i_ldind_u2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_u4 : Inst
    {
        public i_ldind_u4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldlen : Inst
    {
        public i_ldlen(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldloc : Inst
    {
        int _arg;

        public i_ldloc(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloc_0 : Inst
    {
        int _arg;

        public i_ldloc_0(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldloc_1 : Inst
    {
        int _arg;

        public i_ldloc_1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldloc_2 : Inst
    {
        int _arg;

        public i_ldloc_2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldloc_3 : Inst
    {
        int _arg;

        public i_ldloc_3(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldloc_s : Inst
    {
        int _arg;

        public i_ldloc_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca : Inst
    {
        int _arg;

        public i_ldloca(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca_s : Inst
    {
        int _arg;

        public i_ldloca_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldnull : Inst
    {
        public i_ldnull(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldobj : Inst
    {
        public i_ldobj(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldsfld : Inst
    {
        public i_ldsfld(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldsflda : Inst
    {
        public i_ldsflda(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldstr : Inst
    {
        public i_ldstr(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldtoken : Inst
    {
        public i_ldtoken(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldvirtftn : Inst
    {
        public i_ldvirtftn(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_leave : Inst
    {
        public i_leave(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_leave_s : Inst
    {
        public i_leave_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_localloc : Inst
    {
        public i_localloc(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_mkrefany : Inst
    {
        public i_mkrefany(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_mul : BinaryOpInst
        {
        public i_mul(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_mul_ovf : BinaryOpInst
        {
        public i_mul_ovf(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_mul_ovf_un : BinaryOpInst
        {
        public i_mul_ovf_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_neg : Inst
    {
        public i_neg(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_newarr : Inst
    {
        public i_newarr(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_newobj : Inst
    {
        public i_newobj(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_no : Inst
    {
        public i_no(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_nop : Inst
    {
        public i_nop(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_not : Inst
    {
        public i_not(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_or : BinaryOpInst
        {
        public i_or(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_pop : Inst
    {
        public i_pop(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_readonly : Inst
    {
        public i_readonly(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_refanytype : Inst
    {
        public i_refanytype(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_refanyval : Inst
    {
        public i_refanyval(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_rem : BinaryOpInst
        {
        public i_rem(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_rem_un : BinaryOpInst
        {
        public i_rem_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ret : Inst
    {
        public i_ret(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_rethrow : Inst
    {
        public i_rethrow(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_shl : Inst
    {
        public i_shl(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_shr : Inst
    {
        public i_shr(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_shr_un : Inst
    {
        public i_shr_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_sizeof : Inst
    {
        public i_sizeof(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_starg : Inst
    {
        int _arg;

        public i_starg(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_starg_s : Inst
    {
        int _arg;

        public i_starg_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stelem_any : Inst
    {
        public i_stelem_any(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_i1 : Inst
    {
        public i_stelem_i1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_i2 : Inst
    {
        public i_stelem_i2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_i4 : Inst
    {
        public i_stelem_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_i8 : Inst
    {
        public i_stelem_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_i : Inst
    {
        public i_stelem_i(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_r4 : Inst
    {
        public i_stelem_r4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_r8 : Inst
    {
        public i_stelem_r8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stelem_ref : Inst
    {
        public i_stelem_ref(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stfld : Inst
    {
        public i_stfld(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_i1 : Inst
    {
        public i_stind_i1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_i2 : Inst
    {
        public i_stind_i2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_i4 : Inst
    {
        public i_stind_i4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_i8 : Inst
    {
        public i_stind_i8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_i : Inst
    {
        public i_stind_i(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_r4 : Inst
    {
        public i_stind_r4(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_r8 : Inst
    {
        public i_stind_r8(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stind_ref : Inst
    {
        public i_stind_ref(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stloc : Inst
    {
        int _arg;

        public i_stloc(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stloc_0 : Inst
    {
        int _arg;

        public i_stloc_0(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_stloc_1 : Inst
    {
        int _arg;

        public i_stloc_1(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_stloc_2 : Inst
    {
        int _arg;

        public i_stloc_2(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_stloc_3 : Inst
    {
        int _arg;

        public i_stloc_3(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_stloc_s : Inst
    {
        int _arg;

        public i_stloc_s(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stobj : Inst
    {
        public i_stobj(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_stsfld : Inst
    {
        public i_stsfld(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_sub : BinaryOpInst
        {
        public i_sub(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_sub_ovf : BinaryOpInst
        {
        public i_sub_ovf(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_sub_ovf_un : BinaryOpInst
        {
        public i_sub_ovf_un(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_switch : Inst
    {
        public i_switch(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_tail : Inst
    {
        public i_tail(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_throw : Inst
    {
        public i_throw(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_unaligned : Inst
    {
        public i_unaligned(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_unbox : Inst
    {
        public i_unbox(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_unbox_any : Inst
    {
        public i_unbox_any(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_volatile : Inst
    {
        public i_volatile(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_xor : BinaryOpInst
        {
        public i_xor(CIL_Inst i, LLVMCFG.Vertex b)
            : base(i, b)
        {
        }
    }
}
