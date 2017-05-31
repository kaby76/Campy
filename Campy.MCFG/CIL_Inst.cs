using System;
using System.Collections.Generic;
using System.Linq;
using Mono.Cecil.Cil;

namespace Campy.CIL
{
    public class CIL_Inst
    {
        public Mono.Cecil.Cil.Instruction Instruction { get; private set; }
        public CIL_CFG.Vertex Block { get; private set; }
        public static List<CIL_Inst> CallInstructions { get; private set; } = new List<CIL_Inst>();
        public override string ToString() { return Instruction.ToString(); }
        public Mono.Cecil.Cil.OpCode OpCode { get { return Instruction.OpCode; } }
        public object Operand { get { return Instruction.Operand; } }


        public CIL_Inst(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
        {
            Instruction = i;
            Block = b;
            if (i.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Call)
            {
                CIL_Inst.CallInstructions.Add(this);
            }
        }

        static public CIL_Inst Wrap(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
        {
            // Wrap instruction with semantics, def/use/kill properties.
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
    }

    public class LoadArgInst : CIL_Inst
    {
        public int _arg;

        public LoadArgInst(Instruction i, CIL_CFG.Vertex b) : base(i, b)
        {
        }
    }

    public class LDCInstI4 : CIL_Inst
    {
        public Int32 _arg;

        public LDCInstI4(Instruction i, CIL_CFG.Vertex b) : base(i, b)
        {
        }
    }

    public class LDCInstI8 : CIL_Inst
    {
        public Int64 _arg;

        public LDCInstI8(Instruction i, CIL_CFG.Vertex b) : base(i, b)
        {
        }
    }

    public class LdLoc : CIL_Inst
    {
        public int _arg;

        public LdLoc(Instruction i, CIL_CFG.Vertex b) : base(i, b)
        {
        }
    }

    public class StLoc : CIL_Inst
    {
        public int _arg;

        public StLoc(Instruction i, CIL_CFG.Vertex b) : base(i, b)
        {
        }
    }

    public class i_add : CIL_Inst
    {
        public i_add(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_add_ovf : CIL_Inst
    {
        public i_add_ovf(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_add_ovf_un : CIL_Inst
    {
        public i_add_ovf_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_and : CIL_Inst
    {
        public i_and(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_arglist : CIL_Inst
    {
        public i_arglist(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_beq : CIL_Inst
    {
        public i_beq(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_beq_s : CIL_Inst
    {
        public i_beq_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge : CIL_Inst
    {
        public i_bge(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge_un : CIL_Inst
    {
        public i_bge_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge_un_s : CIL_Inst
    {
        public i_bge_un_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bge_s : CIL_Inst
    {
        public i_bge_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt : CIL_Inst
    {
        public i_bgt(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt_s : CIL_Inst
    {
        public i_bgt_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt_un : CIL_Inst
    {
        public i_bgt_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bgt_un_s : CIL_Inst
    {
        public i_bgt_un_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble : CIL_Inst
    {
        public i_ble(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble_s : CIL_Inst
    {
        public i_ble_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble_un : CIL_Inst
    {
        public i_ble_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ble_un_s : CIL_Inst
    {
        public i_ble_un_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt : CIL_Inst
    {
        public i_blt(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt_s : CIL_Inst
    {
        public i_blt_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt_un : CIL_Inst
    {
        public i_blt_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_blt_un_s : CIL_Inst
    {
        public i_blt_un_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bne_un : CIL_Inst
    {
        public i_bne_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_bne_un_s : CIL_Inst
    {
        public i_bne_un_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_box : CIL_Inst
    {
        public i_box(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_br : CIL_Inst
    {
        public i_br(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_br_s : CIL_Inst
    {
        public i_br_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brfalse : CIL_Inst
    {
        public i_brfalse(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_break : CIL_Inst
    {
        public i_break(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brfalse_s : CIL_Inst
    {
        public i_brfalse_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brtrue : CIL_Inst
    {
        public i_brtrue(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_brtrue_s : CIL_Inst
    {
        public i_brtrue_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_call : CIL_Inst
    {
        public i_call(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_calli : CIL_Inst
    {
        public i_calli(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_callvirt : CIL_Inst
    {
        public i_callvirt(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_castclass : CIL_Inst
    {
        public i_castclass(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ceq : CIL_Inst
    {
        public i_ceq(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cgt : CIL_Inst
    {
        public i_cgt(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cgt_un : CIL_Inst
    {
        public i_cgt_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ckfinite : CIL_Inst
    {
        public i_ckfinite(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_clt : CIL_Inst
    {
        public i_clt(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_clt_un : CIL_Inst
    {
        public i_clt_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_constrained : CIL_Inst
    {
        public i_constrained(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i1 : CIL_Inst
    {
        public i_conv_i1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i2 : CIL_Inst
    {
        public i_conv_i2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i4 : CIL_Inst
    {
        public i_conv_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i8 : CIL_Inst
    {
        public i_conv_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_i : CIL_Inst
    {
        public i_conv_i(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i1 : CIL_Inst
    {
        public i_conv_ovf_i1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i1_un : CIL_Inst
    {
        public i_conv_ovf_i1_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i2 : CIL_Inst
    {
        public i_conv_ovf_i2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i2_un : CIL_Inst
    {
        public i_conv_ovf_i2_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i4 : CIL_Inst
    {
        public i_conv_ovf_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i4_un : CIL_Inst
    {
        public i_conv_ovf_i4_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i8 : CIL_Inst
    {
        public i_conv_ovf_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i8_un : CIL_Inst
    {
        public i_conv_ovf_i8_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i : CIL_Inst
    {
        public i_conv_ovf_i(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_i_un : CIL_Inst
    {
        public i_conv_ovf_i_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u1 : CIL_Inst
    {
        public i_conv_ovf_u1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u1_un : CIL_Inst
    {
        public i_conv_ovf_u1_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u2 : CIL_Inst
    {
        public i_conv_ovf_u2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u2_un : CIL_Inst
    {
        public i_conv_ovf_u2_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u4 : CIL_Inst
    {
        public i_conv_ovf_u4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u4_un : CIL_Inst
    {
        public i_conv_ovf_u4_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u8 : CIL_Inst
    {
        public i_conv_ovf_u8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u8_un : CIL_Inst
    {
        public i_conv_ovf_u8_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u : CIL_Inst
    {
        public i_conv_ovf_u(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_ovf_u_un : CIL_Inst
    {
        public i_conv_ovf_u_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_r4 : CIL_Inst
    {
        public i_conv_r4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_r8 : CIL_Inst
    {
        public i_conv_r8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_r_un : CIL_Inst
    {
        public i_conv_r_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u1 : CIL_Inst
    {
        public i_conv_u1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u2 : CIL_Inst
    {
        public i_conv_u2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u4 : CIL_Inst
    {
        public i_conv_u4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u8 : CIL_Inst
    {
        public i_conv_u8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_conv_u : CIL_Inst
    {
        public i_conv_u(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cpblk : CIL_Inst
    {
        public i_cpblk(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_cpobj : CIL_Inst
    {
        public i_cpobj(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_div : CIL_Inst
    {
        public i_div(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_div_un : CIL_Inst
    {
        public i_div_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_dup : CIL_Inst
    {
        public i_dup(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_endfilter : CIL_Inst
    {
        public i_endfilter(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_endfinally : CIL_Inst
    {
        public i_endfinally(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_initblk : CIL_Inst
    {
        public i_initblk(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_initobj : CIL_Inst
    {
        public i_initobj(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_isinst : CIL_Inst
    {
        public i_isinst(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_jmp : CIL_Inst
    {
        public i_jmp(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldarg : LoadArgInst
    {
        public i_ldarg(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldarg_0 : LoadArgInst
    {
        public i_ldarg_0(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            _arg = 0;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldarg_1 : LoadArgInst
    {
        public i_ldarg_1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            _arg = 1;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldarg_2 : LoadArgInst
    {
        public i_ldarg_2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            _arg = 2;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldarg_3 : LoadArgInst
    {

        public i_ldarg_3(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            _arg = 3;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldarg_s : LoadArgInst
    {
        public i_ldarg_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }

        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldarga : CIL_Inst
    {
        int _arg;

        public i_ldarga(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldarga_s : CIL_Inst
    {
        int _arg;

        public i_ldarga_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldc_i4 : LDCInst4
    {
        public i_ldc_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
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

    public class i_ldc_i4_0 : LDCInst4
    {
        public i_ldc_i4_0(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldc_i4_1 : LDCInst4
    {
        public i_ldc_i4_1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_2 : LDCInst4
    {
        public i_ldc_i4_2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldc_i4_3 : LDCInst4
    {
        public i_ldc_i4_3(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldc_i4_4 : LDCInst4
    {
        public i_ldc_i4_4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 4;
            _arg = arg;
        }
    }

    public class i_ldc_i4_5 : LDCInst4
    {
        public i_ldc_i4_5(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 5;
            _arg = arg;
        }
    }

    public class i_ldc_i4_6 : LDCInst4
    {
        public i_ldc_i4_6(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 6;
            _arg = arg;
        }
    }

    public class i_ldc_i4_7 : LDCInstI4
    {
        public i_ldc_i4_7(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 7;
            _arg = arg;
        }
    }

    public class i_ldc_i4_8 : LDCInst4
    {
        public i_ldc_i4_8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 8;
            _arg = arg;
        }
    }

    public class i_ldc_i4_m1 : LDCInst4
    {
        public i_ldc_i4_m1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = -1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_s : LDCInst4
    {
        public i_ldc_i4_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
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

    public class i_ldc_i8 : LDCInst8
    {
        public i_ldc_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
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

    public class i_ldc_r4 : CIL_Inst
    {
        public Single _arg;

        public i_ldc_r4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
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
        public override void ComputeStackLevel(ref int level_after)
        {
            level_after++;
        }
    }

    public class i_ldc_r8 : CIL_Inst
    {
        Double _arg;

        public i_ldc_r8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
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

    public class i_ldelem_any : CIL_Inst
    {
        public i_ldelem_any(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_i1 : CIL_Inst
    {
        public i_ldelem_i1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_i2 : CIL_Inst
    {
        public i_ldelem_i2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_i4 : CIL_Inst
    {
        public i_ldelem_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_i8 : CIL_Inst
    {
        public i_ldelem_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_i : CIL_Inst
    {
        public i_ldelem_i(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_r4 : CIL_Inst
    {
        public i_ldelem_r4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_r8 : CIL_Inst
    {
        public i_ldelem_r8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_ref : CIL_Inst
    {
        public i_ldelem_ref(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_u1 : CIL_Inst
    {
        public i_ldelem_u1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_u2 : CIL_Inst
    {
        public i_ldelem_u2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelem_u4 : CIL_Inst
    {
        public i_ldelem_u4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldelema : CIL_Inst
    {
        public i_ldelema(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ldfld : CIL_Inst
    {
        public i_ldfld(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldflda : CIL_Inst
    {
        public i_ldflda(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldftn : CIL_Inst
    {
        public i_ldftn(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldind_i1 : CIL_Inst
    {
        public i_ldind_i1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i2 : CIL_Inst
    {
        public i_ldind_i2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i4 : CIL_Inst
    {
        public i_ldind_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i8 : CIL_Inst
    {
        public i_ldind_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_i : CIL_Inst
    {
        public i_ldind_i(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_r4 : CIL_Inst
    {
        public i_ldind_r4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_r8 : CIL_Inst
    {
        public i_ldind_r8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_ref : CIL_Inst
    {
        public i_ldind_ref(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_u1 : CIL_Inst
    {
        public i_ldind_u1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_u2 : CIL_Inst
    {
        public i_ldind_u2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldind_u4 : CIL_Inst
    {
        public i_ldind_u4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldlen : CIL_Inst
    {
        public i_ldlen(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldloc : LdLoc
    {
        public i_ldloc(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloc_0 : LdLoc
    {
        public i_ldloc_0(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldloc_1 : LdLoc
    {
        public i_ldloc_1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldloc_2 : LdLoc
    {
        public i_ldloc_2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldloc_3 : LdLoc
    {
        public i_ldloc_3(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldloc_s : LdLoc
    {
        public i_ldloc_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca : LdLoc
    {
        public i_ldloca(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca_s : LdLoc
    {
        public i_ldloca_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldnull : CIL_Inst
    {
        public i_ldnull(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldobj : CIL_Inst
    {
        public i_ldobj(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_ldsfld : CIL_Inst
    {
        public i_ldsfld(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldsflda : CIL_Inst
    {
        public i_ldsflda(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldstr : CIL_Inst
    {
        public i_ldstr(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldtoken : CIL_Inst
    {
        public i_ldtoken(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after++;
    }
    }

    public class i_ldvirtftn : CIL_Inst
    {
        public i_ldvirtftn(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_leave : CIL_Inst
    {
        public i_leave(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_leave_s : CIL_Inst
    {
        public i_leave_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_localloc : CIL_Inst
    {
        public i_localloc(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_mkrefany : CIL_Inst
    {
        public i_mkrefany(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_mul : CIL_Inst
    {
        public i_mul(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_mul_ovf : CIL_Inst
    {
        public i_mul_ovf(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_mul_ovf_un : CIL_Inst
    {
        public i_mul_ovf_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_neg : CIL_Inst
    {
        public i_neg(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_newarr : CIL_Inst
    {
        public i_newarr(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
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

    public class i_newobj : CIL_Inst
    {
        public i_newobj(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
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

    public class i_no : CIL_Inst
    {
        public i_no(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_nop : CIL_Inst
    {
        public i_nop(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_not : CIL_Inst
    {
        public i_not(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_or : CIL_Inst
    {
        public i_or(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_pop : CIL_Inst
    {
        public i_pop(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_readonly : CIL_Inst
    {
        public i_readonly(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_refanytype : CIL_Inst
    {
        public i_refanytype(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_refanyval : CIL_Inst
    {
        public i_refanyval(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_rem : CIL_Inst
    {
        public i_rem(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_rem_un : CIL_Inst
    {
        public i_rem_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_ret : CIL_Inst
    {
        public i_ret(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
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
    }

    public class i_rethrow : CIL_Inst
    {
        public i_rethrow(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_shl : CIL_Inst
    {
        public i_shl(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_shr : CIL_Inst
    {
        public i_shr(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_shr_un : CIL_Inst
    {
        public i_shr_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_sizeof : CIL_Inst
    {
        public i_sizeof(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_starg : CIL_Inst
    {
        int _arg;

        public i_starg(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
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

    public class i_starg_s : CIL_Inst
    {
        int _arg;

        public i_starg_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
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

    public class i_stelem_any : CIL_Inst
    {
        public i_stelem_any(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_i1 : CIL_Inst
    {
        public i_stelem_i1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_i2 : CIL_Inst
    {
        public i_stelem_i2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_i4 : CIL_Inst
    {
        public i_stelem_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_i8 : CIL_Inst
    {
        public i_stelem_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_i : CIL_Inst
    {
        public i_stelem_i(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_r4 : CIL_Inst
    {
        public i_stelem_r4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_r8 : CIL_Inst
    {
        public i_stelem_r8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stelem_ref : CIL_Inst
    {
        public i_stelem_ref(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 3;
    }
    }

    public class i_stfld : CIL_Inst
    {
        public i_stfld(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_i1 : CIL_Inst
    {
        public i_stind_i1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_i2 : CIL_Inst
    {
        public i_stind_i2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_i4 : CIL_Inst
    {
        public i_stind_i4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_i8 : CIL_Inst
    {
        public i_stind_i8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_i : CIL_Inst
    {
        public i_stind_i(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_r4 : CIL_Inst
    {
        public i_stind_r4(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_r8 : CIL_Inst
    {
        public i_stind_r8(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stind_ref : CIL_Inst
    {
        public i_stind_ref(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after = level_after - 2;
    }
    }

    public class i_stloc : StLoc
    {
        public i_stloc(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stloc_0 : StLoc
    {
        public i_stloc_0(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_stloc_1 : StLoc
    {
        public i_stloc_1(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_stloc_2 : StLoc
    {
        public i_stloc_2(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_stloc_3 : StLoc
    {
        public i_stloc_3(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_stloc_s : StLoc
    {
        public i_stloc_s(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stobj : CIL_Inst
    {
        public i_stobj(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    public override void ComputeStackLevel(ref int level_after)
    {
        level_after -= 2;
    }
    }

    public class i_stsfld : CIL_Inst
    {
        public i_stsfld(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_sub : CIL_Inst
    {
        public i_sub(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_sub_ovf : CIL_Inst
    {
        public i_sub_ovf(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_sub_ovf_un : CIL_Inst
    {
        public i_sub_ovf_un(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_switch : CIL_Inst
    {
        public i_switch(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }

    public class i_tail : CIL_Inst
    {
        public i_tail(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_throw : CIL_Inst
    {
        public i_throw(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_unaligned : CIL_Inst
    {
        public i_unaligned(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_unbox : CIL_Inst
    {
        public i_unbox(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_unbox_any : CIL_Inst
    {
        public i_unbox_any(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_volatile : CIL_Inst
    {
        public i_volatile(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }
    }

    public class i_xor : CIL_Inst
    {
        public i_xor(Mono.Cecil.Cil.Instruction i, CIL_CFG.Vertex b)
            : base(i, b)
        {
        }

    public override void ComputeStackLevel(ref int level_after)
    {
        level_after--;
    }
    }
}
