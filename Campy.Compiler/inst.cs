namespace Campy.Compiler
{
    using Mono.Cecil.Cil;
    using Mono.Cecil;
    using Mono.Collections.Generic;
    using Swigged.LLVM;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Text.RegularExpressions;
    using System;
    using Utils;
    using Campy.Meta;
    using System.Runtime.InteropServices;
    using Campy.Compiler.Graph;

    /// <summary>
    /// Wrapper for CIL instructions that are implemented using Mono.Cecil.Cil.
    /// This class adds basic block graph structure on top of these instructions. There
    /// is no semantics encoded in the wrapper.
    /// </summary>
    public class INST : Campy.Compiler.Graph.IInst
    {
        public Mono.Cecil.Cil.Instruction Instruction { get; set; }
        public Mono.Cecil.Cil.MethodBody Body { get; private set; }
        public static List<INST> CallInstructions { get; private set; } = new List<INST>();
        public override string ToString() { return Instruction.ToString(); }
        public Mono.Cecil.Cil.OpCode OpCode { get { return Instruction.OpCode; } }
        public int Offset { get { return Instruction.Offset; } }
        public object Operand { get { return Instruction.Operand; } }
        public static int instruction_id = 1;
        public BuilderRef Builder { get { return Block.LlvmInfo.Builder; } }
        public List<VALUE> LLVMInstructions { get; private set; }
        public CFG.Vertex Block { get; set; }
        public virtual INST Next { get; set; }
        public SequencePoint SeqPoint { get; set; }
        private static Dictionary<string, MetadataRef> debug_files = new Dictionary<string, MetadataRef>();
        private static Dictionary<string, MetadataRef> debug_compile_units = new Dictionary<string, MetadataRef>();
        private static Dictionary<string, MetadataRef> debug_methods = new Dictionary<string, MetadataRef>();
        private static Dictionary<string, MetadataRef> debug_blocks = new Dictionary<string, MetadataRef>();
        public static DIBuilderRef dib;
        private static bool done_this;
        public UInt32 TargetPointerSizeInBits = 64;

        public virtual void DebuggerInfo()
        {
            JITER converter = JITER.Singleton;
            if (this.SeqPoint == null)
                return;
            if (this.SeqPoint.IsHidden)
                return;

            if (!done_this)
            {
                done_this = true;
                dib = LLVM.CreateDIBuilder(RUNTIME.global_llvm_module);
            }
            var doc = SeqPoint.Document;
            string assembly_name = this.Block._original_method_reference.Module.FileName;
            string loc = Path.GetDirectoryName(Path.GetFullPath(doc.Url));
            string file_name = Path.GetFileName(doc.Url);
            MetadataRef file;
            if (!debug_files.ContainsKey(file_name))
            {
                file = LLVM.DIBuilderCreateFile(dib,
                    file_name, (uint)file_name.Length, loc, (uint)loc.Length);
                debug_files[file_name] = file;
            }
            else
            {
                file = debug_files[file_name];
            }

            string producer = "Campy Compiler";
            MetadataRef compile_unit;
            if (!debug_compile_units.ContainsKey(file_name))
            {
                compile_unit = LLVM.DIBuilderCreateCompileUnit(
                    dib,
                    DWARFSourceLanguage.DWARFSourceLanguageJava,
                    file, producer, (uint)producer.Length,
                    false, "", 0, 0, "", 0, DWARFEmissionKind.DWARFEmissionFull,
                    0, false, false);
                debug_compile_units[file_name] = compile_unit;
            }
            else
            {
                compile_unit = debug_compile_units[file_name];
            }

            ContextRef context_ref = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
            var normalized_method_name = METAHELPER.RenameToLegalLLVMName(
                JITER.MethodName(this.Block._original_method_reference));
            MetadataRef sub;
            if (!debug_methods.ContainsKey(normalized_method_name))
            {
                var sub_type = LLVM.DIBuilderCreateSubroutineType(
                    dib,
                    file, new MetadataRef[0], 0, DIFlags.DIFlagNoReturn);
                sub = LLVM.DIBuilderCreateFunction(dib, file,
                    normalized_method_name,
                    normalized_method_name,
                    file,
                    (uint) this.SeqPoint.StartLine,
                    sub_type, 1, 1, (uint) this.SeqPoint.StartLine, 0, 0);

                debug_methods[normalized_method_name] = sub;
                LLVM.SetSubprogram(this.Block.LlvmInfo.MethodValueRef, sub);
            }
            else {
                sub = debug_methods[normalized_method_name];
            }

            MetadataRef lexical_scope;
            if (!debug_blocks.ContainsKey(this.Block.Name))
            {
                lexical_scope = LLVM.DIBuilderCreateLexicalBlock(
                    dib, sub, file,
                    (uint)this.SeqPoint.StartLine,
                    (uint)this.SeqPoint.StartColumn);
                debug_blocks[this.Block.Name] = lexical_scope;
            }
            else
            {
                lexical_scope = debug_blocks[this.Block.Name];
            }

            MetadataRef debug_location = LLVM.DIBuilderCreateDebugLocation(
                LLVM.GetModuleContext(RUNTIME.global_llvm_module),
                (uint)this.SeqPoint.StartLine,
                (uint)this.SeqPoint.StartColumn,
                lexical_scope,
                default(MetadataRef)
                );
            var dv = LLVM.MetadataAsValue(LLVM.GetModuleContext(RUNTIME.global_llvm_module), debug_location);
            LLVM.SetCurrentDebugLocation(Builder, dv);

            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("Created debug loc " + dv);

        }

        public virtual void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            throw new Exception("Must have an implementation for GenerateGenerics! The instruction is: "
                                + this.ToString());
        }

        public virtual unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            throw new Exception("Must have an implementation for Convert! The instruction is: "
                                + this.ToString());
        }

        protected INST(Mono.Cecil.Cil.Instruction i)
        {
            Instruction = i;
            if (i.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Call)
            {
                INST.CallInstructions.Add(this);
            }
        }

        static public INST Wrap(Mono.Cecil.Cil.Instruction i, CFG.Vertex block, SequencePoint sp)
        {
            // Wrap instruction with semantics, def/use/kill properties.
            Mono.Cecil.Cil.OpCode op = i.OpCode;
            INST wrapped_inst;
            switch (op.Code)
            {
                case Mono.Cecil.Cil.Code.Add:
                    wrapped_inst = new i_add(i);
                    break;
                case Mono.Cecil.Cil.Code.Add_Ovf:
                    wrapped_inst = new i_add_ovf(i);
                    break;
                case Mono.Cecil.Cil.Code.Add_Ovf_Un:
                    wrapped_inst = new i_add_ovf_un(i);
                    break;
                case Mono.Cecil.Cil.Code.And:
                    wrapped_inst = new i_and(i);
                    break;
                case Mono.Cecil.Cil.Code.Arglist:
                    wrapped_inst = new i_arglist(i);
                    break;
                case Mono.Cecil.Cil.Code.Beq:
                    wrapped_inst = new i_beq(i);
                    break;
                case Mono.Cecil.Cil.Code.Beq_S:
                    wrapped_inst = new i_beq(i);
                    break;
                case Mono.Cecil.Cil.Code.Bge:
                    wrapped_inst = new i_bge(i);
                    break;
                case Mono.Cecil.Cil.Code.Bge_S:
                    wrapped_inst = new i_bge_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Bge_Un:
                    wrapped_inst = new i_bge_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Bge_Un_S:
                    wrapped_inst = new i_bge_un_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Bgt:
                    wrapped_inst = new i_bgt(i);
                    break;
                case Mono.Cecil.Cil.Code.Bgt_S:
                    wrapped_inst = new i_bgt_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Bgt_Un:
                    wrapped_inst = new i_bgt_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Bgt_Un_S:
                    wrapped_inst = new i_bgt_un_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ble:
                    wrapped_inst = new i_ble(i);
                    break;
                case Mono.Cecil.Cil.Code.Ble_S:
                    wrapped_inst = new i_ble_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ble_Un:
                    wrapped_inst = new i_ble_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Ble_Un_S:
                    wrapped_inst = new i_ble_un_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Blt:
                    wrapped_inst = new i_blt(i);
                    break;
                case Mono.Cecil.Cil.Code.Blt_S:
                    wrapped_inst = new i_blt_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Blt_Un:
                    wrapped_inst = new i_blt_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Blt_Un_S:
                    wrapped_inst = new i_blt_un_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Bne_Un:
                    wrapped_inst = new i_bne_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Bne_Un_S:
                    wrapped_inst = new i_bne_un_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Box:
                    wrapped_inst = new i_box(i);
                    break;
                case Mono.Cecil.Cil.Code.Br:
                    wrapped_inst = new i_br(i);
                    break;
                case Mono.Cecil.Cil.Code.Br_S:
                    wrapped_inst = new i_br_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Break:
                    wrapped_inst = new i_break(i);
                    break;
                case Mono.Cecil.Cil.Code.Brfalse:
                    wrapped_inst = new i_brfalse(i);
                    break;
                case Mono.Cecil.Cil.Code.Brfalse_S:
                    wrapped_inst = new i_brfalse_s(i);
                    break;
                // Missing brnull
                // Missing brzero
                case Mono.Cecil.Cil.Code.Brtrue:
                    wrapped_inst = new i_brtrue(i);
                    break;
                case Mono.Cecil.Cil.Code.Brtrue_S:
                    wrapped_inst = new i_brtrue_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Call:
                    wrapped_inst = new i_call(i);
                    break;
                case Mono.Cecil.Cil.Code.Calli:
                    wrapped_inst = new i_calli(i);
                    break;
                case Mono.Cecil.Cil.Code.Callvirt:
                    wrapped_inst = new i_callvirt(i);
                    break;
                case Mono.Cecil.Cil.Code.Castclass:
                    wrapped_inst = new i_castclass(i);
                    break;
                case Mono.Cecil.Cil.Code.Ceq:
                    wrapped_inst = new i_ceq(i);
                    break;
                case Mono.Cecil.Cil.Code.Cgt:
                    wrapped_inst = new i_cgt(i);
                    break;
                case Mono.Cecil.Cil.Code.Cgt_Un:
                    wrapped_inst = new i_cgt_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Ckfinite:
                    wrapped_inst = new i_ckfinite(i);
                    break;
                case Mono.Cecil.Cil.Code.Clt:
                    wrapped_inst = new i_clt(i);
                    break;
                case Mono.Cecil.Cil.Code.Clt_Un:
                    wrapped_inst = new i_clt_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Constrained:
                    wrapped_inst = new i_constrained(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_I1:
                    wrapped_inst = new i_conv_i1(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_I2:
                    wrapped_inst = new i_conv_i2(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_I4:
                    wrapped_inst = new i_conv_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_I8:
                    wrapped_inst = new i_conv_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_I:
                    wrapped_inst = new i_conv_i(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I1:
                    wrapped_inst = new i_conv_ovf_i1(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I1_Un:
                    wrapped_inst = new i_conv_ovf_i1_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I2:
                    wrapped_inst = new i_conv_ovf_i2(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I2_Un:
                    wrapped_inst = new i_conv_ovf_i2_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I4:
                    wrapped_inst = new i_conv_ovf_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I4_Un:
                    wrapped_inst = new i_conv_ovf_i4_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I8:
                    wrapped_inst = new i_conv_ovf_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I8_Un:
                    wrapped_inst = new i_conv_ovf_i8_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I:
                    wrapped_inst = new i_conv_ovf_i(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_I_Un:
                    wrapped_inst = new i_conv_ovf_i_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U1:
                    wrapped_inst = new i_conv_ovf_u1(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U1_Un:
                    wrapped_inst = new i_conv_ovf_u1_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U2:
                    wrapped_inst = new i_conv_ovf_u2(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U2_Un:
                    wrapped_inst = new i_conv_ovf_u2_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U4:
                    wrapped_inst = new i_conv_ovf_u4(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U4_Un:
                    wrapped_inst = new i_conv_ovf_u4_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U8:
                    wrapped_inst = new i_conv_ovf_u8(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U8_Un:
                    wrapped_inst = new i_conv_ovf_u8_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U:
                    wrapped_inst = new i_conv_ovf_u(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_Ovf_U_Un:
                    wrapped_inst = new i_conv_ovf_u_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_R4:
                    wrapped_inst = new i_conv_r4(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_R8:
                    wrapped_inst = new i_conv_r8(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_R_Un:
                    wrapped_inst = new i_conv_r_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_U1:
                    wrapped_inst = new i_conv_u1(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_U2:
                    wrapped_inst = new i_conv_u2(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_U4:
                    wrapped_inst = new i_conv_u4(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_U8:
                    wrapped_inst = new i_conv_u8(i);
                    break;
                case Mono.Cecil.Cil.Code.Conv_U:
                    wrapped_inst = new i_conv_u(i);
                    break;
                case Mono.Cecil.Cil.Code.Cpblk:
                    wrapped_inst = new i_cpblk(i);
                    break;
                case Mono.Cecil.Cil.Code.Cpobj:
                    wrapped_inst = new i_cpobj(i);
                    break;
                case Mono.Cecil.Cil.Code.Div:
                    wrapped_inst = new i_div(i);
                    break;
                case Mono.Cecil.Cil.Code.Div_Un:
                    wrapped_inst = new i_div_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Dup:
                    wrapped_inst = new i_dup(i);
                    break;
                case Mono.Cecil.Cil.Code.Endfilter:
                    wrapped_inst = new i_endfilter(i);
                    break;
                case Mono.Cecil.Cil.Code.Endfinally:
                    wrapped_inst = new i_endfinally(i);
                    break;
                case Mono.Cecil.Cil.Code.Initblk:
                    wrapped_inst = new i_initblk(i);
                    break;
                case Mono.Cecil.Cil.Code.Initobj:
                    wrapped_inst = new i_initobj(i);
                    break;
                case Mono.Cecil.Cil.Code.Isinst:
                    wrapped_inst = new i_isinst(i);
                    break;
                case Mono.Cecil.Cil.Code.Jmp:
                    wrapped_inst = new i_jmp(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarg:
                    wrapped_inst = new i_ldarg(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarg_0:
                    wrapped_inst = new i_ldarg_0(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarg_1:
                    wrapped_inst = new i_ldarg_1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarg_2:
                    wrapped_inst = new i_ldarg_2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarg_3:
                    wrapped_inst = new i_ldarg_3(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarg_S:
                    wrapped_inst = new i_ldarg_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarga:
                    wrapped_inst = new i_ldarga(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldarga_S:
                    wrapped_inst = new i_ldarga_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4:
                    wrapped_inst = new i_ldc_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_0:
                    wrapped_inst = new i_ldc_i4_0(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_1:
                    wrapped_inst = new i_ldc_i4_1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_2:
                    wrapped_inst = new i_ldc_i4_2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_3:
                    wrapped_inst = new i_ldc_i4_3(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_4:
                    wrapped_inst = new i_ldc_i4_4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_5:
                    wrapped_inst = new i_ldc_i4_5(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_6:
                    wrapped_inst = new i_ldc_i4_6(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_7:
                    wrapped_inst = new i_ldc_i4_7(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_8:
                    wrapped_inst = new i_ldc_i4_8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_M1:
                    wrapped_inst = new i_ldc_i4_m1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I4_S:
                    wrapped_inst = new i_ldc_i4_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_I8:
                    wrapped_inst = new i_ldc_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_R4:
                    wrapped_inst = new i_ldc_r4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldc_R8:
                    wrapped_inst = new i_ldc_r8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_Any:
                    wrapped_inst = new i_ldelem_any(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_I1:
                    wrapped_inst = new i_ldelem_i1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_I2:
                    wrapped_inst = new i_ldelem_i2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_I4:
                    wrapped_inst = new i_ldelem_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_I8:
                    wrapped_inst = new i_ldelem_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_I:
                    wrapped_inst = new i_ldelem_i(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_R4:
                    wrapped_inst = new i_ldelem_r4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_R8:
                    wrapped_inst = new i_ldelem_r8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_Ref:
                    wrapped_inst = new i_ldelem_ref(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_U1:
                    wrapped_inst = new i_ldelem_u1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_U2:
                    wrapped_inst = new i_ldelem_u2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelem_U4:
                    wrapped_inst = new i_ldelem_u4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldelema:
                    wrapped_inst = new i_ldelema(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldfld:
                    wrapped_inst = new i_ldfld(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldflda:
                    wrapped_inst = new i_ldflda(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldftn:
                    wrapped_inst = new i_ldftn(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_I1:
                    wrapped_inst = new i_ldind_i1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_I2:
                    wrapped_inst = new i_ldind_i2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_I4:
                    wrapped_inst = new i_ldind_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_I8:
                    wrapped_inst = new i_ldind_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_I:
                    wrapped_inst = new i_ldind_i(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_R4:
                    wrapped_inst = new i_ldind_r4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_R8:
                    wrapped_inst = new i_ldind_r8(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_Ref:
                    wrapped_inst = new i_ldind_ref(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_U1:
                    wrapped_inst = new i_ldind_u1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_U2:
                    wrapped_inst = new i_ldind_u2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldind_U4:
                    wrapped_inst = new i_ldind_u4(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldlen:
                    wrapped_inst = new i_ldlen(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloc:
                    wrapped_inst = new i_ldloc(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloc_0:
                    wrapped_inst = new i_ldloc_0(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloc_1:
                    wrapped_inst = new i_ldloc_1(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloc_2:
                    wrapped_inst = new i_ldloc_2(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloc_3:
                    wrapped_inst = new i_ldloc_3(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloc_S:
                    wrapped_inst = new i_ldloc_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloca:
                    wrapped_inst = new i_ldloca(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldloca_S:
                    wrapped_inst = new i_ldloca_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldnull:
                    wrapped_inst = new i_ldnull(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldobj:
                    wrapped_inst = new i_ldobj(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldsfld:
                    wrapped_inst = new i_ldsfld(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldsflda:
                    wrapped_inst = new i_ldsflda(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldstr:
                    wrapped_inst = new i_ldstr(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldtoken:
                    wrapped_inst = new i_ldtoken(i);
                    break;
                case Mono.Cecil.Cil.Code.Ldvirtftn:
                    wrapped_inst = new i_ldvirtftn(i);
                    break;
                case Mono.Cecil.Cil.Code.Leave:
                    wrapped_inst = new i_leave(i);
                    break;
                case Mono.Cecil.Cil.Code.Leave_S:
                    wrapped_inst = new i_leave_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Localloc:
                    wrapped_inst = new i_localloc(i);
                    break;
                case Mono.Cecil.Cil.Code.Mkrefany:
                    wrapped_inst = new i_mkrefany(i);
                    break;
                case Mono.Cecil.Cil.Code.Mul:
                    wrapped_inst = new i_mul(i);
                    break;
                case Mono.Cecil.Cil.Code.Mul_Ovf:
                    wrapped_inst = new i_mul_ovf(i);
                    break;
                case Mono.Cecil.Cil.Code.Mul_Ovf_Un:
                    wrapped_inst = new i_mul_ovf_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Neg:
                    wrapped_inst = new i_neg(i);
                    break;
                case Mono.Cecil.Cil.Code.Newarr:
                    wrapped_inst = new i_newarr(i);
                    break;
                case Mono.Cecil.Cil.Code.Newobj:
                    wrapped_inst = new i_newobj(i);
                    break;
                case Mono.Cecil.Cil.Code.No:
                    wrapped_inst = new i_no(i);
                    break;
                case Mono.Cecil.Cil.Code.Nop:
                    wrapped_inst = new i_nop(i);
                    break;
                case Mono.Cecil.Cil.Code.Not:
                    wrapped_inst = new i_not(i);
                    break;
                case Mono.Cecil.Cil.Code.Or:
                    wrapped_inst = new i_or(i);
                    break;
                case Mono.Cecil.Cil.Code.Pop:
                    wrapped_inst = new i_pop(i);
                    break;
                case Mono.Cecil.Cil.Code.Readonly:
                    wrapped_inst = new i_readonly(i);
                    break;
                case Mono.Cecil.Cil.Code.Refanytype:
                    wrapped_inst = new i_refanytype(i);
                    break;
                case Mono.Cecil.Cil.Code.Refanyval:
                    wrapped_inst = new i_refanyval(i);
                    break;
                case Mono.Cecil.Cil.Code.Rem:
                    wrapped_inst = new i_rem(i);
                    break;
                case Mono.Cecil.Cil.Code.Rem_Un:
                    wrapped_inst = new i_rem_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Ret:
                    wrapped_inst = new i_ret(i);
                    break;
                case Mono.Cecil.Cil.Code.Rethrow:
                    wrapped_inst = new i_rethrow(i);
                    break;
                case Mono.Cecil.Cil.Code.Shl:
                    wrapped_inst = new i_shl(i);
                    break;
                case Mono.Cecil.Cil.Code.Shr:
                    wrapped_inst = new i_shr(i);
                    break;
                case Mono.Cecil.Cil.Code.Shr_Un:
                    wrapped_inst = new i_shr_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Sizeof:
                    wrapped_inst = new i_sizeof(i);
                    break;
                case Mono.Cecil.Cil.Code.Starg:
                    wrapped_inst = new i_starg(i);
                    break;
                case Mono.Cecil.Cil.Code.Starg_S:
                    wrapped_inst = new i_starg_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_Any:
                    wrapped_inst = new i_stelem_any(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_I1:
                    wrapped_inst = new i_stelem_i1(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_I2:
                    wrapped_inst = new i_stelem_i2(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_I4:
                    wrapped_inst = new i_stelem_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_I8:
                    wrapped_inst = new i_stelem_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_I:
                    wrapped_inst = new i_stelem_i(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_R4:
                    wrapped_inst = new i_stelem_r4(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_R8:
                    wrapped_inst = new i_stelem_r8(i);
                    break;
                case Mono.Cecil.Cil.Code.Stelem_Ref:
                    wrapped_inst = new i_stelem_ref(i);
                    break;
                case Mono.Cecil.Cil.Code.Stfld:
                    wrapped_inst = new i_stfld(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_I1:
                    wrapped_inst = new i_stind_i1(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_I2:
                    wrapped_inst = new i_stind_i2(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_I4:
                    wrapped_inst = new i_stind_i4(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_I8:
                    wrapped_inst = new i_stind_i8(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_I:
                    wrapped_inst = new i_stind_i(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_R4:
                    wrapped_inst = new i_stind_r4(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_R8:
                    wrapped_inst = new i_stind_r8(i);
                    break;
                case Mono.Cecil.Cil.Code.Stind_Ref:
                    wrapped_inst = new i_stind_ref(i);
                    break;
                case Mono.Cecil.Cil.Code.Stloc:
                    wrapped_inst = new i_stloc(i);
                    break;
                case Mono.Cecil.Cil.Code.Stloc_0:
                    wrapped_inst = new i_stloc_0(i);
                    break;
                case Mono.Cecil.Cil.Code.Stloc_1:
                    wrapped_inst = new i_stloc_1(i);
                    break;
                case Mono.Cecil.Cil.Code.Stloc_2:
                    wrapped_inst = new i_stloc_2(i);
                    break;
                case Mono.Cecil.Cil.Code.Stloc_3:
                    wrapped_inst = new i_stloc_3(i);
                    break;
                case Mono.Cecil.Cil.Code.Stloc_S:
                    wrapped_inst = new i_stloc_s(i);
                    break;
                case Mono.Cecil.Cil.Code.Stobj:
                    wrapped_inst = new i_stobj(i);
                    break;
                case Mono.Cecil.Cil.Code.Stsfld:
                    wrapped_inst = new i_stsfld(i);
                    break;
                case Mono.Cecil.Cil.Code.Sub:
                    wrapped_inst = new i_sub(i);
                    break;
                case Mono.Cecil.Cil.Code.Sub_Ovf:
                    wrapped_inst = new i_sub_ovf(i);
                    break;
                case Mono.Cecil.Cil.Code.Sub_Ovf_Un:
                    wrapped_inst = new i_sub_ovf_un(i);
                    break;
                case Mono.Cecil.Cil.Code.Switch:
                    wrapped_inst = new i_switch(i);
                    break;
                case Mono.Cecil.Cil.Code.Tail:
                    wrapped_inst = new i_tail(i);
                    break;
                case Mono.Cecil.Cil.Code.Throw:
                    wrapped_inst = new i_throw(i);
                    break;
                case Mono.Cecil.Cil.Code.Unaligned:
                    wrapped_inst = new i_unaligned(i);
                    break;
                case Mono.Cecil.Cil.Code.Unbox:
                    wrapped_inst = new i_unbox(i);
                    break;
                case Mono.Cecil.Cil.Code.Unbox_Any:
                    wrapped_inst = new i_unbox_any(i);
                    break;
                case Mono.Cecil.Cil.Code.Volatile:
                    wrapped_inst = new i_volatile(i);
                    break;
                case Mono.Cecil.Cil.Code.Xor:
                    wrapped_inst = new i_xor(i);
                    break;
                default:
                    throw new Exception("Unknown instruction type " + i);
            }
            wrapped_inst.SeqPoint = sp;
            return wrapped_inst;
        }

        public void Replace(Instruction inst)
        {
            this.Instruction = inst;
        }
    }

    public class BinaryOpInst : INST
    {
        public BinaryOpInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var lhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(lhs);

            var result = lhs;

            state._stack.Push(result);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
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


        TYPE binaryOpType(System.Type Opcode, TYPE Type1, TYPE Type2)
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
                if (Size1 > Size2)
                {
                    return Type1;
                }
                if (Size2 > Size1)
                {
                    return Type2;
                }
            }
            else
            {
                bool Type1IsUnmanagedPointer = false;
                bool Type2IsUnmanagedPointer = false;
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
                        return new TYPE(TYPE.getIntNTy(LLVM.GetModuleContext(RUNTIME.global_llvm_module),
                            TargetPointerSizeInBits));
                    }
                }
                else if (Type1.isPointerTy())
                {
                    if (IsSub && Type2.isPointerTy())
                    {
                        // The difference of two managed pointers is a native int.
                        return new TYPE(TYPE.getIntNTy(LLVM.GetModuleContext(RUNTIME.global_llvm_module),
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
        VALUE genPointerAdd(VALUE Arg1, VALUE Arg2)
        {
            // Assume 1 is base and 2 is offset
            VALUE BasePtr = Arg1;
            VALUE Offset = Arg2;

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
            TYPE OffsetTy = Offset.T;
            if (!OffsetTy.isIntegerTy())
            {
                return null;
            }

            // Build an LLVM GEP for the resulting address.
            // For now we "flatten" to byte offsets.

            TYPE CharPtrTy = new TYPE(
                TYPE.getInt8PtrTy(
                LLVM.GetModuleContext(RUNTIME.global_llvm_module),
                BasePtr.T.getPointerAddressSpace()));
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(CharPtrTy);

            VALUE BasePtrCast = new VALUE(LLVM.BuildBitCast(Builder, BasePtr.V, CharPtrTy.IntermediateType, "i"+instruction_id++));
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(BasePtrCast);

            VALUE ResultPtr = new VALUE(LLVM.BuildInBoundsGEP(Builder, BasePtrCast.V, new ValueRef[] {Offset.V}, ""));
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(ResultPtr);

            return ResultPtr;
        }

        // Handle pointer - int by emitting a flattened LLVM GEP.
        VALUE genPointerSub(VALUE Arg1, VALUE Arg2)
        {

            // Assume 1 is base and 2 is offset
            VALUE BasePtr = Arg1;
            VALUE Offset = Arg2;

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
            TYPE OffsetTy = Offset.T;
            if (!OffsetTy.isIntegerTy())
            {
                return null;
            }

            // Build an LLVM GEP for the resulting address.
            // For now we "flatten" to byte offsets.
            TYPE CharPtrTy = new TYPE(TYPE.getInt8PtrTy(
                LLVM.GetModuleContext(RUNTIME.global_llvm_module), BasePtr.T.getPointerAddressSpace()));
            VALUE BasePtrCast = new VALUE(LLVM.BuildBitCast(Builder, BasePtr.V, CharPtrTy.IntermediateType, "i" + instruction_id++));
            VALUE NegOffset = new VALUE(LLVM.BuildNeg(Builder, Offset.V, "i" + instruction_id++));
            VALUE ResultPtr = new VALUE(LLVM.BuildGEP(Builder, BasePtrCast.V, new ValueRef[] { NegOffset.V }, "i" + instruction_id++));
            return ResultPtr;
        }

        // This method only handles basic arithmetic conversions for use in
        // binary operations.
        public VALUE convert(TYPE Ty, VALUE Node, bool SourceIsSigned)
        {
            TYPE SourceTy = Node.T;
            VALUE Result = null;

            if (Ty == SourceTy)
            {
                Result = Node;
            }
            else if (SourceTy.isIntegerTy() && Ty.isIntegerTy())
            {
                Result = new VALUE(LLVM.BuildIntCast(Builder, Node.V, Ty.IntermediateType, "i" + instruction_id++));//SourceIsSigned);
            }
            else if (SourceTy.isFloatingPointTy() && Ty.isFloatingPointTy())
            {
                Result = new VALUE(LLVM.BuildFPCast(Builder, Node.V, Ty.IntermediateType, "i" + instruction_id++));
            }
            else if (SourceTy.isPointerTy() && Ty.isIntegerTy())
            {
                Result = new VALUE(LLVM.BuildPtrToInt(Builder, Node.V, Ty.IntermediateType, "i" + instruction_id++));
            }
            else
            {
                Debug.Assert(false);
            }

            return Result;
        }

        VALUE binaryOp(System.Type Opcode, VALUE Arg1, VALUE Arg2)
        {
            TYPE Type1 = Arg1.T;
            TYPE Type2 = Arg2.T;
            TYPE ResultType = binaryOpType(Opcode, Type1, Type2);
            TYPE ArithType = ResultType;

            // If the result is a pointer, see if we have simple
            // pointer + int op...
            if (ResultType.isPointerTy())
            {
                if (Opcode == typeof(i_add))
                {
                    VALUE PtrAdd = genPointerAdd(Arg1, Arg2);
                    if (PtrAdd != null)
                    {
                        return PtrAdd;
                    }
                }
                else if (Opcode == typeof(i_add_ovf_un))
                {
                    VALUE PtrSub = genPointerSub(Arg1, Arg2);
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
                    ArithType = new TYPE(TYPE.getIntNTy(LLVM.GetModuleContext(RUNTIME.global_llvm_module), TargetPointerSizeInBits));
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

            VALUE Result;
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
                        VALUE IsZero = new VALUE(LLVM.BuildIsNull(Builder, Arg2.V, "i" + instruction_id++));
                        //genConditionalThrow(IsZero, CORINFO_HELP_THROWDIVZERO, "ThrowDivideByZero");
                    }
                    else
                    {
                        // This configuration isn't really supported.  To support it we'd
                        // need to annotate the divide we're about to generate as possibly
                        // throwing an exception (that would be raised from a machine trap).
                    }
                }

                Result = new VALUE(LLVM.BuildBinOp(Builder, OpI.Opcode, Arg1.V, Arg2.V, "i"+instruction_id++));
            }

            if (ResultType != ArithType)
            {
                Debug.Assert(ResultType.isPointerTy());
                Debug.Assert(ArithType.isIntegerTy());

                Result = new VALUE(LLVM.BuildIntToPtr(Builder, Result.V, ResultType.IntermediateType, "i" + instruction_id++));
            }
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(Result);

            return Result;
        }

        public bool UseExplicitZeroDivideChecks { get; set; }
    }


    public class ConvertCallInst : INST
    {
        public ConvertCallInst(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            INST new_inst = this;
            object method = this.Operand;
            if (method as Mono.Cecil.MethodReference == null) throw new Exception();
            Mono.Cecil.MethodReference orig_mr = method as Mono.Cecil.MethodReference;
            var mr = orig_mr;
            bool has_this = false;
            if (mr.HasThis) has_this = true;
            if (OpCode.Code == Code.Callvirt) has_this = true;
            bool is_explicit_this = mr.ExplicitThis;
            int xargs = (has_this && !is_explicit_this ? 1 : 0) + mr.Parameters.Count;
            List<TypeReference> args = new List<TypeReference>();
            for (int k = 0; k < xargs; ++k)
            {
                var v = state._stack.Pop();
                args.Insert(0, v);
            }
            var args_array = args.ToArray();
            mr = orig_mr.SubstituteMethod(this.Block._original_method_reference.DeclaringType, args_array);
            if (mr == null) return; // Can't do anything with this.
            if (mr.ReturnType.FullName != "System.Void")
            {
                state._stack.Push(mr.ReturnType);
            }
            IMPORTER.Singleton().Add(mr);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            object method = this.Operand;
            if (method as Mono.Cecil.MethodReference == null) throw new Exception();
            Mono.Cecil.MethodReference orig_mr = method as Mono.Cecil.MethodReference;
            var mr = orig_mr.FixGenericMethods(this.Block._original_method_reference);

            // Two general cases here: (1) Calling a method that is in CIL. (2) calling
            // a BCL method that has no CIL body.

            // Find bb entry.
            CFG.Vertex entry_corresponding_to_method_called = this.Block._graph.Vertices.Where(node
                =>
            {
                var g = this.Block._graph;
                CFG.Vertex v = node;
                JITER c = JITER.Singleton;
                if (v.IsEntry && JITER.MethodName(v._original_method_reference) == mr.FullName)
                    return true;
                else return false;
            }).ToList().FirstOrDefault();

            if (entry_corresponding_to_method_called == null)
            {
                // If there is no entry block discovered, so this function is probably to a BCL for GPU method.
                var name = mr.Name;
                var full_name = mr.FullName;
                // For now, look for qualified name not including parameters.
                Regex regex = new Regex(@"^[^\s]+\s+(?<name>[^\(]+).+$");
                Match m = regex.Match(full_name);
                if (!m.Success)
                    throw new Exception();
                var mangled_name = m.Groups["name"].Value;
                mangled_name = mangled_name.Replace("::", "_");
                mangled_name = mangled_name.Replace(".", "_");

                BuilderRef bu = this.Builder;

                // Find the specific function called in BCL.
                var xx = RUNTIME._bcl_runtime_csharp_internal_to_valueref.Where(t => t.Key.Contains(mangled_name) || mangled_name.Contains(t.Key));
                var first_kv_pair = xx.FirstOrDefault();
                if (first_kv_pair.Key == null)
                {
                    // No direct entry in the BCL--we don't have a direct implementation.
                    // This can happen with arrays, e.g.,
                    // "System.Void System.Int32[0...,0...]::Set(System.Int32,System.Int32,System.Int32)"
                    TypeReference declaring_type = mr.DeclaringType;
                    if (declaring_type != null && declaring_type.IsArray)
                    {
                        // Handle array calls with special code.
                        var the_array_type = declaring_type as Mono.Cecil.ArrayType;
                        TypeReference element_type = declaring_type.GetElementType();
                        Collection<ArrayDimension> dimensions = the_array_type.Dimensions;
                        var count = dimensions.Count;

                        if (mr.Name == "Set")
                        {
                            // Make "set" call
                            unsafe
                            {
                                ValueRef[] args = new ValueRef[1 // this
                                                               + 1 // indices
                                                               + 1 // val
                                ];

                                // Allocate space on stack for one value to be passed to function call.
                                var val_type = element_type.ToTypeRef();
                                var val_buffer = LLVM.BuildAlloca(Builder, val_type, "i" + instruction_id++);
                                LLVM.SetAlignment(val_buffer, 64);
                                LLVM.BuildStore(Builder, state._stack.Pop().V, val_buffer);

                                // Assign value arg for function call to set.
                                args[2] = LLVM.BuildPtrToInt(Builder, val_buffer, LLVM.Int64Type(), "i" + instruction_id++);

                                // Allocate space on stack for "count" indices, 64 bits each.
                                var ind_buffer = LLVM.BuildAlloca(Builder, LLVM.ArrayType(LLVM.Int64Type(), (uint)count), "i" + instruction_id++);
                                LLVM.SetAlignment(ind_buffer, 64);
                                var base_of_indices = LLVM.BuildPointerCast(Builder, ind_buffer, LLVM.PointerType(LLVM.Int64Type(), 0), "i" + instruction_id++);

                                // Place each value in indices array.
                                for (int i = count - 1; i >= 0; i--)
                                {
                                    VALUE index = state._stack.Pop();
                                    if (Campy.Utils.Options.IsOn("jit_trace"))
                                        System.Console.WriteLine(index);
                                    ValueRef[] id = new ValueRef[1] { LLVM.ConstInt(LLVM.Int64Type(), (ulong)i, true) };
                                    var add = LLVM.BuildInBoundsGEP(Builder, base_of_indices, id, "i" + instruction_id++);
                                    var cast = LLVM.BuildIntCast(Builder, index.V, LLVM.Int64Type(), "i" + instruction_id++);
                                    ValueRef store = LLVM.BuildStore(Builder, cast, add);
                                    if (Campy.Utils.Options.IsOn("jit_trace"))
                                        System.Console.WriteLine(new VALUE(store));
                                }

                                // Assign indices arg for function call to set.
                                args[1] = LLVM.BuildPtrToInt(Builder, ind_buffer, LLVM.Int64Type(), "i" + instruction_id++);

                                // Assign "this" array to arg for function call to set.
                                VALUE p = state._stack.Pop();
                                args[0] = LLVM.BuildPtrToInt(Builder, p.V, LLVM.Int64Type(), "i" + instruction_id++);

                                string nme = "_Z31SystemArray_StoreElementIndicesPhPyS0_";
                                var list2 = RUNTIME.PtxFunctions.ToList();
                                var f = list2.Where(t => t._mangled_name == nme).First();
                                ValueRef fv = f._valueref;
                                var call = LLVM.BuildCall(Builder, fv, args, "");
                                if (Campy.Utils.Options.IsOn("jit_trace"))
                                    System.Console.WriteLine(call.ToString());
                            }
                            return;
                        }
                        else if (mr.Name == "Get")
                        {
                            unsafe
                            {
                                ValueRef[] args = new ValueRef[1 // this
                                                               + 1 // indices
                                                               + 1 // val
                                ];

                                // Allocate space on stack for one value to be received from function call.
                                var val_type = element_type.ToTypeRef();
                                var val_buffer = LLVM.BuildAlloca(Builder, val_type, "i" + instruction_id++);
                                LLVM.SetAlignment(val_buffer, 64);

                                // Assign value arg for function call to set.
                                args[2] = LLVM.BuildPtrToInt(Builder, val_buffer, LLVM.Int64Type(), "i" + instruction_id++);

                                // Allocate space on stack for "count" indices, 64 bits each.
                                var ind_buffer = LLVM.BuildAlloca(Builder,
                                    LLVM.ArrayType(
                                    LLVM.Int64Type(),
                                    (uint)count), "i" + instruction_id++);
                                LLVM.SetAlignment(ind_buffer, 64);
                                var base_of_indices = LLVM.BuildPointerCast(Builder, ind_buffer, LLVM.PointerType(LLVM.Int64Type(), 0), "i" + instruction_id++);
                                for (int i = count - 1; i >= 0; i--)
                                {
                                    VALUE index = state._stack.Pop();
                                    if (Campy.Utils.Options.IsOn("jit_trace"))
                                        System.Console.WriteLine(index);
                                    ValueRef[] id = new ValueRef[1]
                                        {LLVM.ConstInt(LLVM.Int64Type(), (ulong) i, true)};
                                    var add = LLVM.BuildInBoundsGEP(Builder, base_of_indices, id, "i" + instruction_id++);
                                    var cast = LLVM.BuildIntCast(Builder, index.V, LLVM.Int64Type(), "i" + instruction_id++);
                                    ValueRef store = LLVM.BuildStore(Builder, cast, add);
                                    if (Campy.Utils.Options.IsOn("jit_trace"))
                                        System.Console.WriteLine(new VALUE(store));
                                }

                                // Assign indices arg for function call to set.
                                args[1] = LLVM.BuildPtrToInt(Builder, ind_buffer, LLVM.Int64Type(), "i" + instruction_id++);

                                // Assign "this" array to arg for function call to set.
                                VALUE p = state._stack.Pop();
                                args[0] = LLVM.BuildPtrToInt(Builder, p.V, LLVM.Int64Type(), "i" + instruction_id++);

                                string nme = "_Z30SystemArray_LoadElementIndicesPhPyS0_";
                                var list = RUNTIME.BclNativeMethods.ToList();
                                var list2 = RUNTIME.PtxFunctions.ToList();
                                var f = list2.Where(t => t._mangled_name == nme).First();
                                ValueRef fv = f._valueref;
                                var call = LLVM.BuildCall(Builder, fv, args, "");
                                if (Campy.Utils.Options.IsOn("jit_trace"))
                                    System.Console.WriteLine(call.ToString());
                                var load = LLVM.BuildLoad(Builder, val_buffer, "i" + instruction_id++);
                                var result = new VALUE(load);
                                state._stack.Push(result);
                                if (Campy.Utils.Options.IsOn("jit_trace"))
                                    System.Console.WriteLine(result);
                            }
                            return;
                        }
                    }
                    throw new Exception("Unknown, internal, function for which there is no body and no C/C++ code. "
                                        + mangled_name
                                        + " "
                                        + full_name);
                }
                else
                {

                    Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                    Mono.Cecil.TypeReference tr = rt.ReturnType;
                    var ret = tr.FullName != "System.Void";
                    var HasScalarReturnValue = ret && !tr.IsStruct() && !tr.IsReferenceType();
                    var HasStructReturnValue = ret && tr.IsStruct() && !tr.IsReferenceType();
                    bool has_this = false;
                    if (mr.HasThis) has_this = true;

                    if (OpCode.Code == Code.Callvirt) has_this = true;
                    bool is_explicit_this = mr.ExplicitThis;
                    int xargs = (has_this && !is_explicit_this ? 1 : 0) + mr.Parameters.Count;

                    var NumberOfArguments = mr.Parameters.Count
                                            + (has_this ? 1 : 0)
                                            + (HasStructReturnValue ? 1 : 0);
                    int locals = 0;
                    var NumberOfLocals = locals;
                    int xret = (HasScalarReturnValue || HasStructReturnValue) ? 1 : 0;

                    ValueRef fv = first_kv_pair.Value;
                    var t_fun = LLVM.TypeOf(fv);
                    var t_fun_con = LLVM.GetTypeContext(t_fun);
                    var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);

                    RUNTIME.BclNativeMethod mat = null;
                    foreach (RUNTIME.BclNativeMethod ci in RUNTIME.BclNativeMethods)
                    {
                        if (ci._full_name == full_name)
                        {
                            mat = ci;
                            break;
                        }
                    }

                    {
                        ValueRef[] args = new ValueRef[3];

                        // Set up "this".
                        ValueRef nul = LLVM.ConstPointerNull(LLVM.PointerType(LLVM.VoidType(), 0));
                        VALUE t = new VALUE(nul);

                        // Pop all parameters and stuff into params buffer. Note, "this" and
                        // "return" are separate parameters in GPU BCL runtime C-functions,
                        // unfortunately, reminates of the DNA runtime I decided to use.
                        var entry = this.Block.Entry.LlvmInfo.BasicBlock;
                        var beginning = LLVM.GetFirstInstruction(entry);
                        //LLVM.PositionBuilderBefore(Builder, beginning);
                        var parameter_type = LLVM.ArrayType(LLVM.Int64Type(), (uint)mr.Parameters.Count);
                        var param_buffer = LLVM.BuildAlloca(Builder, parameter_type, "i" + instruction_id++);
                        LLVM.SetAlignment(param_buffer, 64);
                        //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);
                        var base_of_parameters = LLVM.BuildPointerCast(Builder, param_buffer,
                            LLVM.PointerType(LLVM.Int64Type(), 0), "i" + instruction_id++);
                        for (int i = mr.Parameters.Count - 1; i >= 0; i--)
                        {
                            VALUE p = state._stack.Pop();
                            ValueRef[] index = new ValueRef[1] { LLVM.ConstInt(LLVM.Int32Type(), (ulong)i, true) };
                            var gep = LLVM.BuildGEP(Builder, param_buffer, index, "i" + instruction_id++);
                            var add = LLVM.BuildInBoundsGEP(Builder, base_of_parameters, index, "i" + instruction_id++);
                            ValueRef v = LLVM.BuildPointerCast(Builder, add, LLVM.PointerType(LLVM.TypeOf(p.V), 0),
                                "i" + instruction_id++);
                            ValueRef store = LLVM.BuildStore(Builder, p.V, v);
                            if (Campy.Utils.Options.IsOn("jit_trace"))
                                System.Console.WriteLine(new VALUE(store));
                        }

                        if (has_this)
                        {
                            t = state._stack.Pop();
                        }

                        // Set up return. For now, always allocate buffer.
                        // Note function return is type of third parameter.
                        var return_type = mat._returnType.ToTypeRef();
                        if (mat._returnType.FullName == "System.Void")
                            return_type = typeof(IntPtr).ToMonoTypeReference().ToTypeRef();
                        var return_buffer = LLVM.BuildAlloca(Builder, return_type, "i" + instruction_id++);
                        LLVM.SetAlignment(return_buffer, 64);
                        //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);

                        // Set up call.
                        var pt = LLVM.BuildPtrToInt(Builder, t.V, LLVM.Int64Type(), "i" + instruction_id++);
                        var pp = LLVM.BuildPtrToInt(Builder, param_buffer, LLVM.Int64Type(), "i" + instruction_id++);
                        var pr = LLVM.BuildPtrToInt(Builder, return_buffer, LLVM.Int64Type(), "i" + instruction_id++);

                        //var pt = LLVM.BuildPointerCast(Builder, t.V,
                        //    LLVM.PointerType(LLVM.VoidType(), 0), "i" + instruction_id++);
                        //var pp = LLVM.BuildPointerCast(Builder, param_buffer,
                        //    LLVM.PointerType(LLVM.VoidType(), 0), "i" + instruction_id++);
                        //var pr = LLVM.BuildPointerCast(Builder, return_buffer,
                        //    LLVM.PointerType(LLVM.VoidType(), 0), "i" + instruction_id++);

                        args[0] = pt;
                        args[1] = pp;
                        args[2] = pr;

                        var call = LLVM.BuildCall(Builder, fv, args, "");

                        if (ret)
                        {
                            var load = LLVM.BuildLoad(Builder, return_buffer, "i" + instruction_id++);
                            state._stack.Push(new VALUE(load));
                        }

                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(call.ToString());
                    }
                }
            }
            else
            {
                // There is an entry block discovered for this call.
                // For return, we need to leave something on the damn stack regardless of how it's implmented.
                int xret = (entry_corresponding_to_method_called.HasScalarReturnValue || entry_corresponding_to_method_called.HasStructReturnValue) ? 1 : 0;
                int xargs = entry_corresponding_to_method_called.StackNumberOfArguments;

                var name = JITER.MethodName(mr);
                BuilderRef bu = this.Builder;
                ValueRef fv = entry_corresponding_to_method_called.LlvmInfo.MethodValueRef;
                var t_fun = LLVM.TypeOf(fv);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");
                //LLVM.VerifyFunction(fv, VerifierFailureAction.PrintMessageAction);

                // Set up args, type casting if required.
                ValueRef[] args = new ValueRef[xargs];
                if (entry_corresponding_to_method_called.HasStructReturnValue)
                {
                    // Special case for call with struct return. The return value is actually another
                    // parameter on the stack, which must be allocated.
                    // Further, the return for LLVM code is actually void.
                    ValueRef ret_par = LLVM.GetParam(fv, (uint)0);
                    var alloc_type = LLVM.GetElementType(LLVM.TypeOf(ret_par));

                    var entry = this.Block.Entry.LlvmInfo.BasicBlock;
                    var beginning = LLVM.GetFirstInstruction(entry);
                    //LLVM.PositionBuilderBefore(Builder, beginning);

                    var new_obj =
                        LLVM.BuildAlloca(Builder, alloc_type,
                            "i" + instruction_id++); // Allocates struct on stack, but returns a pointer to struct.
                    //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(new_obj));
                    args[0] = new_obj;
                    for (int k = xargs - 1; k >= 1; --k)
                    {
                        VALUE v = state._stack.Pop();
                        ValueRef par = LLVM.GetParam(fv, (uint)k);
                        ValueRef value = v.V;
                        if (LLVM.TypeOf(value) != LLVM.TypeOf(par))
                        {
                            if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.StructTypeKind
                                && LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildLoad(Builder, value, "i" + instruction_id++);
                            }
                            else if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildPointerCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                            else
                            {
                                value = LLVM.BuildBitCast(Builder, value, LLVM.TypeOf(par), "");
                            }
                        }
                        args[k] = value;
                    }
                    var call = LLVM.BuildCall(Builder, fv, args, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(call.ToString());
                    // Push the return on the stack. Note, it's not the call, but the new obj dereferenced.
                    var dereferenced_return_value = LLVM.BuildLoad(Builder, new_obj, "i" + instruction_id++);
                    state._stack.Push(new VALUE(dereferenced_return_value));
                }
                else if (entry_corresponding_to_method_called.HasScalarReturnValue)
                {
                    for (int k = xargs - 1; k >= 0; --k)
                    {
                        VALUE v = state._stack.Pop();
                        ValueRef par = LLVM.GetParam(fv, (uint)k);
                        ValueRef value = v.V;
                        if (LLVM.TypeOf(value) != LLVM.TypeOf(par))
                        {
                            if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.StructTypeKind
                                && LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildLoad(Builder, value, "i" + instruction_id++);
                            }
                            else if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildPointerCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                            else
                            {
                                value = LLVM.BuildBitCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                        }
                        args[k] = value;
                    }
                    var call = LLVM.BuildCall(Builder, fv, args, "");
                    state._stack.Push(new VALUE(call));
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(call.ToString());
                }
                else
                {
                    // No return.
                    for (int k = xargs - 1; k >= 0; --k)
                    {
                        VALUE v = state._stack.Pop();
                        ValueRef par = LLVM.GetParam(fv, (uint)k);
                        ValueRef value = v.V;
                        if (LLVM.TypeOf(value) != LLVM.TypeOf(par))
                        {
                            if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.StructTypeKind
                                && LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildLoad(Builder, value, "i" + instruction_id++);
                            }
                            else if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildPointerCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                            else
                            {
                                value = LLVM.BuildBitCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                        }
                        args[k] = value;
                    }
                    var call = LLVM.BuildCall(Builder, fv, args, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(call.ToString());
                }

            }
        }
    }

    /// <summary>
    /// The LoadArgInst is a class for representing Load Arg instructions. The purpose to
    /// provide a representation of the arg operand of the instruction.
    /// </summary>
    public class ConvertLdArgInst : INST
    {
        public int _arg;

        public ConvertLdArgInst(Mono.Cecil.Cil.Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = state._arguments[_arg];
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value.ToString());
            state._stack.Push(value);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            // For ldarg.1 of a compiler generated closure method, generate code
            // to create an int index for the thread.
            var bb = this.Block;
            var mn = bb._original_method_reference.FullName;
            if (mn.EndsWith("(System.Int32)")
                && mn.Contains("<>c__DisplayClass")
                && _arg == 1)
            {
                //threadId
                var tidx = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.tid.x"];
                var tidy = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.tid.y"];
                var tidz = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.tid.z"];

                //blockIdx
                var ctaidx = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.ctaid.x"];
                var ctaidy = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.ctaid.y"];
                var ctaidz = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.ctaid.z"];

                //blockDim
                var ntidx = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.ntid.x"];
                var ntidy = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.ntid.y"];
                var ntidz = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.ntid.z"];

                //gridDim
                var nctaidx = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.nctaid.x"];
                var nctaidy = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.nctaid.y"];
                var nctaidz = RUNTIME._bcl_runtime_csharp_internal_to_valueref["llvm.nvvm.read.ptx.sreg.nctaid.z"];

                var v_tidx = LLVM.BuildCall(bb.LlvmInfo.Builder, tidx, new ValueRef[] { }, "tidx");
                var v_tidy = LLVM.BuildCall(bb.LlvmInfo.Builder, tidy, new ValueRef[] { }, "tidy");
                var v_ntidx = LLVM.BuildCall(bb.LlvmInfo.Builder, ntidx, new ValueRef[] { }, "ntidx");
                var v_ntidy = LLVM.BuildCall(bb.LlvmInfo.Builder, ntidy, new ValueRef[] { }, "ntidy");
                var v_ctaidx = LLVM.BuildCall(bb.LlvmInfo.Builder, ctaidx, new ValueRef[] { }, "ctaidx");
                var v_ctaidy = LLVM.BuildCall(bb.LlvmInfo.Builder, ctaidy, new ValueRef[] { }, "ctaidx");
                var v_nctaidx = LLVM.BuildCall(bb.LlvmInfo.Builder, nctaidx, new ValueRef[] { }, "nctaidx");

                //int i = (threadIdx.x
                //         + blockDim.x * blockIdx.x
                //         + blockDim.x * gridDim.x * blockDim.y * blockIdx.y
                //         + blockDim.x * gridDim.x * threadIdx.y);

                var t1 = v_tidx;

                var t2 = LLVM.BuildMul(bb.LlvmInfo.Builder, v_ntidx, v_ctaidx, "i" + instruction_id++);

                var t3 = LLVM.BuildMul(bb.LlvmInfo.Builder, v_ntidx, v_nctaidx, "i" + instruction_id++);
                t3 = LLVM.BuildMul(bb.LlvmInfo.Builder, t3, v_ntidy, "i" + instruction_id++);
                t3 = LLVM.BuildMul(bb.LlvmInfo.Builder, t3, v_ctaidy, "i" + instruction_id++);

                var t4 = LLVM.BuildMul(bb.LlvmInfo.Builder, v_ntidx, v_nctaidx, "i" + instruction_id++);
                t4 = LLVM.BuildMul(bb.LlvmInfo.Builder, t4, v_tidy, "i" + instruction_id++);

                var sum = LLVM.BuildAdd(bb.LlvmInfo.Builder, t1, t2, "i" + instruction_id++);
                sum = LLVM.BuildAdd(bb.LlvmInfo.Builder, sum, t3, "i" + instruction_id++);
                sum = LLVM.BuildAdd(bb.LlvmInfo.Builder, sum, t4, "i" + instruction_id++);

                unsafe
                {
                    ValueRef[] args = new ValueRef[0];

                    string name = "_Z21get_kernel_base_indexv";
                    var list2 = RUNTIME.PtxFunctions.ToList();
                    var f = list2.Where(t => t._mangled_name == name).First();
                    ValueRef fv = f._valueref;
                    var call = LLVM.BuildCall(Builder, fv, args, "");
                    sum = LLVM.BuildAdd(bb.LlvmInfo.Builder, sum, call, "i" + instruction_id++);
                }

                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("load " + new VALUE(sum));
                state._stack.Push(new VALUE(sum));
            }
            else
            {
                VALUE value = state._arguments[_arg];
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(value.ToString());
                state._stack.Push(value);
            }
        }
    }

    public class ConvertStArgInst : INST
    {
        public int _arg;

        public ConvertStArgInst(Mono.Cecil.Cil.Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value);

            state._arguments[_arg] = value;
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE value = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value);

            state._arguments[_arg] = value;
        }
    }

    /// <summary>
    /// The LDCInstI4 and LDCInstI8 are classes for representing load constant instructions. The constant
    /// of the instruction is encoded here.
    /// </summary>
    public class ConvertLDCInstI4 : INST
    {
        public Int32 _arg;

        public ConvertLDCInstI4(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = typeof(System.Int32).ToMonoTypeReference();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value);

            state._stack.Push(value);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE value = new VALUE(LLVM.ConstInt(LLVM.Int32Type(), (ulong)_arg, true));
            state._stack.Push(value);
        }
    }

    public class ConvertLDCInstI8 : INST
    {
        public Int64 _arg;

        public ConvertLDCInstI8(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = typeof(System.Int64).ToMonoTypeReference();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value);

            state._stack.Push(value);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE value = new VALUE(LLVM.ConstInt(LLVM.Int64Type(), (ulong)_arg, true));
            state._stack.Push(value);
        }
    }

    public class ConvertLDCInstR4 : INST
    {
        public double _arg;

        public ConvertLDCInstR4(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = typeof(System.Single).ToMonoTypeReference();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value);

            state._stack.Push(value);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE value = new VALUE(LLVM.ConstReal(LLVM.FloatType(), _arg));
            state._stack.Push(value);
        }
    }

    public class ConvertLDCInstR8 : INST
    {
        public double _arg;

        public ConvertLDCInstR8(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = typeof(System.Double).ToMonoTypeReference();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(value);

            state._stack.Push(value);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE value = new VALUE(LLVM.ConstReal(LLVM.DoubleType(), _arg));
            state._stack.Push(value);
        }
    }

    /// <summary>
    /// The LdLoc is a class for representing load local instructions.
    /// </summary>
    public class ConvertLdLoc : INST
    {
        public int _arg;

        public ConvertLdLoc(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._locals[_arg];
            state._stack.Push(v);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var v = state._locals[_arg];
            state._stack.Push(v);
        }
    }

    /// <summary>
    /// The StLoc is a class for representing store local instructions.
    /// </summary>
    public class ConvertStLoc : INST
    {
        public int _arg;

        public ConvertStLoc(Instruction i) : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._stack.Pop();
            state._locals[_arg] = v;
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var v = state._stack.Pop();
            state._locals[_arg] = v;
        }
    }


    public class ConvertCompareInst : INST
    {
        public ConvertCompareInst(Mono.Cecil.Cil.Instruction i) : base(i)
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
        public virtual bool IsSigned { get; set; }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v2 = state._stack.Pop();
            var v1 = state._stack.Pop();
            state._stack.Push(v1);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            // NB: the result of comparisons is a 32-bit quantity, not a bool
            // It must be 32 bits because that is what the spec says.
            // ceq instruction -- page 346 of ecma

            VALUE v2 = state._stack.Pop();
            VALUE v1 = state._stack.Pop();
            // TODO Undoubtably, this will be much more complicated than my initial stab.
            TYPE t1 = v1.T;
            TYPE t2 = v2.T;
            ValueRef cmp = default(ValueRef);
            // Deal with various combinations of types.
            if (t1.isIntegerTy() && t2.isIntegerTy())
            {
                IntPredicate op;
                if (IsSigned) op = _int_pred[(int) Predicate];
                else op = _uint_pred[(int) Predicate];
                cmp = LLVM.BuildICmp(Builder, op, v1.V, v2.V, "i" + instruction_id++);
                // Set up for push of 0/1.
                var return_type = new TYPE(typeof(bool));
                var ret_llvm = LLVM.BuildZExt(Builder, cmp, return_type.IntermediateType, "");
                var ret = new VALUE(ret_llvm, return_type);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(ret);
                state._stack.Push(ret);
            }
            else if (t1.isPointerTy() && t2.isPointerTy())
            {
                // Cast pointers to integer, then compare.
                var i1 = LLVM.BuildPtrToInt(Builder, v1.V, LLVM.Int64Type(), "i" + instruction_id++);
                var i2 = LLVM.BuildPtrToInt(Builder, v2.V, LLVM.Int64Type(), "i" + instruction_id++);
                IntPredicate op;
                if (IsSigned) op = _int_pred[(int)Predicate];
                else op = _uint_pred[(int)Predicate];
                cmp = LLVM.BuildICmp(Builder, op, i1, i2, "i" + instruction_id++);
                // Set up for push of 0/1.
                var return_type = new TYPE(typeof(bool));
                var ret_llvm = LLVM.BuildZExt(Builder, cmp, return_type.IntermediateType, "");
                var ret = new VALUE(ret_llvm, return_type);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(ret);
                state._stack.Push(ret);
            }
            else
                throw new Exception("Unhandled binary operation for given types. "
                                    + t1 + " " + t2);
        }
    }

    public class ConvertCompareAndBranchInst : INST
    {
        public ConvertCompareAndBranchInst(Mono.Cecil.Cil.Instruction i) : base(i)
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

        public Swigged.LLVM.RealPredicate[] _real_pred = new Swigged.LLVM.RealPredicate[]
        {
            Swigged.LLVM.RealPredicate.RealOEQ,
            Swigged.LLVM.RealPredicate.RealONE,
            Swigged.LLVM.RealPredicate.RealOGT,
            Swigged.LLVM.RealPredicate.RealOLT,
            Swigged.LLVM.RealPredicate.RealOGE,
            Swigged.LLVM.RealPredicate.RealOLE,
        };

        public virtual PredicateType Predicate { get; set; }
        public virtual bool IsSigned { get; set; }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v2 = state._stack.Pop();
            var v1 = state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE v2 = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v2);

            VALUE v1 = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v1);

            // TODO Undoubtably, this will be much more complicated than my initial stab.
            TYPE t1 = v1.T;
            TYPE t2 = v2.T;
            ValueRef cmp = default(ValueRef);
            // Deal with various combinations of types.
            if (t1.isIntegerTy() && t2.isIntegerTy())
            {
                IntPredicate op;
                if (IsSigned) op = _int_pred[(int)Predicate];
                else op = _uint_pred[(int)Predicate];

                cmp = LLVM.BuildICmp(Builder, op, v1.V, v2.V, "i" + instruction_id++);

                var edge1 = Block._graph.SuccessorEdges(Block).ToList()[0];
                var edge2 = Block._graph.SuccessorEdges(Block).ToList()[1];
                var s1 = edge1.To;
                var s2 = edge2.To;
                // Now, in order to select the correct branch, we need to know what
                // edge represents the "true" branch. During construction, there is
                // no guarentee that the order is consistent.
                var owner = Block._graph.Vertices.Where(
					n => n.Instructions.Where(ins =>
                    {
                        if (n.Entry._original_method_reference != Block.Entry._original_method_reference)
                            return false;
                        if (ins.Instruction.Offset != this.Instruction.Offset)
                            return false;
                        return true;
                    }).Any()).ToList();
                if (owner.Count() != 1)
                    throw new Exception("Cannot find instruction!");
                CFG.Vertex true_node = owner.FirstOrDefault();
                if (s2 == true_node)
                {
                    s1 = s2;
                    s2 = true_node;
                }
                LLVM.BuildCondBr(Builder, cmp, s1.LlvmInfo.BasicBlock, s2.LlvmInfo.BasicBlock);
                return;
            }
            if (t1.isFloatingPointTy() && t2.isFloatingPointTy())
            {
                RealPredicate op;
                if (IsSigned) op = _real_pred[(int)Predicate];
                else op = _real_pred[(int)Predicate];

                cmp = LLVM.BuildFCmp(Builder, op, v1.V, v2.V, "i" + instruction_id++);

                var edge1 = Block._graph.SuccessorEdges(Block).ToList()[0];
                var edge2 = Block._graph.SuccessorEdges(Block).ToList()[1];
                var s1 = edge1.To;
                var s2 = edge2.To;
                // Now, in order to select the correct branch, we need to know what
                // edge represents the "true" branch. During construction, there is
                // no guarentee that the order is consistent.
                var owner = Block._graph.Vertices.Where(
                    n => n.Instructions.Where(ins =>
                    {
                        if (n.Entry._original_method_reference != Block.Entry._original_method_reference)
                            return false;
                        if (ins.Instruction.Offset != this.Instruction.Offset)
                            return false;
                        return true;
                    }).Any()).ToList();
                if (owner.Count() != 1)
                    throw new Exception("Cannot find instruction!");
                CFG.Vertex true_node = owner.FirstOrDefault();
                if (s2 == true_node)
                {
                    s1 = s2;
                    s2 = true_node;
                }
                LLVM.BuildCondBr(Builder, cmp, s1.LlvmInfo.BasicBlock, s2.LlvmInfo.BasicBlock);
                return;
            }
            throw new Exception("Unhandled compare and branch.");
        }
    }

    public class ConvertConvInst : INST
    {
        protected TYPE _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        VALUE convert_full(VALUE src)
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
                    return new VALUE(
                        _dst.is_unsigned
                        ? LLVM.BuildZExt(Builder, src.V, dtype, "i" + instruction_id++)
                        : LLVM.BuildSExt(Builder, src.V, dtype, "i" + instruction_id++));

                if (dtype == LLVM.DoubleType() && stype == LLVM.FloatType())
                    return new VALUE(LLVM.BuildFPExt(Builder, src.V, dtype, "i" + instruction_id++));

                /* Trunc */
                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.Int32Type() || dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type()))
                    return new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
                if (stype == LLVM.Int32Type()
                    && (dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type()))
                    return new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
                if (stype == LLVM.Int16Type()
                    && dtype == LLVM.Int8Type())
                    return new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
                if (stype == LLVM.DoubleType()
                    && dtype == LLVM.FloatType())
                    return new VALUE(LLVM.BuildFPTrunc(Builder, src.V, dtype, "i" + instruction_id++));

                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.FloatType()))
                    return new VALUE(LLVM.BuildSIToFP(Builder, src.V, dtype, "i" + instruction_id++));
                if (stype == LLVM.Int32Type()
                    && (dtype == LLVM.FloatType()))
                    return new VALUE(LLVM.BuildSIToFP(Builder, src.V, dtype, "i" + instruction_id++));
                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.DoubleType()))
                    return new VALUE(LLVM.BuildSIToFP(Builder, src.V, dtype, "i" + instruction_id++));
                if (stype == LLVM.Int32Type()
                    && (dtype == LLVM.DoubleType()))
                    return new VALUE(LLVM.BuildSIToFP(Builder, src.V, dtype, "i" + instruction_id++));

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
                return new VALUE(default(ValueRef));
            }
            else
            {
                return src;
            }
        }

        public ConvertConvInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var s = state._stack.Pop();
            state._stack.Push(s);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE s = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(s.ToString());

            VALUE d = convert_full(s);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(d.ToString());

            state._stack.Push(d);
        }
    }

    public class ConvertConvOvfInst : ConvertConvInst
    {
        public ConvertConvOvfInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _check_overflow = true;
        }
    }

    public class ConvertConvOvfUnsInst : ConvertConvInst
    {
        public ConvertConvOvfUnsInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _check_overflow = true;
            _from_unsigned = true;
        }
    }

    public class ConvertUnsInst : ConvertConvInst
    {
        public ConvertUnsInst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _from_unsigned = true;
        }
    }

    public class ConvertLoadElement : INST
    {
        protected TYPE _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertLoadElement(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var i = state._stack.Pop();
            var a = state._stack.Pop();
            var e = a.GetElementType();
            var ar = a as ArrayType;
            var e2 = ar.ElementType;
            state._stack.Push(e2);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            VALUE a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var load = a.V;
            load = LLVM.BuildLoad(Builder, load, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(load));

            // Load array base.
            ValueRef extract_value = LLVM.BuildExtractValue(Builder, load, 0, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(extract_value));

            // Now add in index to pointer.
            ValueRef[] indexes = new ValueRef[1];
            indexes[0] = i.V;
            ValueRef gep = LLVM.BuildInBoundsGEP(Builder, extract_value, indexes, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(gep));

            load = LLVM.BuildLoad(Builder, gep, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(load));

            if (_dst != null &&_dst.IntermediateType != LLVM.TypeOf(load))
            {
                load = LLVM.BuildIntCast(Builder, load, _dst.IntermediateType, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(load));
            }
            else if (_dst == null)
            {
                var t_v = LLVM.TypeOf(load);
                TypeRef t_to;
                // Type information for instruction obtuse. 
                // Use LLVM type and set stack type.
                if (t_v == LLVM.Int8Type() || t_v == LLVM.Int16Type())
                {
                    load = LLVM.BuildIntCast(Builder, load, LLVM.Int32Type(), "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(load));
                }
                else
                    t_to = t_v;
                //var op = this.Operand;
                //var tt = op.GetType();
            }

            state._stack.Push(new VALUE(load));
        }
    }

    public class ConvertStoreElement : INST
    {
        protected TYPE _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertStoreElement(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());

            var i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            var a = state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());

            VALUE i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            VALUE a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var load = a.V;
            load = LLVM.BuildLoad(Builder, load, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(load));

            ValueRef extract_value = LLVM.BuildExtractValue(Builder, load, 0, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(extract_value));

            // Now add in index to pointer.
            ValueRef[] indexes = new ValueRef[1];
            indexes[0] = i.V;
            ValueRef gep = LLVM.BuildInBoundsGEP(Builder, extract_value, indexes, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(gep));

            var value = v.V;
            if (_dst != null && _dst.VerificationType.ToTypeRef() != v.T.IntermediateType)
            {
                value = LLVM.BuildIntCast(Builder, value, _dst.VerificationType.ToTypeRef(), "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(value));
            }
            else if (_dst == null)
            {
                var t_v = LLVM.TypeOf(value);
                var t_d = LLVM.TypeOf(gep);
                var t_e = LLVM.GetElementType(t_d);
                if (t_v != t_e && LLVM.GetTypeKind(t_e) != TypeKind.StructTypeKind)
                {
                    value = LLVM.BuildIntCast(Builder, value, t_e, "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(value));
                }
            }

            // Store.
            var store = LLVM.BuildStore(Builder, value, gep);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(store));
        }
    }


    public class ConvertLoadElementA : INST
    {
        public ConvertLoadElementA(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            var a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var e = a.GetElementType();

            // Create reference type of element type.
            var v = new Mono.Cecil.ByReferenceType(e);

            state._stack.Push(v);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE i = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(i.ToString());

            VALUE a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a.ToString());

            var load = a.V;
            load = LLVM.BuildLoad(Builder, load, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(load));

            // Load array base.
            ValueRef extract_value = LLVM.BuildExtractValue(Builder, load, 0, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(extract_value));

            // Now add in index to pointer.
            ValueRef[] indexes = new ValueRef[1];
            indexes[0] = i.V;
            ValueRef gep = LLVM.BuildInBoundsGEP(Builder, extract_value, indexes, "i" + instruction_id++);
            var result = new VALUE(gep);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(result);

            state._stack.Push(result);
        }
    }

    public class ConvertStoreField : INST
    {
        public ConvertStoreField(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // stfld, page 427 of ecma 335
            var v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());
            var o = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(o.ToString());
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // stfld, page 427 of ecma 335
            var operand = this.Operand;
            if (operand as FieldReference == null) throw new Exception("Error in parsing stfld.");
            var field_reference = operand as FieldReference;
            VALUE v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v);
            VALUE o = state._stack.Pop();
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
                        field_size = BUFFERS.SizeOf(typeof(IntPtr));
                        alignment = BUFFERS.Alignment(typeof(IntPtr));
                    }
                    else
                    {
                        var ft = f.FieldType.ToSystemType();
                        field_size = BUFFERS.SizeOf(ft);
                        alignment = BUFFERS.Alignment(ft);
                    }

                    int padding = BUFFERS.Padding(size, alignment);
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

                var dst = LLVM.BuildStructGEP(Builder, o.V, offset, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(dst));

                var dd = LLVM.TypeOf(dst);
                var ddd = LLVM.GetElementType(dd);
                var src = v;
                TypeRef stype = LLVM.TypeOf(src.V);
                TypeRef dtype = ddd;

                /* Trunc */
                if (stype == LLVM.Int64Type()
                    && (dtype == LLVM.Int32Type() || dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type() ||
                        dtype == LLVM.Int1Type()))
                    src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
                else if (stype == LLVM.Int32Type()
                         && (dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                    src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
                else if (stype == LLVM.Int16Type()
                         && (dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                    src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));

                var store = LLVM.BuildStore(Builder, src.V, dst);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(store));
            }
            else if (isSt)
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
                        field_size = BUFFERS.SizeOf(typeof(IntPtr));
                        alignment = BUFFERS.Alignment(typeof(IntPtr));
                    }
                    else
                    {
                        var ft = f.FieldType.ToSystemType();
                        field_size = BUFFERS.SizeOf(ft);
                        alignment = BUFFERS.Alignment(ft);
                    }

                    int padding = BUFFERS.Padding(size, alignment);
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

                var value = LLVM.BuildExtractValue(Builder, o.V, offset, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(value));

                var load_value = new VALUE(value);
                bool isPtrLoad = load_value.T.isPointerTy();
                if (isPtrLoad)
                {
                    var mono_field_type = field.FieldType;
                    TypeRef type = mono_field_type.ToTypeRef();
                    value = LLVM.BuildBitCast(Builder,
                        value, type, "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(value));
                }

                var store = LLVM.BuildStore(Builder, v.V, value);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(store));
            }
            else
            {
                throw new Exception("Value type ldfld not implemented!");
            }
        }
    }

    public class ConvertLoadIndirect : INST
    {
        protected TYPE _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertLoadIndirect(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var i = state._stack.Pop();
            var v = i.GetElementType();
            state._stack.Push(v);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("ConvertLoadIndirect into function " + v.ToString());

            TypeRef tr = LLVM.TypeOf(v.V);
            TypeKind kind = LLVM.GetTypeKind(tr);

            var load = v.V;
            load = LLVM.BuildLoad(Builder, load, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(load));

            if (_dst != null && _dst.IntermediateType != LLVM.TypeOf(load))
            {
                load = LLVM.BuildIntCast(Builder, load, _dst.IntermediateType, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(load));
            }
            else if (_dst == null)
            {
                var t_v = LLVM.TypeOf(load);
                TypeRef t_to;
                // Type information for instruction obtuse. 
                // Use LLVM type and set stack type.
                if (t_v == LLVM.Int8Type() || t_v == LLVM.Int16Type())
                {
                    load = LLVM.BuildIntCast(Builder, load, LLVM.Int32Type(), "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(load));
                }
                else
                    t_to = t_v;
                //var op = this.Operand;
                //var tt = op.GetType();
            }

            state._stack.Push(new VALUE(load));
        }
    }

    public class ConvertStoreIndirect : INST
    {
        protected TYPE _dst;
        protected bool _check_overflow;
        protected bool _from_unsigned;

        public ConvertStoreIndirect(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());

            var o = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(o.ToString());
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            VALUE src = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(src);

            VALUE a = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(a);

            TypeRef stype = LLVM.TypeOf(src.V);
            TypeRef dtype = _dst.IntermediateType;

            /* Trunc */
            if (stype == LLVM.Int64Type()
                  && (dtype == LLVM.Int32Type() || dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
            else if (stype == LLVM.Int32Type()
                  && (dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
            else if (stype == LLVM.Int16Type()
                  && (dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
                        
            var zz = LLVM.BuildStore(Builder, src.V, a.V);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("Store = " + new VALUE(zz).ToString());
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

    public class i_arglist : INST
    {
        public i_arglist(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_beq : ConvertCompareAndBranchInst
    {
        public i_beq(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.eq;
            IsSigned = true;
        }
    }

    public class i_beq_s : ConvertCompareAndBranchInst
    {
        public i_beq_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.eq;
            IsSigned = true;
        }
    }

    public class i_bge : ConvertCompareAndBranchInst
    {
        public i_bge(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = true;
        }
    }

    public class i_bge_un : ConvertCompareAndBranchInst
    {
        public i_bge_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = false;
        }
    }

    public class i_bge_un_s : ConvertCompareAndBranchInst
    {
        public i_bge_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = false;
        }
    }

    public class i_bge_s : ConvertCompareAndBranchInst
    {
        public i_bge_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ge;
            IsSigned = true;
        }
    }

    public class i_bgt : ConvertCompareAndBranchInst
    {
        public i_bgt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = true;
        }
    }

    public class i_bgt_s : ConvertCompareAndBranchInst
    {
        public i_bgt_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = true;
        }
    }

    public class i_bgt_un : ConvertCompareAndBranchInst
    {
        public i_bgt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = false;
        }
    }

    public class i_bgt_un_s : ConvertCompareAndBranchInst
    {
        public i_bgt_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = false;
        }
    }

    public class i_ble : ConvertCompareAndBranchInst
    {
        public i_ble(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
            IsSigned = true;
        }
    }

    public class i_ble_s : ConvertCompareAndBranchInst
    {
        public i_ble_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
        }
    }

    public class i_ble_un : ConvertCompareAndBranchInst
    {
        public i_ble_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
            IsSigned = false;
        }
    }

    public class i_ble_un_s : ConvertCompareAndBranchInst
    {
        public i_ble_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.le;
            IsSigned = false;
        }
    }

    public class i_blt : ConvertCompareAndBranchInst
    {
        public i_blt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = true;
        }
    }

    public class i_blt_s : ConvertCompareAndBranchInst
    {
        public i_blt_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = true;
        }
    }

    public class i_blt_un : ConvertCompareAndBranchInst
    {
        public i_blt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = false;
        }
    }

    public class i_blt_un_s : ConvertCompareAndBranchInst
    {
        public i_blt_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = false;
        }
    }

    public class i_bne_un : ConvertCompareAndBranchInst
    {
        public i_bne_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ne;
            IsSigned = false;
        }
    }

    public class i_bne_un_s : ConvertCompareAndBranchInst
    {
        public i_bne_un_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.ne;
            IsSigned = false;
        }
    }

    public class i_box : INST
    {
        public i_box(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // box – convert a boxable value to its boxed form, page 394
            var typetok = this.Operand;
            var tr = typetok as TypeReference;
            var tr2 = tr.RewriteMonoTypeReference();
            var v = tr2.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            TypeReference v2 = state._stack.Pop();
            state._stack.Push(v);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            ValueRef new_obj;

            // Get meta of object.
            var operand = this.Operand;
            var tr = operand as TypeReference;
            tr = tr.RewriteMonoTypeReference();
            tr = tr.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            var meta = RUNTIME.GetBclType(tr);

            // Generate code to allocate object and stuff.
            // This boxes the value.
            var xx1 = RUNTIME.BclNativeMethods.ToList();
            var xx2 = RUNTIME.PtxFunctions.ToList();
            var xx = xx2
                .Where(t => { return t._mangled_name == "_Z23Heap_AllocTypeVoidStarsPv"; });
            var xxx = xx.ToList();
            RUNTIME.PtxFunction first_kv_pair = xx.FirstOrDefault();
            if (first_kv_pair == null)
                throw new Exception("Yikes.");

            ValueRef fv2 = first_kv_pair._valueref;
            ValueRef[] args = new ValueRef[1];

            //args[0] = LLVM.BuildIntToPtr(Builder,
            //    LLVM.ConstInt(LLVM.Int64Type(), (ulong)meta.ToInt64(), false),
            //    LLVM.PointerType(LLVM.VoidType(), 0),
            //    "i" + instruction_id++);
            args[0] = LLVM.ConstInt(LLVM.Int64Type(), (ulong)meta.ToInt64(), false);
            var call = LLVM.BuildCall(Builder, fv2, args, "i" + instruction_id++);
            var type_casted = LLVM.BuildIntToPtr(Builder, call,
                typeof(System.Object).ToMonoTypeReference().ToTypeRef(),
                "i" + instruction_id++);
            new_obj = type_casted;
            // Stuff value in buffer of object.
            var s = state._stack.Pop();
            ValueRef v = LLVM.BuildPointerCast(Builder, new_obj, LLVM.PointerType(LLVM.TypeOf(s.V), 0),
                "i" + instruction_id++);
            ValueRef store = LLVM.BuildStore(Builder, s.V, v);

            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(new_obj));

            state._stack.Push(new VALUE(new_obj));
        }
    }

    public class i_br : INST
    {
        public i_br(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var edge = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s = edge.To;
            var br = LLVM.BuildBr(Builder, s.LlvmInfo.BasicBlock);
        }
    }

    public class i_br_s : INST
    {
        public i_br_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var edge = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s = edge.To;
            var br = LLVM.BuildBr(Builder, s.LlvmInfo.BasicBlock);
        }
    }

    public class i_break : INST
    {
        public i_break(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_brfalse : INST
    {
        public i_brfalse(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // brfalse, page 340 of ecma 335
            var v = state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // brfalse, page 340 of ecma 335
            object operand = this.Operand;
            Instruction instruction = operand as Instruction;
            var v = state._stack.Pop();
            var value = v.V;
            ValueRef condition;
            var type_of_value = LLVM.TypeOf(v.V);
            if (LLVM.GetTypeKind(type_of_value) == TypeKind.PointerTypeKind)
            {
                var cast = LLVM.BuildPtrToInt(Builder, v.V, LLVM.Int64Type(), "i" + instruction_id++);
                var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, cast, v2, "i" + instruction_id++);
            }
            else if (LLVM.GetTypeKind(type_of_value) == TypeKind.IntegerTypeKind)
            {
                if (type_of_value == LLVM.Int8Type() || type_of_value == LLVM.Int16Type())
                {
                    value = LLVM.BuildIntCast(Builder, value, LLVM.Int32Type(), "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(value));
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int32Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int64Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, value, v2, "i" + instruction_id++);
                }
                else throw new Exception("Unhandled type in brfalse.s");
            }
            else throw new Exception("Unhandled type in brfalse.s");
            // In order to select the correct branch, we need to know what
            // edge represents the "true" branch. During construction, there is
            // no guarentee that the order is consistent.
            var owner = Block._graph.Vertices.Where(
                n => n.Instructions.Where(ins =>
                {
                    if (n.Entry._original_method_reference != Block.Entry._original_method_reference)
                        return false;
                    if (ins.Instruction.Offset != instruction.Offset)
                        return false;
                    return true;
                }).Any()).ToList();
            if (owner.Count != 1)
                throw new Exception("Cannot find instruction!");
            var edge1 = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s1 = edge1.To;
            var edge2 = Block._graph.SuccessorEdges(Block).ToList()[1];
            var s2 = edge2.To;
            CFG.Vertex then_node = owner.FirstOrDefault();
            CFG.Vertex else_node = s1 == then_node ? s2 : s1;
            LLVM.BuildCondBr(Builder, condition, then_node.LlvmInfo.BasicBlock, else_node.LlvmInfo.BasicBlock);
        }
    }

    public class i_brfalse_s : INST
    {
        public i_brfalse_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // brfalse.s, page 340 of ecma 335
            var v = state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // brfalse.s, page 340 of ecma 335
            object operand = this.Operand;
            Instruction instruction = operand as Instruction;
            var v = state._stack.Pop();
            var value = v.V;
            ValueRef condition;
            var type_of_value = LLVM.TypeOf(v.V);
            if (LLVM.GetTypeKind(type_of_value) == TypeKind.PointerTypeKind)
            {
                var cast = LLVM.BuildPtrToInt(Builder, v.V, LLVM.Int64Type(), "i" + instruction_id++);
                var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, cast, v2, "i" + instruction_id++);
            }
            else if (LLVM.GetTypeKind(type_of_value) == TypeKind.IntegerTypeKind)
            {
                if (type_of_value == LLVM.Int8Type() || type_of_value == LLVM.Int16Type())
                {
                    value = LLVM.BuildIntCast(Builder, value, LLVM.Int32Type(), "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(value));
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int32Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int64Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntEQ, value, v2, "i" + instruction_id++);
                }
                else throw new Exception("Unhandled type in brfalse.s");
            }
            else throw new Exception("Unhandled type in brfalse.s");
            // In order to select the correct branch, we need to know what
            // edge represents the "true" branch. During construction, there is
            // no guarentee that the order is consistent.
            var owner = Block._graph.Vertices.Where(
                n => n.Instructions.Where(ins =>
                {
                    if (n.Entry._original_method_reference != Block.Entry._original_method_reference)
                        return false;
                    if (ins.Instruction.Offset != instruction.Offset)
                        return false;
                    return true;
                }).Any()).ToList();
            if (owner.Count != 1)
                throw new Exception("Cannot find instruction!");
            var edge1 = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s1 = edge1.To;
            var edge2 = Block._graph.SuccessorEdges(Block).ToList()[1];
            var s2 = edge2.To;
            CFG.Vertex then_node = owner.FirstOrDefault();
            CFG.Vertex else_node = s1 == then_node ? s2 : s1;
            LLVM.BuildCondBr(Builder, condition, then_node.LlvmInfo.BasicBlock, else_node.LlvmInfo.BasicBlock);
        }
    }

    public class i_brtrue : INST
    {
        public i_brtrue(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // brtrue, page 341 of ecma 335
            object operand = this.Operand;
            Instruction instruction = operand as Instruction;
            var v = state._stack.Pop();
            var value = v.V;
            ValueRef condition;
            var type_of_value = LLVM.TypeOf(v.V);
            if (LLVM.GetTypeKind(type_of_value) == TypeKind.PointerTypeKind)
            {
                var cast = LLVM.BuildPtrToInt(Builder, v.V, LLVM.Int64Type(), "i" + instruction_id++);
                // Verify an object, as according to spec. We'll do that using BCL.
                var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, cast, v2, "i" + instruction_id++);
            }
            else if (LLVM.GetTypeKind(type_of_value) == TypeKind.IntegerTypeKind)
            {
                if (type_of_value == LLVM.Int8Type() || type_of_value == LLVM.Int16Type())
                {
                    value = LLVM.BuildIntCast(Builder, value, LLVM.Int32Type(), "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(value));
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int32Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int64Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, value, v2, "i" + instruction_id++);
                }
                else throw new Exception("Unhandled type in brtrue");
            }
            else throw new Exception("Unhandled type in brtrue");
            // In order to select the correct branch, we need to know what
            // edge represents the "true" branch. During construction, there is
            // no guarentee that the order is consistent.
            var owner = Block._graph.Vertices.Where(
                n => n.Instructions.Where(ins =>
                {
                    if (n.Entry._original_method_reference != Block.Entry._original_method_reference)
                        return false;
                    if (ins.Instruction.Offset != instruction.Offset)
                        return false;
                    return true;
                }).Any()).ToList();
            if (owner.Count != 1)
                throw new Exception("Cannot find instruction!");
            var edge1 = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s1 = edge1.To;
            var edge2 = Block._graph.SuccessorEdges(Block).ToList()[1];
            var s2 = edge2.To;
            CFG.Vertex then_node = owner.FirstOrDefault();
            CFG.Vertex else_node = s1 == then_node ? s2 : s1;
            LLVM.BuildCondBr(Builder, condition, then_node.LlvmInfo.BasicBlock, else_node.LlvmInfo.BasicBlock);
        }
    }

    public class i_brtrue_s : INST
    {
        public i_brtrue_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // brtrue, page 341 of ecma 335
            object operand = this.Operand;
            Instruction instruction = operand as Instruction;
            var v = state._stack.Pop();
            var value = v.V;
            ValueRef condition;
            var type_of_value = LLVM.TypeOf(v.V);
            if (LLVM.GetTypeKind(type_of_value) == TypeKind.PointerTypeKind)
            {
                var cast = LLVM.BuildPtrToInt(Builder, v.V, LLVM.Int64Type(), "i" + instruction_id++);
                var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, cast, v2, "i" + instruction_id++);
            }
            else if (LLVM.GetTypeKind(type_of_value) == TypeKind.IntegerTypeKind)
            {
                if (type_of_value == LLVM.Int8Type() || type_of_value == LLVM.Int16Type())
                {
                    value = LLVM.BuildIntCast(Builder, value, LLVM.Int32Type(), "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(value));
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int32Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int32Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, value, v2, "i" + instruction_id++);
                }
                else if (type_of_value == LLVM.Int64Type())
                {
                    var v2 = LLVM.ConstInt(LLVM.Int64Type(), 0, false);
                    condition = LLVM.BuildICmp(Builder, IntPredicate.IntNE, value, v2, "i" + instruction_id++);
                }
                else throw new Exception("Unhandled type in brtrue");
            }
            else throw new Exception("Unhandled type in brtrue");
            // In order to select the correct branch, we need to know what
            // edge represents the "true" branch. During construction, there is
            // no guarentee that the order is consistent.
            var owner = Block._graph.Vertices.Where(
                n => n.Instructions.Where(ins =>
                {
                    if (n.Entry._original_method_reference != Block.Entry._original_method_reference)
                        return false;
                    if (ins.Instruction.Offset != instruction.Offset)
                        return false;
                    return true;
                }).Any()).ToList();
            if (owner.Count != 1)
                throw new Exception("Cannot find instruction!");
            var edge1 = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s1 = edge1.To;
            var edge2 = Block._graph.SuccessorEdges(Block).ToList()[1];
            var s2 = edge2.To;
            CFG.Vertex then_node = owner.FirstOrDefault();
            CFG.Vertex else_node = s1 == then_node ? s2 : s1;
            LLVM.BuildCondBr(Builder, condition, then_node.LlvmInfo.BasicBlock, else_node.LlvmInfo.BasicBlock);
        }
    }

    public class i_call : ConvertCallInst
    {
        public i_call(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_calli : ConvertCallInst
    {
        public i_calli(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_callvirt : ConvertCallInst
    {
        public i_callvirt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // callvirt – call a method associated, at runtime, with an object, page 396.
            object method = this.Operand;
            if (method as Mono.Cecil.MethodReference == null) throw new Exception();
            Mono.Cecil.MethodReference orig_mr = method as Mono.Cecil.MethodReference;
            orig_mr = orig_mr.SubstituteMethod(null, null);
            var mr = orig_mr.FixGenericMethods(this.Block._original_method_reference);
            var md = mr.Resolve();
            bool is_virtual = md.IsVirtual;
            bool has_this = true;
            bool is_explicit_this = mr.ExplicitThis;
            has_this = has_this && !is_explicit_this;
            int xargs = (has_this ? 1 : 0) + mr.Parameters.Count;
            // callvirt can be called for non-virtual functions!!!!!!!! BS!
            // Switch appropriately.
           // if (!is_virtual) throw new Exception("Fucked.");
            {
                VALUE this_parameter = state._stack.PeekTop(xargs - 1);
                ValueRef[] args1 = new ValueRef[2];
                var this_ptr = LLVM.BuildPtrToInt(Builder, this_parameter.V, LLVM.Int64Type(), "i" + instruction_id++);
                args1[0] = this_ptr;
                var token = 0x06000000 | mr.MetadataToken.RID;
                var v2 = LLVM.ConstInt(LLVM.Int32Type(), token, false);
                args1[1] = v2;
                var f = RUNTIME.PtxFunctions.Where(t => t._mangled_name == "_Z21MetaData_GetMethodJitPvi").First();
                var addr_method = LLVM.BuildCall(Builder, f._valueref, args1, "");
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(addr_method));
                bool has_return = mr.ReturnType.FullName != "System.Void";
                TypeRef[] lparams = new TypeRef[xargs];
                ValueRef[] args = new ValueRef[xargs];
                var pars = mr.Parameters;
                for (int k = mr.Parameters.Count - 1; k >= 0; --k)
                {
                    VALUE v = state._stack.Pop();
                    var par_type = pars[k].ParameterType.InstantiateGeneric(mr);
                    TypeRef par = par_type.ToTypeRef();
                    ValueRef value = v.V;
                    if (LLVM.TypeOf(value) != par)
                    {
                        if (LLVM.GetTypeKind(par) == TypeKind.StructTypeKind
                            && LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.PointerTypeKind)
                            value = LLVM.BuildLoad(Builder, value, "i" + instruction_id++);
                        else if (LLVM.GetTypeKind(par) == TypeKind.PointerTypeKind)
                            value = LLVM.BuildPointerCast(Builder, value, par, "i" + instruction_id++);
                        else if (LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.IntegerTypeKind)
                            value = LLVM.BuildIntCast(Builder, value, par, "i" + instruction_id++);
                        else
                            value = LLVM.BuildBitCast(Builder, value, par, "i" + instruction_id++);
                    }
                    lparams[k + xargs - mr.Parameters.Count] = par;
                    args[k + xargs - mr.Parameters.Count] = value;
                }
                if (has_this)
                {
                    VALUE v = state._stack.Pop();
                    TypeRef par = mr.DeclaringType.ToTypeRef();
                    ValueRef value = v.V;
                    if (LLVM.TypeOf(value) != par)
                    {
                        if (LLVM.GetTypeKind(par) == TypeKind.StructTypeKind
                            && LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.PointerTypeKind)
                            value = LLVM.BuildLoad(Builder, value, "i" + instruction_id++);
                        else if (LLVM.GetTypeKind(par) == TypeKind.PointerTypeKind)
                            value = LLVM.BuildPointerCast(Builder, value, par, "i" + instruction_id++);
                        else if (LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.IntegerTypeKind)
                            value = LLVM.BuildIntCast(Builder, value, par, "i" + instruction_id++);
                        else
                            value = LLVM.BuildBitCast(Builder, value, par, "");
                    }
                    lparams[0] = par;
                    args[0] = value;
                }
                TypeRef return_type = has_return ?
                    mr.ReturnType.InstantiateGeneric(mr).ToTypeRef() : LLVM.Int64Type();

                // There are two ways a function can be called: with direct parameters,
                // or with arrayed parameters. One way is "direct" where parameters are
                // passed as is. This occurs for Campy JIT code. The other way is "indirect"
                // where the parameters are passed via arrays. This occurs for BCL internal
                // functions.

                CFG.Vertex the_entry = this.Block._graph.Vertices.Where(v =>
                    (v.IsEntry && JITER.MethodName(v._original_method_reference) == mr.FullName)).ToList().FirstOrDefault();

                if (the_entry != null)
                {
                    var function_type = LLVM.FunctionType(return_type, lparams, false);
                    var ptr_function_type = LLVM.PointerType(function_type, 0);
                    var ptr_method = LLVM.BuildIntToPtr(Builder, addr_method, ptr_function_type, "i" + instruction_id++);
                    var call = LLVM.BuildCall(Builder, ptr_method, args, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(call.ToString());
                    if (has_return)
                    {
                        state._stack.Push(new VALUE(call));
                    }
                }
                else
                {
                    TypeRef[] internal_lparams = new TypeRef[3];
                    ValueRef[] internal_args = new ValueRef[3];
                    TypeRef internal_return_type = LLVM.VoidType();
                    internal_lparams[0] = internal_lparams[1] = internal_lparams[2] = LLVM.Int64Type();
                    var function_type = LLVM.FunctionType(internal_return_type, internal_lparams, false);
                    var ptr_function_type = LLVM.PointerType(function_type, 0);
                    var ptr_method = LLVM.BuildIntToPtr(Builder, addr_method, ptr_function_type, "i" + instruction_id++);
                    var parameter_type = LLVM.ArrayType(LLVM.Int64Type(), (uint)args.Count() - 1);
                    var arg_buffer = LLVM.BuildAlloca(Builder, parameter_type, "i" + instruction_id++);
                    LLVM.SetAlignment(arg_buffer, 64);
                    var base_of_args = LLVM.BuildPointerCast(Builder, arg_buffer,
                        LLVM.PointerType(LLVM.Int64Type(), 0), "i" + instruction_id++);
                    for (int i = 1; i < args.Count(); ++i)
                    {
                        var im1 = i - 1;
                        ValueRef[] index = new ValueRef[1] { LLVM.ConstInt(LLVM.Int32Type(), (ulong)im1, true) };
                        var add = LLVM.BuildInBoundsGEP(Builder, base_of_args, index, "i" + instruction_id++);
                        ValueRef v = LLVM.BuildPointerCast(Builder, add, LLVM.PointerType(LLVM.TypeOf(args[i]), 0), "i" + instruction_id++);
                        ValueRef store = LLVM.BuildStore(Builder, args[i], v);
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new VALUE(store));
                    }
                    ValueRef return_buffer = LLVM.BuildAlloca(Builder, return_type, "i" + instruction_id++);
                    LLVM.SetAlignment(return_buffer, 64);
                    var pt = LLVM.BuildPtrToInt(Builder, args[0], LLVM.Int64Type(), "i" + instruction_id++);
                    var pp = LLVM.BuildPtrToInt(Builder, arg_buffer, LLVM.Int64Type(), "i" + instruction_id++);
                    var pr = LLVM.BuildPtrToInt(Builder, return_buffer, LLVM.Int64Type(), "i" + instruction_id++);
                    internal_args[0] = pt;
                    internal_args[1] = pp;
                    internal_args[2] = pr;
                    var call = LLVM.BuildCall(Builder, ptr_method, internal_args, "");
                    if (has_return)
                    {
                        var load = LLVM.BuildLoad(Builder, return_buffer, "i" + instruction_id++);
                        state._stack.Push(new VALUE(load));
                    }
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(call.ToString());
                }
            }
        }
    }

    public class i_castclass : INST
    {
        public i_castclass(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var typetok = Operand;
            var tr = typetok as TypeReference;
            var tr2 = tr.RewriteMonoTypeReference();
            var v = tr2.Deresolve(this.Block._original_method_reference.DeclaringType, null);
        }

        public override void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
        }
    }

    public class i_ceq : ConvertCompareInst
    {
        public i_ceq(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.eq;
            IsSigned = true;
        }
    }

    public class i_cgt : ConvertCompareInst
    {
        public i_cgt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = true;
        }
    }

    public class i_cgt_un : ConvertCompareInst
    {
        public i_cgt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.gt;
            IsSigned = false;
        }
    }

    public class i_ckfinite : INST
    {
        public i_ckfinite(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_clt : ConvertCompareInst
    {
        public i_clt(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = true;
        }
    }

    public class i_clt_un : ConvertCompareInst
    {
        public i_clt_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Predicate = PredicateType.lt;
            IsSigned = false;
        }
    }

    public class i_constrained : INST
    {
        public i_constrained(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
        }

        public override void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            base.Convert(state);
        }
    }

    public class i_conv_i1 : ConvertConvInst
    {
        public i_conv_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_conv_i2 : ConvertConvInst
    {
        public i_conv_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_conv_i4 : ConvertConvInst
    {
        public i_conv_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_conv_i8 : ConvertConvInst
    {
        public i_conv_i8(Mono.Cecil.Cil.Instruction i)
                : base(i)
        {
            _dst = new TYPE(typeof(long));
        }
    }

    public class i_conv_i : ConvertConvInst
    {
        public i_conv_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_conv_ovf_i1 : ConvertConvOvfInst
    {
        public i_conv_ovf_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_conv_ovf_i1_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_i1_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_conv_ovf_i2 : ConvertConvOvfInst
    {
        public i_conv_ovf_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_conv_ovf_i2_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_i2_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_conv_ovf_i4 : ConvertConvOvfInst
    {
        public i_conv_ovf_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_conv_ovf_i4_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_i4_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_conv_ovf_i8 : ConvertConvOvfInst
    {
        public i_conv_ovf_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(long));
        }
    }

    public class i_conv_ovf_i8_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_i8_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(long));
        }
    }

    public class i_conv_ovf_i : ConvertConvOvfInst
    {
        public i_conv_ovf_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_conv_ovf_i_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_i_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_conv_ovf_u1 : ConvertConvOvfInst
    {
        public i_conv_ovf_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(byte));
        }
    }

    public class i_conv_ovf_u1_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_u1_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(byte));
        }
    }

    public class i_conv_ovf_u2 : ConvertConvOvfInst
    {
        public i_conv_ovf_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ushort));
        }
    }

    public class i_conv_ovf_u2_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_u2_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ushort));
        }
    }

    public class i_conv_ovf_u4 : ConvertConvOvfInst
    {
        public i_conv_ovf_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_conv_ovf_u4_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_u4_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_conv_ovf_u8 : ConvertConvOvfInst
    {
        public i_conv_ovf_u8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ulong));
        }
    }

    public class i_conv_ovf_u8_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_u8_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ulong));
        }
    }

    public class i_conv_ovf_u : ConvertConvOvfInst
    {
        public i_conv_ovf_u(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_conv_ovf_u_un : ConvertConvOvfUnsInst
    {
        public i_conv_ovf_u_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_conv_r4 : ConvertConvInst
    {
        public i_conv_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(float));
        }
    }

    public class i_conv_r8 : ConvertConvInst
    {
        public i_conv_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(double));
        }
    }

    public class i_conv_r_un : ConvertUnsInst
    {
        public i_conv_r_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(float));
        }
    }

    public class i_conv_u1 : ConvertConvInst
    {
        public i_conv_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(byte));
        }
    }

    public class i_conv_u2 : ConvertConvInst
    {
        public i_conv_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ushort));
        }
    }

    public class i_conv_u4 : ConvertConvInst
    {
        public i_conv_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_conv_u8 : ConvertConvInst
    {
        public i_conv_u8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ulong));
        }
    }

    public class i_conv_u : ConvertConvInst
    {
        public i_conv_u(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_cpblk : INST
    {
        public i_cpblk(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_cpobj : INST
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

    public class i_dup : INST
    {
        public i_dup(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var rhs = state._stack.Pop();
            state._stack.Push(rhs);
            state._stack.Push(rhs);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var rhs = state._stack.Pop();
            state._stack.Push(rhs);
            state._stack.Push(rhs);
        }

    }

    public class i_endfilter : INST
    {
        public i_endfilter(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_endfinally : INST
    {
        public i_endfinally(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // leave.* page 372 of ecma 335
            var edges = Block._graph.SuccessorEdges(Block).ToList();
            if (edges.Count > 1)
                throw new Exception("There shouldn't be more than one edge from a leave instruction.");
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // leave.* page 372 of ecma 335
            var edge = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s = edge.To;
            // Build a branch to appease LLVM. CUDA does not seem to support exception handling.
            var br = LLVM.BuildBr(Builder, s.LlvmInfo.BasicBlock);
        }
    }

    public class i_initblk : INST
    {
        public i_initblk(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_initobj : INST
    {
        public i_initobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // initobj – initialize the value at an address page 400
            var dst = state._stack.Pop();
        }

        public override void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // initobj – initialize the value at an address page 400
            var dst = state._stack.Pop();
            var typetok = this.Operand as TypeReference;
            if (typetok == null) throw new Exception("Unknown operand for instruction: " + this.Instruction);
            if (typetok.IsStruct())
            {

            }
            else if (typetok.IsValueType)
            {

            }
            else
            {
                ValueRef nul = LLVM.ConstPointerNull(LLVM.PointerType(LLVM.VoidType(), 0));
                var v = new VALUE(nul);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(v);
                var zz = LLVM.BuildStore(Builder, v.V, dst.V);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("Store = " + new VALUE(zz).ToString());
            }
        }
    }

    public class i_isinst : INST
    {
        public i_isinst(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // isinst – test if an object is an instance of a class or interface, page 401
            var typetok = Operand;
            var tr = typetok as TypeReference;
            var tr2 = tr.RewriteMonoTypeReference();
            var v = tr2.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            state._stack.Pop();
            state._stack.Push(v);
        }

        public override void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // isinst – test if an object is an instance of a class or interface, page 401
            var typetok = Operand;
            var tr = typetok as TypeReference;
            var tr2 = tr.RewriteMonoTypeReference();
            var v = tr2.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            // No change for now.
        }
    }

    public class i_jmp : INST
    {
        public i_jmp(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldarg : ConvertLdArgInst
    {
        public i_ldarg(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }
    }

    public class i_ldarg_0 : ConvertLdArgInst
    {
        public i_ldarg_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 0;
        }
    }

    public class i_ldarg_1 : ConvertLdArgInst
    {
        public i_ldarg_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 1;
        }
    }

    public class i_ldarg_2 : ConvertLdArgInst
    {
        public i_ldarg_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 2;
        }
    }

    public class i_ldarg_3 : ConvertLdArgInst
    {

        public i_ldarg_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _arg = 3;
        }
    }

    public class i_ldarg_s : ConvertLdArgInst
    {
        public i_ldarg_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int ar = pr.Index;
            _arg = ar;
        }
    }

    public class i_ldarga : ConvertLdArgInst
    {
        public i_ldarga(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldarga_s : ConvertLdArgInst
    {
        public i_ldarga_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldc_i4 : ConvertLDCInstI4
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

    public class i_ldc_i4_0 : ConvertLDCInstI4
    {
        public i_ldc_i4_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldc_i4_1 : ConvertLDCInstI4
    {
        public i_ldc_i4_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_2 : ConvertLDCInstI4
    {
        public i_ldc_i4_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldc_i4_3 : ConvertLDCInstI4
    {
        public i_ldc_i4_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldc_i4_4 : ConvertLDCInstI4
    {
        public i_ldc_i4_4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 4;
            _arg = arg;
        }
    }

    public class i_ldc_i4_5 : ConvertLDCInstI4
    {
        public i_ldc_i4_5(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 5;
            _arg = arg;
        }
    }

    public class i_ldc_i4_6 : ConvertLDCInstI4
    {
        public i_ldc_i4_6(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 6;
            _arg = arg;
        }
    }

    public class i_ldc_i4_7 : ConvertLDCInstI4
    {
        public i_ldc_i4_7(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 7;
            _arg = arg;
        }
    }

    public class i_ldc_i4_8 : ConvertLDCInstI4
    {
        public i_ldc_i4_8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 8;
            _arg = arg;
        }
    }

    public class i_ldc_i4_m1 : ConvertLDCInstI4
    {
        public i_ldc_i4_m1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = -1;
            _arg = arg;
        }
    }

    public class i_ldc_i4_s : ConvertLDCInstI4
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

    public class i_ldc_i8 : ConvertLDCInstI8
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

    public class i_ldc_r4 : ConvertLDCInstR4
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

    public class i_ldc_r8 : ConvertLDCInstR8
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
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_ldelem_i2 : ConvertLoadElement
    {
        public i_ldelem_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_ldelem_i4 : ConvertLoadElement
    {
        public i_ldelem_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_ldelem_i8 : ConvertLoadElement
    {
        public i_ldelem_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(long));
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
            _dst = new TYPE(typeof(float));
        }
    }

    public class i_ldelem_r8 : ConvertLoadElement
    {
        public i_ldelem_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(double));
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
            _dst = new TYPE(typeof(byte));
        }
    }

    public class i_ldelem_u2 : ConvertLoadElement
    {
        public i_ldelem_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(ushort));
        }
    }

    public class i_ldelem_u4 : ConvertLoadElement
    {
        public i_ldelem_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_ldelema : ConvertLoadElementA
    {
        public i_ldelema(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldfld : INST
    {
        public i_ldfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // ldfld, page 406 of ecma 335
            var v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());
            var operand = this.Instruction.Operand;
            var field = operand as Mono.Cecil.FieldReference;
            if (field == null) throw new Exception("Cannot convert ldfld.");
            var value = field.FieldType.InstantiateGeneric(this.Block._original_method_reference);
            state._stack.Push(value);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // ldfld, page 405 of ecma 335
            VALUE v = state._stack.Pop();
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
                    declaring_type_tr = this.Block._original_method_reference.DeclaringType;
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
                        field_size = BUFFERS.SizeOf(typeof(IntPtr));
                        alignment = BUFFERS.Alignment(typeof(IntPtr));
                    }
                    else
                    {
                        var ft = f.FieldType.ToSystemType();
                        field_size = BUFFERS.SizeOf(ft);
                        alignment = BUFFERS.Alignment(ft);
                    }

                    int padding = BUFFERS.Padding(size, alignment);
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

                var addr = LLVM.BuildStructGEP(Builder, v.V, offset, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(addr));

                var load = LLVM.BuildLoad(Builder, addr, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(load));


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

                var load_value = new VALUE(load);
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

                state._stack.Push(new VALUE(load));
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
                        field_size = BUFFERS.SizeOf(typeof(IntPtr));
                        alignment = BUFFERS.Alignment(typeof(IntPtr));
                    }
                    else
                    {
                        field_size = BUFFERS.SizeOf(ft);
                        alignment = BUFFERS.Alignment(ft);
                    }

                    int padding = BUFFERS.Padding(size, alignment);
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

                var load = LLVM.BuildExtractValue(Builder, v.V, offset, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(load));

                bool xInt = LLVM.GetTypeKind(tt) == TypeKind.IntegerTypeKind;
                bool xP = LLVM.GetTypeKind(tt) == TypeKind.PointerTypeKind;
                bool xA = LLVM.GetTypeKind(tt) == TypeKind.ArrayTypeKind;

                // If load result is a pointer, then cast it to proper type.
                // This is because I had to avoid recursive data types in classes
                // as LLVM cannot handle these at all. So, all pointer types
                // were defined as void* in the LLVM field.

                var load_value = new VALUE(load);
                bool isPtrLoad = load_value.T.isPointerTy();
                if (isPtrLoad)
                {
                    var mono_field_type = field.FieldType;
                    TypeRef type = mono_field_type.ToTypeRef();
                    load = LLVM.BuildBitCast(Builder,
                        load, type, "i" + instruction_id++);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(load));
                }

                state._stack.Push(new VALUE(load));
            }
        }
    }

    public class i_ldflda : INST
    {
        public i_ldflda(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // ldflda, page 407 of ecma 335
            var v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());
            var operand = this.Instruction.Operand;
            var field = operand as Mono.Cecil.FieldReference;
            if (field == null) throw new Exception("Cannot convert ldfld.");
            var type = field.FieldType.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            var value = new ByReferenceType(type);
            state._stack.Push(value);
        }
    }

    public class i_ldftn : INST
    {
        public i_ldftn(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            state._stack.Push(typeof(System.UInt32).ToMonoTypeReference());
        }
    }

    public class i_ldind_i1 : ConvertLoadIndirect
    {
        public i_ldind_i1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_ldind_i2 : ConvertLoadIndirect
    {
        public i_ldind_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_ldind_i4 : ConvertLoadIndirect
    {
        public i_ldind_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_ldind_i8 : ConvertLoadIndirect
    {
        public i_ldind_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(long));
        }
    }

    public class i_ldind_i : ConvertLoadIndirect
    {
        public i_ldind_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
			_dst = new TYPE(typeof(int));
        }
    }

    public class i_ldind_r4 : ConvertLoadIndirect
    {
        public i_ldind_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(float));
        }
    }

    public class i_ldind_r8 : ConvertLoadIndirect
    {
        public i_ldind_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(double));
        }
    }

    public class i_ldind_ref : ConvertLoadIndirect
    {
        public i_ldind_ref(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
			_dst = new TYPE(typeof(object));
        }
    }

    public class i_ldind_u1 : ConvertLoadIndirect
    {
        public i_ldind_u1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(byte));
        }
    }

    public class i_ldind_u2 : ConvertLoadIndirect
    {
        public i_ldind_u2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
			_dst = new TYPE(typeof(ushort));
        }
    }

    public class i_ldind_u4 : ConvertLoadIndirect
    {
        public i_ldind_u4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(uint));
        }
    }

    public class i_ldlen : INST
    {
        public i_ldlen(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var a = state._stack.Pop();
            state._stack.Push(typeof(System.UInt32).ToMonoTypeReference());
        }

        // For array implementation, see https://www.codeproject.com/Articles/3467/Arrays-UNDOCUMENTED

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            //VALUE v = state._stack.Pop();
            //if (Campy.Utils.Options.IsOn("jit_trace"))
            //    System.Console.WriteLine(v);

            //var load = v.V;
            //load = LLVM.BuildLoad(Builder, load, "i" + instruction_id++);
            //if (Campy.Utils.Options.IsOn("jit_trace"))
            //    System.Console.WriteLine(new VALUE(load));

            // The length of an array is the product of all dimensions, but this instruction
            // is only used for 1d arrays.

            //// Load len.
            //load = LLVM.BuildExtractValue(Builder, load, 2, "i" + instruction_id++);
            //if (Campy.Utils.Options.IsOn("jit_trace"))
            //    System.Console.WriteLine(new VALUE(load));

            //load = LLVM.BuildTrunc(Builder, load, LLVM.Int32Type(), "i" + instruction_id++);
            //if (Campy.Utils.Options.IsOn("jit_trace"))
            //    System.Console.WriteLine(new VALUE(load));

            {
                // Call PTX method.

                var ret = true;
                var HasScalarReturnValue = true;
                var HasStructReturnValue = false;
                var HasThis = true;
                var NumberOfArguments = 0
                                      + (HasThis ? 1 : 0)
                                      + (HasStructReturnValue ? 1 : 0);
                int locals = 0;
                var NumberOfLocals = locals;
                int xret = (HasScalarReturnValue || HasStructReturnValue) ? 1 : 0;
                int xargs = NumberOfArguments;

                BuilderRef bu = this.Builder;

                string demangled_name = "_Z31System_Array_Internal_GetLengthPhS_S_";
                string full_name = "System.Int32 System.Array::Internal_GetLength()";
                // Find the specific function called.
                var xx = RUNTIME._bcl_runtime_csharp_internal_to_valueref.Where(
                    t =>
                        t.Key.Contains(demangled_name)
                         || demangled_name.Contains(t.Key));
                var first_kv_pair = xx.FirstOrDefault();
                ValueRef fv = first_kv_pair.Value;
                var t_fun = LLVM.TypeOf(fv);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);

                RUNTIME.BclNativeMethod mat = null;
                foreach (RUNTIME.BclNativeMethod ci in RUNTIME.BclNativeMethods)
                {
                    if (ci._full_name == full_name)
                    {
                        mat = ci;
                        break;
                    }
                }

                {
                    ValueRef[] args = new ValueRef[3];

                    // Set up "this".
                    ValueRef nul = LLVM.ConstPointerNull(LLVM.PointerType(LLVM.VoidType(), 0));
                    VALUE t = new VALUE(nul);

                    // Pop all parameters and stuff into params buffer. Note, "this" and
                    // "return" are separate parameters in GPU BCL runtime C-functions,
                    // unfortunately, reminates of the DNA runtime I decided to use.
                    var entry = this.Block.Entry.LlvmInfo.BasicBlock;
                    var beginning = LLVM.GetFirstInstruction(entry);
                    //LLVM.PositionBuilderBefore(Builder, beginning);
                    var parameter_type = LLVM.ArrayType(LLVM.Int64Type(), (uint)0);
                    var param_buffer = LLVM.BuildAlloca(Builder, parameter_type, "i" + instruction_id++);
                    LLVM.SetAlignment(param_buffer, 64);
                    //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);
                    var base_of_parameters = LLVM.BuildPointerCast(Builder, param_buffer,
                        LLVM.PointerType(LLVM.Int64Type(), 0), "i" + instruction_id++);

                    if (HasThis)
                    {
                        t = state._stack.Pop();
                        var ll = t.V;
                        //ll = LLVM.BuildLoad(Builder, ll, "i" + instruction_id++);
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new VALUE(ll));
                        t = new VALUE(ll);
                    }

                    // Set up return. For now, always allocate buffer.
                    // Note function return is type of third parameter.
                    var return_type = mat._returnType.ToTypeRef();
                    var return_buffer = LLVM.BuildAlloca(Builder, return_type, "i" + instruction_id++);
                    LLVM.SetAlignment(return_buffer, 64);
                    //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);

                    // Set up call.
                    var pt = LLVM.BuildPtrToInt(Builder, t.V, LLVM.Int64Type(), "i" + instruction_id++);
                    var pp = LLVM.BuildPtrToInt(Builder, param_buffer, LLVM.Int64Type(), "i" + instruction_id++);
                    var pr = LLVM.BuildPtrToInt(Builder, return_buffer, LLVM.Int64Type(), "i" + instruction_id++);

                    args[0] = pt;
                    args[1] = pp;
                    args[2] = pr;

                    var call = LLVM.BuildCall(Builder, fv, args, "");

                    if (ret)
                    {
                        var load = LLVM.BuildLoad(Builder, return_buffer, "i" + instruction_id++);
                        //var load = LLVM.ConstInt(LLVM.Int32Type(), 11, false);
                        state._stack.Push(new VALUE(load));
                    }

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(call.ToString());
                }
            }

            //state._stack.Push(new VALUE(load));
        }
    }

    public class i_ldloc : ConvertLdLoc
    {
        public i_ldloc(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloc_0 : ConvertLdLoc
    {
        public i_ldloc_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_ldloc_1 : ConvertLdLoc
    {
        public i_ldloc_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_ldloc_2 : ConvertLdLoc
    {
        public i_ldloc_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_ldloc_3 : ConvertLdLoc
    {
        public i_ldloc_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_ldloc_s : ConvertLdLoc
    {
        public i_ldloc_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca : ConvertLdLoc
    {
        public i_ldloca(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldloca_s : ConvertLdLoc
    {
        public i_ldloca_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableDefinition pr = i.Operand as Mono.Cecil.Cil.VariableDefinition;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_ldnull : INST
    {
        public i_ldnull(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            state._stack.Push(typeof(System.Object).ToMonoTypeReference());
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            ValueRef nul = LLVM.ConstPointerNull(LLVM.PointerType(LLVM.VoidType(), 0));
            var v = new VALUE(nul);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v);
            state._stack.Push(v);
        }
    }

    public class i_ldobj : INST
    {
        public i_ldobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // ldobj, copy a value from an address to the stack, ecma 335, page 409
            var v = state._stack.Pop();
            object operand = this.Operand;
            var o = operand as TypeReference;
            o = o.RewriteMonoTypeReference();
            var p = o.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            state._stack.Push(p);
        }
    }

    public class i_ldsfld : INST
    {
        public i_ldsfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // ldsfld (load static field), ecma 335 page 410
            var operand = this.Operand;
            var operand_field_reference = operand as FieldReference;
            if (operand_field_reference == null)
                throw new Exception("Unknown field type");
            var ft = operand_field_reference.FieldType;
            state._stack.Push(ft);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // ldsfld (load static field), ecma 335 page 410
            var operand = this.Operand;
            var mono_field_reference = operand as FieldReference;
            if (mono_field_reference == null)
                throw new Exception("Unknown field type");
            var type = mono_field_reference.DeclaringType;
            type = type.RewriteMonoTypeReference();
            type = type.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            var mono_field_type = mono_field_reference.FieldType;
            mono_field_type = mono_field_type.RewriteMonoTypeReference();
            var llvm_field_type = mono_field_type.ToTypeRef();
            // Call meta to get static field. This can be done now because
            // the address of the static field does not change.
            var bcl_type = RUNTIME.GetBclType(type);
            if (bcl_type == IntPtr.Zero) throw new Exception();
            IntPtr[] fields = null;
            IntPtr* buf;
            int len;
            RUNTIME.BclGetFields(bcl_type, &buf, &len);
            fields = new IntPtr[len];
            for (int i = 0; i < len; ++i) fields[i] = buf[i];
            var mono_fields = type.ResolveFields().ToArray();
            var find = fields.Where(f =>
            {
                var ptrName = RUNTIME.BclGetFieldName(f);
                string name = Marshal.PtrToStringAnsi(ptrName);
                return name == mono_field_reference.Name;
            });
            IntPtr first = find.FirstOrDefault();
            if (first == IntPtr.Zero) throw new Exception("Cannot find field--ldsfld");
            var ptr = RUNTIME.BclGetStaticField(first);
            bool isArr = mono_field_type.IsArray;
            bool isSt = mono_field_type.IsStruct();
            bool isRef = mono_field_type.IsReferenceType();
            if (Campy.Utils.Options.IsOn("jit_trace"))
            System.Console.WriteLine(LLVM.PrintTypeToString(llvm_field_type));
            var ptr_llvm_field_type = LLVM.PointerType(llvm_field_type, 0);
            var address = LLVM.ConstInt(LLVM.Int64Type(), (ulong)ptr, false);
            var f1 = LLVM.BuildIntToPtr(Builder, address, ptr_llvm_field_type, "i" + instruction_id++);
            var load = LLVM.BuildLoad(Builder, f1, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(load));
            state._stack.Push(new VALUE(load));
        }
    }

    public class i_ldsflda : INST
    {
        public i_ldsflda(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_ldstr : INST
    {
        public i_ldstr(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = typeof(string).ToMonoTypeReference();
            
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());

            state._stack.Push(v);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            // Call SystemString_FromCharPtrASCII and push new string object on the stack.
            // _Z29SystemString_FromCharPtrASCIIPc

            unsafe {
                ValueRef[] args = new ValueRef[1];

                // Get char string froom instruction.
                var operand = Operand;
                string str = (string)operand;

                var llvm_cstr_t = LLVM.BuildGlobalString(Builder, str, "i" + instruction_id++);
                var llvm_cstr = LLVM.BuildPtrToInt(Builder, llvm_cstr_t, LLVM.Int64Type(), "i" + instruction_id++);
                args[0] = llvm_cstr;
                string name = "_Z29SystemString_FromCharPtrASCIIPc";
                var list = RUNTIME.BclNativeMethods.ToList();
                var list2 = RUNTIME.PtxFunctions.ToList();
                var f = list2.Where(t => t._mangled_name == name).First();
                ValueRef fv = f._valueref;
                var call = LLVM.BuildCall(Builder, fv, args, "");

                // Find type of System.String in BCL.
                Mono.Cecil.TypeReference tr = RUNTIME.FindBCLType(typeof(System.String));
                var llvm_type = tr.ToTypeRef();

                // Convert to pointer to pointer of string.
                var cast = LLVM.BuildIntToPtr(Builder, call, llvm_type, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(cast));

                state._stack.Push(new VALUE(cast));
            }
        }
    }

    public class i_ldtoken : INST
    {
        public i_ldtoken(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // ldtoken (load token handle), ecma 335 page 413
            var rth = typeof(System.RuntimeTypeHandle).ToMonoTypeReference().RewriteMonoTypeReference();
            var v = rth.Deresolve(this.Block._original_method_reference.DeclaringType, null);
			state._stack.Push(v);
            // Parse System.RuntimeTypeHandle.ctor(IntPtr). We'll make
            // a call to that in code generation.
            var list = rth.Resolve().Methods.Where(m => m.FullName.Contains("ctor")).ToList();
            if (list.Count() != 1) throw new Exception("There should be only one constructor for System.RuntimeTypeHandle.");
            var mr = list.First();
            IMPORTER.Singleton().Add(mr);
        }

        public override void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // ldtoken (load token handle), ecma 335 page 413
            // Load System.RuntimeTypeHandle for arg.
            var arg = this.Operand as TypeReference;
            if (arg == null) throw new Exception("Cannot parse ldtoken type for whatever reason.");
            var arg2 = arg.RewriteMonoTypeReference();
            var v = arg2.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            var meta = RUNTIME.GetBclType(v);
            var rth = typeof(System.RuntimeTypeHandle).ToMonoTypeReference().RewriteMonoTypeReference();
            var type = rth.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            var llvm_type = type.ToTypeRef();
            ValueRef new_obj;
            {
                // Generate code to allocate object and stuff.
                var xx1 = RUNTIME.BclNativeMethods.ToList();
                var xx2 = RUNTIME.PtxFunctions.ToList();
                var xx = xx2
                    .Where(t => { return t._mangled_name == "_Z23Heap_AllocTypeVoidStarsPv"; });
                var xxx = xx.ToList();
                RUNTIME.PtxFunction first_kv_pair = xx.FirstOrDefault();
                if (first_kv_pair == null)
                    throw new Exception("Yikes.");
                ValueRef fv2 = first_kv_pair._valueref;
                ValueRef[] args = new ValueRef[1];
                args[0] = LLVM.ConstInt(LLVM.Int64Type(), (ulong)meta.ToInt64(), false);
                var call = LLVM.BuildCall(Builder, fv2, args, "i" + instruction_id++);
                var cast = LLVM.BuildIntToPtr(Builder, call, llvm_type, "i" + instruction_id++);
                new_obj = cast;
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(new_obj));
                state._stack.Push(new VALUE(new_obj));
            }
            state._stack.Push(new VALUE(LLVM.ConstInt(LLVM.Int64Type(), (ulong)meta.ToInt64(), false)));
            {
                var t = typeof(System.RuntimeTypeHandle).ToMonoTypeReference().RewriteMonoTypeReference();
                var list = rth.Resolve().Methods.Where(m => m.FullName.Contains("ctor")).ToList();
                if (list.Count() != 1) throw new Exception("There should be only one constructor for System.RuntimeTypeHandle.");
                var mr = list.First();
                // Find bb entry.
                CFG.Vertex entry_corresponding_to_method_called = this.Block._graph.Vertices.Where(node
                    =>
                {
                    if (node.IsEntry && JITER.MethodName(node._original_method_reference) == mr.FullName)
                        return true;
                    return false;
                }).ToList().FirstOrDefault();
                if (entry_corresponding_to_method_called == null) throw new Exception("Cannot find constructor for System.RuntimeTypeHandle.");

                int xret = (entry_corresponding_to_method_called.HasScalarReturnValue || entry_corresponding_to_method_called.HasStructReturnValue) ? 1 : 0;
                int xargs = entry_corresponding_to_method_called.StackNumberOfArguments;
                var name = JITER.MethodName(mr);
                BuilderRef bu = this.Builder;
                ValueRef fv = entry_corresponding_to_method_called.LlvmInfo.MethodValueRef;
                var t_fun = LLVM.TypeOf(fv);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");
                ValueRef[] args = new ValueRef[xargs];
                // No return.
                for (int k = xargs - 1; k >= 0; --k)
                {
                    VALUE a = state._stack.Pop();
                    ValueRef par = LLVM.GetParam(fv, (uint)k);
                    ValueRef value = a.V;
                    if (LLVM.TypeOf(value) != LLVM.TypeOf(par))
                    {
                        if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.StructTypeKind
                            && LLVM.GetTypeKind(LLVM.TypeOf(value)) == TypeKind.PointerTypeKind)
                        {
                            value = LLVM.BuildLoad(Builder, value, "i" + instruction_id++);
                        }
                        else if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.PointerTypeKind)
                        {
                            value = LLVM.BuildPointerCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                        }
                        else
                        {
                            value = LLVM.BuildBitCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                        }
                    }
                    args[k] = value;
                }
                var call = LLVM.BuildCall(Builder, fv, args, "");
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(call.ToString());
                state._stack.Push(new VALUE(new_obj));
            }
        }
    }

    public class i_ldvirtftn : INST
    {
        public i_ldvirtftn(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_leave : INST
    {
        public i_leave(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // leave.* page 372 of ecma 335
            var edges = Block._graph.SuccessorEdges(Block).ToList();
            if (edges.Count > 1)
                throw new Exception("There shouldn't be more than one edge from a leave instruction.");
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // leave.* page 372 of ecma 335
            var edge = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s = edge.To;
            var br = LLVM.BuildBr(Builder, s.LlvmInfo.BasicBlock);
        }
    }

    public class i_leave_s : INST
    {
        public i_leave_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // leave.* page 372 of ecma 335
            var edges = Block._graph.SuccessorEdges(Block).ToList();
            if (edges.Count > 1)
                throw new Exception("There shouldn't be more than one edge from a leave instruction.");
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // leave.* page 372 of ecma 335
            var edge = Block._graph.SuccessorEdges(Block).ToList()[0];
            var s = edge.To;
            var br = LLVM.BuildBr(Builder, s.LlvmInfo.BasicBlock);
        }
    }

    public class i_localloc : INST
    {
        public i_localloc(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_mkrefany : INST
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

    public class i_neg : INST
    {
        public i_neg(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var v = state._stack.Pop();

            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());

            state._stack.Push(v);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var @typeof = LLVM.TypeOf(rhs.V);
            var kindof = LLVM.GetTypeKind(@typeof);
            ValueRef neg;
            if (kindof == TypeKind.DoubleTypeKind || kindof == TypeKind.FloatTypeKind)
                neg = LLVM.BuildFNeg(Builder, rhs.V, "i" + instruction_id++);
            else
                neg = LLVM.BuildNeg(Builder, rhs.V, "i" + instruction_id++);

            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(neg));

            state._stack.Push(new VALUE(neg));
        }
    }

    public class i_newarr : INST
    {
        public i_newarr(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // newarr, page 416 of ecma 335
            var v = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(v.ToString());
            object operand = this.Operand;
            TypeReference type = operand as TypeReference;
            type = type.RewriteMonoTypeReference();
            var actual_element_type = type.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            TypeReference new_array_type = new ArrayType(actual_element_type, 1 /* 1D array */);
            state._stack.Push(new_array_type);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // newarr, page 416 of ecma 335
            object operand = this.Operand;
            TypeReference element_type = operand as TypeReference;
            element_type = element_type.RewriteMonoTypeReference();
            element_type = element_type.Deresolve(this.Block._original_method_reference.DeclaringType, null);

            TypeReference new_array_type = new ArrayType(element_type, 1 /* 1D array */);
            var meta = RUNTIME.GetBclType(new_array_type);
            var array_type_to_create = new_array_type.ToTypeRef();
            var xx2 = RUNTIME.PtxFunctions.ToList();
            var xx = xx2.Where(t => { return t._mangled_name == "_Z21SystemArray_NewVectorP12tMD_TypeDef_jPj"; });
            RUNTIME.PtxFunction first_kv_pair = xx.FirstOrDefault();
            if (first_kv_pair == null) throw new Exception("Yikes.");
            ValueRef fv2 = first_kv_pair._valueref;
            ValueRef[] args = new ValueRef[3];
            var length_buffer = LLVM.BuildAlloca(Builder, LLVM.ArrayType(LLVM.Int32Type(), (uint)1), "i" + instruction_id++);
            LLVM.SetAlignment(length_buffer, 64);
            var base_of_lengths = LLVM.BuildPointerCast(Builder, length_buffer, LLVM.PointerType(LLVM.Int32Type(), 0), "i" + instruction_id++);
            int rank = 1;
            for (int i = 0; i < rank; ++i)
            {
                VALUE len = state._stack.Pop();
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(len);
                ValueRef[] id = new ValueRef[1] { LLVM.ConstInt(LLVM.Int32Type(), (ulong)i, true) };
                var add = LLVM.BuildInBoundsGEP(Builder, base_of_lengths, id, "i" + instruction_id++);
                var lcast = LLVM.BuildIntCast(Builder, len.V, LLVM.Int32Type(), "i" + instruction_id++);
                ValueRef store = LLVM.BuildStore(Builder, lcast, add);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(store));
            }
            args[2] = LLVM.BuildPtrToInt(Builder, length_buffer, LLVM.Int64Type(), "i" + instruction_id++);
            args[1] = LLVM.ConstInt(LLVM.Int32Type(), (ulong)rank, false);
            args[0] = LLVM.ConstInt(LLVM.Int64Type(), (ulong)meta, false);
            var call = LLVM.BuildCall(Builder, fv2, args, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(call));
            var new_obj = LLVM.BuildIntToPtr(Builder, call, array_type_to_create, "i" + instruction_id++);
            var stack_result = new VALUE(new_obj);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(stack_result);
            state._stack.Push(stack_result);
        }
    }

    public class i_newobj : INST
    {
        public i_newobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            INST new_inst = this;

            // Successor is fallthrough.
            object method = this.Operand;

            if (method as Mono.Cecil.MethodReference == null)
                throw new Exception();

            Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
            int xargs = /* always pass "this", but it does not count because newobj
                creates the object. So, it is just the number of standard parameters
                of the contructor. */
                mr.Parameters.Count;

            var fn = mr.GetType().FullName;

            // We need to make sure the declaring type is completely de-resolved.

            List<TypeReference> args = new List<TypeReference>();
            for (int k = 0; k < xargs; ++k)
            {
                var v = state._stack.Pop();
                args.Insert(0, v);
            }
            var args_array = args.ToArray();
            if (mr.DeclaringType.ContainsGenericParameter)
            {
                var dt = mr.DeclaringType.Deresolve(this.Block._original_method_reference.DeclaringType, null);
                mr = mr.Deresolve(dt, args_array);
            }

            if (mr.DeclaringType == null)
                throw new Exception("can't handle.");
            if (mr.DeclaringType.HasGenericParameters)
                throw new Exception("can't handle.");
            if (mr.ReturnType.FullName != "System.Void")
            {
                throw new Exception(
                    "Constructor has a return type, but they should never have a type. Something is wrong.");
            }
            state._stack.Push(mr.DeclaringType);
            IMPORTER.Singleton().Add(mr);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            // The JIT of a call instructure requires a little explanation. The operand
            // for the instruction is a MethodReference, which is a C# method of some type.
            // Note, however, there are two cases here. One case is that the method has
            // CLI code that implements the method. The other are those that are DLL references.
            // These have no code that Mono.Cecil can pick up as it is usally C-code, which
            // is compiled for a specific target. The function signature of the native function
            // is not necessarily the same as that declared in the C#/NET assembly. This method,
            // Convert(), needs to handle native function calls carefully. These functions will
            // create native structures that C# references.

            // Get some basic information about the instruction, method, and type of object to create.
            var inst = this;
            object operand = this.Operand;
            MethodReference method = operand as MethodReference;
            CFG graph = (CFG)this.Block._graph;
            TypeReference type = method.DeclaringType;
            if (type == null)
                throw new Exception("Cannot get type of object/value for newobj instruction.");
            bool is_type_value_type = type.IsValueType;
            var name = JITER.MethodName(method);
            CFG.Vertex the_entry = this.Block._graph.Vertices.Where(node
                =>
            {
                var g = inst.Block._graph;
                CFG.Vertex v = node;
                JITER c = JITER.Singleton;
                if (v.IsEntry && JITER.MethodName(v._original_method_reference) == name)
                    return true;
                else return false;
            }).ToList().FirstOrDefault();
            var llvm_type = type.ToTypeRef();
            var td = type.Resolve();

            // There four basic cases for newobj:
            // 1) type is a value type
            //   The object must be allocated on the stack, and the contrustor called with a pointer to that.
            //   a) the_entry is null, which means the constructor is a C function.
            //   b) the_entry is NOT null, which means the constructor is further CIL code.
            // 2) type is a reference_type.
            //   The object will be allocated on the heap, but done according to a convention of DNA.
            //   b) the_entry is null, which means the constructor is a C function, and it performs the allocation.
            //   c) the_entry is NOT null, which means we must allocate the object, then call the constructor, which is further CIL code.
            if (is_type_value_type && the_entry == null)
            {

            }
            else if (is_type_value_type && the_entry != null)
            {
                int nargs = the_entry.StackNumberOfArguments;
                int ret = the_entry.HasScalarReturnValue ? 1 : 0;

                // First, create a struct.
                var entry = this.Block.Entry.LlvmInfo.BasicBlock;
                var beginning = LLVM.GetFirstInstruction(entry);
                //LLVM.PositionBuilderBefore(Builder, beginning);
                var new_obj = LLVM.BuildAlloca(Builder, llvm_type, "i" + instruction_id++); // Allocates struct on stack, but returns a pointer to struct.
                //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(new_obj));

                BuilderRef bu = this.Builder;
                ValueRef fv = the_entry.LlvmInfo.MethodValueRef;
                var t_fun = LLVM.TypeOf(fv);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");

                // Set up args, type casting if required.
                ValueRef[] args = new ValueRef[nargs];
                for (int k = nargs - 1; k >= 1; --k)
                {
                    VALUE v = state._stack.Pop();
                    ValueRef par = LLVM.GetParam(fv, (uint)k);
                    ValueRef value = v.V;
                    if (LLVM.TypeOf(value) != LLVM.TypeOf(par))
                    {
                        if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.PointerTypeKind)
                        {
                            value = LLVM.BuildPointerCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                        }
                        else
                        {
                            value = LLVM.BuildBitCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                        }
                    }
                    args[k] = value;
                }
                args[0] = new_obj;

                var call = LLVM.BuildCall(Builder, fv, args, "");
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(call));

                var load = LLVM.BuildLoad(Builder, new_obj, "i" + instruction_id++);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(load));

                state._stack.Push(new VALUE(load));
            }
            else if (!is_type_value_type && the_entry == null)
            {
                // As noted in JIT_execute.c code of BCL:
                // "All internal constructors MUST allocate their own 'this' objects"
                // So, we don't call any allocator here, just the internal function in the BCL,
                // as that function will do the allocation over on the GPU.
                Mono.Cecil.MethodReturnType cs_method_return_type_aux = method.MethodReturnType;
                Mono.Cecil.TypeReference cs_method_return_type = cs_method_return_type_aux.ReturnType;
                var cs_has_ret = cs_method_return_type.FullName != "System.Void";
                var cs_HasScalarReturnValue = cs_has_ret && !cs_method_return_type.IsStruct();
                var cs_HasStructReturnValue = cs_has_ret && cs_method_return_type.IsStruct();
                var cs_HasThis = method.HasThis;
                var cs_NumberOfArguments = method.Parameters.Count
                                        + (cs_HasThis ? 1 : 0)
                                        + (cs_HasStructReturnValue ? 1 : 0);
                int locals = 0;
                var NumberOfLocals = locals;
                int cs_xret = (cs_HasScalarReturnValue || cs_HasStructReturnValue) ? 1 : 0;
                int cs_xargs = cs_NumberOfArguments;

                // Search for native function in loaded libraries.
                name = method.Name;
                var full_name = method.FullName;
                Regex regex = new Regex(@"^[^\s]+\s+(?<name>[^\(]+).+$");
                Match m = regex.Match(full_name);
                if (!m.Success) throw new Exception();
                var demangled_name = m.Groups["name"].Value;
                demangled_name = demangled_name.Replace("::", "_");
                demangled_name = demangled_name.Replace(".", "_");
                demangled_name = demangled_name.Replace("__", "_");
                BuilderRef bu = this.Builder;
                var as_name = method.Module.Assembly.Name;
                var xx = RUNTIME.BclNativeMethods
                    .Where(t =>
                    {
                        return t._full_name == full_name;
                    });
                var xxx = xx.ToList();
                RUNTIME.BclNativeMethod first_kv_pair = xx.FirstOrDefault();
                if (first_kv_pair == null)
                    throw new Exception("Yikes.");

                RUNTIME.PtxFunction fffv = RUNTIME.PtxFunctions.Where(
                    t => first_kv_pair._native_name.Contains(t._mangled_name)).FirstOrDefault();
                ValueRef fv = fffv._valueref;
                var t_fun = LLVM.TypeOf(fv);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);

                {
                    ValueRef[] args = new ValueRef[3];

                    // Set up "this".
                    ValueRef nul = LLVM.ConstPointerNull(LLVM.PointerType(LLVM.VoidType(), 0));
                    VALUE t = new VALUE(nul);

                    // Pop all parameters and stuff into params buffer. Note, "this" and
                    // "return" are separate parameters in GPU BCL runtime C-functions,
                    // unfortunately, reminates of the DNA runtime I decided to use.
                    var entry = this.Block.Entry.LlvmInfo.BasicBlock;
                    var beginning = LLVM.GetFirstInstruction(entry);
                    //LLVM.PositionBuilderBefore(Builder, beginning);
                    var parameter_type = LLVM.ArrayType(
                        LLVM.Int64Type(),
                        (uint)method.Parameters.Count);
                    var param_buffer = LLVM.BuildAlloca(Builder, parameter_type, "i"+instruction_id++);
                    LLVM.SetAlignment(param_buffer, 64);
                    var base_of_parameters = LLVM.BuildPointerCast(Builder, param_buffer,
                        LLVM.PointerType(LLVM.Int64Type(), 0), "i" + instruction_id++);

                    for (int i = method.Parameters.Count - 1; i >= 0; i--)
                    {
                        VALUE p = state._stack.Pop();
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(p);
                        ValueRef[] index = new ValueRef[1] { LLVM.ConstInt(LLVM.Int32Type(), (ulong)i, true) };
                        var add = LLVM.BuildInBoundsGEP(Builder, base_of_parameters, index, "i" + instruction_id++);
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new VALUE(add));
                        ValueRef v = LLVM.BuildPointerCast(Builder, add, LLVM.PointerType(LLVM.TypeOf(p.V), 0), "i" + instruction_id++);
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new VALUE(v));
                        ValueRef store = LLVM.BuildStore(Builder, p.V, v);
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(new VALUE(store));
                    }

                    // Set up return. For now, always allocate buffer.
                    // Note function return is type of third parameter.
                    var native_return_type2 = first_kv_pair._returnType.ToTypeRef();

                    var native_return_type = LLVM.ArrayType(
                       LLVM.Int64Type(),
                       (uint)1);
                    var native_return_buffer = LLVM.BuildAlloca(Builder,
                        native_return_type, "i" + instruction_id++);
                    LLVM.SetAlignment(native_return_buffer, 64);
                    //LLVM.PositionBuilderAtEnd(Builder, this.Block.BasicBlock);

                    // Set up call.
                    var pt = LLVM.BuildPointerCast(Builder, t.V,
                        LLVM.PointerType(LLVM.VoidType(), 0), "i" + instruction_id++);
                    var pp = LLVM.BuildPointerCast(Builder, param_buffer,
                        LLVM.PointerType(LLVM.VoidType(), 0), "i" + instruction_id++);
                    var pr = LLVM.BuildPointerCast(Builder, native_return_buffer,
                        LLVM.PointerType(LLVM.VoidType(), 0), "i" + instruction_id++);

                    args[0] = pt;
                    args[1] = pp;
                    args[2] = pr;

                    var call = LLVM.BuildCall(Builder, fv, args, name);
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(call));

                    // There is always a return from a newobj instruction.
                    var ptr_cast = LLVM.BuildBitCast(Builder,
                        native_return_buffer,
                        LLVM.PointerType(llvm_type, 0), "i" + instruction_id++);

                    var load = LLVM.BuildLoad(Builder, ptr_cast, "i" + instruction_id++);

                    // Cast the damn object into the right type.
                    state._stack.Push(new VALUE(load));

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(load));
                }
            }
            else if (!is_type_value_type && the_entry != null)
            {
                ValueRef new_obj;
                {
                    var meta = RUNTIME.GetBclType(type);

                    // Generate code to allocate object and stuff.
                    // This boxes the value.
                    var xx1 = RUNTIME.BclNativeMethods.ToList();
                    var xx2 = RUNTIME.PtxFunctions.ToList();
                    var xx = xx2
                        .Where(t => { return t._mangled_name == "_Z23Heap_AllocTypeVoidStarsPv"; });
                    var xxx = xx.ToList();
                    RUNTIME.PtxFunction first_kv_pair = xx.FirstOrDefault();
                    if (first_kv_pair == null)
                        throw new Exception("Yikes.");

                    ValueRef fv2 = first_kv_pair._valueref;
                    ValueRef[] args = new ValueRef[1];

                    args[0] = LLVM.ConstInt(LLVM.Int64Type(), (ulong) meta.ToInt64(), false);
                    var call = LLVM.BuildCall(Builder, fv2, args, "i" + instruction_id++);
                    var cast = LLVM.BuildIntToPtr(Builder, call, llvm_type, "i" + instruction_id++);
                    new_obj = cast;

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(new_obj));
                }

                {
                    int nargs = the_entry.StackNumberOfArguments;
                    int ret = the_entry.HasScalarReturnValue ? 1 : 0;

                    BuilderRef bu = this.Builder;
                    ValueRef fv = the_entry.LlvmInfo.MethodValueRef;
                    var t_fun = LLVM.TypeOf(fv);
                    var t_fun_con = LLVM.GetTypeContext(t_fun);
                    var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                    if (t_fun_con != context) throw new Exception("not equal");

                    // Set up args, type casting if required.
                    ValueRef[] args = new ValueRef[nargs];
                    for (int k = nargs - 1; k >= 1; --k)
                    {
                        VALUE v = state._stack.Pop();
                        ValueRef par = LLVM.GetParam(fv, (uint)k);
                        ValueRef value = v.V;
                        if (LLVM.TypeOf(value) != LLVM.TypeOf(par))
                        {
                            if (LLVM.GetTypeKind(LLVM.TypeOf(par)) == TypeKind.PointerTypeKind)
                            {
                                value = LLVM.BuildPointerCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                            else
                            {
                                value = LLVM.BuildBitCast(Builder, value, LLVM.TypeOf(par), "i" + instruction_id++);
                            }
                        }
                        args[k] = value;
                    }
                    args[0] = new_obj;

                    var call = LLVM.BuildCall(Builder, fv, args, "");
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(new VALUE(call));

                    state._stack.Push(new VALUE(new_obj));
                }
            }
        }
    }

    public class i_no : INST
    {
        public i_no(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_nop : INST
    {
        public i_nop(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
        }
    }

    public class i_not : INST
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

    public class i_pop : INST
    {
        public i_pop(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            state._stack.Pop();
        }
    }

    public class i_readonly : INST
    {
        public i_readonly(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_refanytype : INST
    {
        public i_refanytype(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_refanyval : INST
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

    public class i_ret : INST
    {
        public i_ret(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            // There are really two different stacks here:
            // one for the called method, and the other for the caller of the method.
            // When returning, the stack of the method is pretty much unchanged.
            // In fact the top of stack often contains the return value from the method.
            // Back in the caller, the stack is popped of all arguments to the callee.
            // And, the return value is pushed on the top of stack.
            // This is handled by the call instruction.

            if (!(this.Block.HasStructReturnValue || this.Block.HasScalarReturnValue))
            {
            }
            else if (this.Block.HasScalarReturnValue)
            {
                // See this on struct return--https://groups.google.com/forum/#!topic/llvm-dev/RSnV-Vr17nI
                // The following fails for structs, so do not do this for struct returns.
                var v = state._stack.Pop();
                state._stack.Push(v);
            }
            else if (this.Block.HasStructReturnValue)
            {
            }
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            // There are really two different stacks here:
            // one for the called method, and the other for the caller of the method.
            // When returning, the stack of the method is pretty much unchanged.
            // In fact the top of stack often contains the return value from the method.
            // Back in the caller, the stack is popped of all arguments to the callee.
            // And, the return value is pushed on the top of stack.
            // This is handled by the call instruction.

            if (!(this.Block.HasStructReturnValue || this.Block.HasScalarReturnValue))
            {
                var i = LLVM.BuildRetVoid(Builder);
            }
            else if (this.Block.HasScalarReturnValue)
            {
                // See this on struct return--https://groups.google.com/forum/#!topic/llvm-dev/RSnV-Vr17nI
                // The following fails for structs, so do not do this for struct returns.
                var v = state._stack.Pop();
                var i = LLVM.BuildRet(Builder, v.V);
                state._stack.Push(new VALUE(i));
            }
            else if (this.Block.HasStructReturnValue)
            {
                var v = state._stack.Pop();
                var p = state._struct_ret[0];
                var store = LLVM.BuildStore(Builder, v.V, p.V);
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine(new VALUE(store));
                var i = LLVM.BuildRetVoid(Builder);
            }
        }
    }

    public class i_rethrow : INST
    {
        public i_rethrow(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_shl : INST
    {
        public i_shl(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var lhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(lhs);

            var result = lhs;
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(result);

            state._stack.Push(result);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var lhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(lhs);

            var result = LLVM.BuildShl(Builder, lhs.V, rhs.V, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(result));

            state._stack.Push(new VALUE(result));
        }
    }

    public class i_shr : INST
    {
        public i_shr(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var lhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(lhs);

            var result = lhs;
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(result);

            state._stack.Push(result);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);

            var lhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(lhs);

            var result = LLVM.BuildAShr(Builder, lhs.V, rhs.V, "i" + instruction_id++);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(new VALUE(result));

            state._stack.Push(new VALUE(result));
        }
    }

    public class i_shr_un : INST
    {
        public i_shr_un(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_sizeof : INST
    {
        public i_sizeof(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var value = typeof(System.IntPtr).ToMonoTypeReference();
            object operand = this.Operand;
            System.Type t = operand.GetType();
            if (t.FullName == "Mono.Cecil.PointerType")
                state._stack.Push(value);
            else
                throw new Exception("Unimplemented sizeof");
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            object operand = this.Operand;
            System.Type t = operand.GetType();
            if (t.FullName == "Mono.Cecil.PointerType")
                state._stack.Push(new VALUE(LLVM.ConstInt(LLVM.Int32Type(), 8, false)));
            else
                throw new Exception("Unimplemented sizeof");
        }
    }

    public class i_starg : ConvertStArgInst
    {
        public i_starg(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_starg_s : ConvertStArgInst
    {
        public i_starg_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
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
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_stelem_i2 : ConvertStoreElement
    {
        public i_stelem_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_stelem_i4 : ConvertStoreElement
    {
        public i_stelem_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_stelem_i8 : ConvertStoreElement
    {
        public i_stelem_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(long));
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
            _dst = new TYPE(typeof(float));
        }
    }

    public class i_stelem_r8 : ConvertStoreElement
    {
        public i_stelem_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(double));
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
            _dst = new TYPE(typeof(sbyte));
        }
    }

    public class i_stind_i2 : ConvertStoreIndirect
    {
        public i_stind_i2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(short));
        }
    }

    public class i_stind_i4 : ConvertStoreIndirect
    {
        public i_stind_i4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(int));
        }
    }

    public class i_stind_i8 : ConvertStoreIndirect
    {
        public i_stind_i8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(long));
        }
    }

    public class i_stind_i : ConvertStoreIndirect
    {
        public i_stind_i(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
			_dst = new TYPE(typeof(int)); // native and c# int the same.
        }
    }

    public class i_stind_r4 : ConvertStoreIndirect
    {
        public i_stind_r4(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(float));
        }
    }

    public class i_stind_r8 : ConvertStoreIndirect
    {
        public i_stind_r8(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            _dst = new TYPE(typeof(double));
        }
    }

    public class i_stind_ref : ConvertStoreIndirect
    {
        public i_stind_ref(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
			_dst = new TYPE(typeof(object));
        }
    }

    public class i_stloc : ConvertStLoc
    {
        public i_stloc(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.ParameterReference pr = i.Operand as Mono.Cecil.ParameterReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stloc_0 : ConvertStLoc
    {
        public i_stloc_0(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 0;
            _arg = arg;
        }
    }

    public class i_stloc_1 : ConvertStLoc
    {
        public i_stloc_1(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 1;
            _arg = arg;
        }
    }

    public class i_stloc_2 : ConvertStLoc
    {
        public i_stloc_2(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 2;
            _arg = arg;
        }
    }

    public class i_stloc_3 : ConvertStLoc
    {
        public i_stloc_3(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            int arg = 3;
            _arg = arg;
        }
    }

    public class i_stloc_s : ConvertStLoc
    {
        public i_stloc_s(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
            Mono.Cecil.Cil.VariableReference pr = i.Operand as Mono.Cecil.Cil.VariableReference;
            int arg = pr.Index;
            _arg = arg;
        }
    }

    public class i_stobj : INST
    {
        public i_stobj(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   //  stobj – store a value at an address , page 428
            var s = state._stack.Pop();
            var d = state._stack.Pop();
            object operand = this.Operand;
            var o = operand as TypeReference;
            o = o.RewriteMonoTypeReference();
            var p = o.Deresolve(this.Block._original_method_reference.DeclaringType, null);
        }

        public override void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   //  stobj – store a value at an address, page 428
            var src = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(src);
            var dst = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(dst);
            object operand = this.Operand;
            var o = operand as TypeReference;
            o = o.RewriteMonoTypeReference();
            var p = o.Deresolve(this.Block._original_method_reference.DeclaringType, null);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(p);

            TypeRef stype = LLVM.TypeOf(src.V);
            TypeRef dtype = LLVM.TypeOf(dst.V);

            if (stype == LLVM.Int64Type()
                  && (dtype == LLVM.Int32Type() || dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
            else if (stype == LLVM.Int32Type()
                  && (dtype == LLVM.Int16Type() || dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
            else if (stype == LLVM.Int16Type()
                  && (dtype == LLVM.Int8Type() || dtype == LLVM.Int1Type()))
                src = new VALUE(LLVM.BuildTrunc(Builder, src.V, dtype, "i" + instruction_id++));
            else if (LLVM.GetTypeKind(stype) == TypeKind.PointerTypeKind)
                ;
            var zz = LLVM.BuildStore(Builder, src.V, dst.V);
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("Store = " + new VALUE(zz).ToString());
        }
    }

    public class i_stsfld : INST
    {
        public i_stsfld(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {   // stsfld (store static field), ecma 335 page 429
            state._stack.Pop();
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {   // stsfld (store static field), ecma 335 page 429
            var value = state._stack.Pop();
            var operand = this.Operand;
            var mono_field_reference = operand as FieldReference;
            if (mono_field_reference == null)
                throw new Exception("Unknown field type");
            var type = mono_field_reference.ResolveDeclaringType();
            var mono_field_type = mono_field_reference.FieldType;
            mono_field_type = mono_field_type.RewriteMonoTypeReference();
            var llvm_field_type = mono_field_type.ToTypeRef();
            // Call meta to get static field. This can be done now because
            // the address of the static field does not change.
            var bcl_type = RUNTIME.GetBclType(type);
            if (bcl_type == IntPtr.Zero) throw new Exception();
            IntPtr[] fields = null;
            IntPtr* buf;
            int len;
            RUNTIME.BclGetFields(bcl_type, &buf, &len);
            fields = new IntPtr[len];
            for (int i = 0; i < len; ++i) fields[i] = buf[i];
            var mono_fields = type.ResolveFields().ToArray();
            var find = fields.Where(f =>
            {
                var ptrName = RUNTIME.BclGetFieldName(f);
                string name = Marshal.PtrToStringAnsi(ptrName);
                return name == mono_field_reference.Name;
            });
            IntPtr first = find.FirstOrDefault();
            if (first == IntPtr.Zero) throw new Exception("Cannot find field--stsfld");
            var ptr = RUNTIME.BclGetStaticField(first);
            bool isArr = mono_field_type.IsArray;
            bool isSt = mono_field_type.IsStruct();
            bool isRef = mono_field_type.IsReferenceType();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(LLVM.PrintTypeToString(llvm_field_type));
            var address = LLVM.ConstInt(LLVM.Int64Type(), (ulong)ptr, false);
            var f1 = LLVM.BuildIntToPtr(Builder, address, llvm_field_type, "i" + instruction_id++);
            LLVM.BuildStore(Builder, value.V, f1);
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

    public class i_switch : INST
    {
        public i_switch(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_tail : INST
    {
        public i_tail(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_throw : INST
    {
        public i_throw(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }

        public override void GenerateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);
        }

        public override unsafe void Convert(STATE<VALUE, StackQueue<VALUE>> state)
        {
            var rhs = state._stack.Pop();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(rhs);
        }
    }

    public class i_unaligned : INST
    {
        public i_unaligned(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_unbox : INST
    {
        public i_unbox(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_unbox_any : INST
    {
        public i_unbox_any(Mono.Cecil.Cil.Instruction i)
            : base(i)
        {
        }
    }

    public class i_volatile : INST
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

