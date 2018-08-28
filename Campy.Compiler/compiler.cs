using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using Campy.Graphs;
using Campy.Utils;
using Campy.Meta;
using Mono.Cecil;
using Swigged.LLVM;
using System.Runtime.InteropServices;
using Mono.Cecil.Cil;
using Swigged.Cuda;
using Mono.Cecil.Rocks;
using Mono.Collections.Generic;
using FieldAttributes = Mono.Cecil.FieldAttributes;

namespace Campy.Compiler
{

    static class PHASES
    {
        public static List<CFG.Vertex> RemoveBasicBlocksAlreadyCompiled(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            List<CFG.Vertex> weeded = new List<CFG.Vertex>();

            // Remove any blocks already compiled.
            foreach (var bb in basic_blocks_to_compile)
            {
                if (!bb.AlreadyCompiled)
                {
                    weeded.Add(bb);
                    bb.AlreadyCompiled = true;
                }
                else
                {

                }
            }

            return weeded;
        }

        private static void InitStateGenerics(STATE<TypeReference, SafeStackQueue<TypeReference>> state,
            Dictionary<CFG.Vertex, STATE<TypeReference, SafeStackQueue<TypeReference>>> states_in,
            Dictionary<CFG.Vertex, STATE<TypeReference, SafeStackQueue<TypeReference>>> states_out,
            CFG.Vertex bb)
        {
            int in_level = -1;
            int args = bb.StackNumberOfArguments;
            bool scalar_ret = bb.HasScalarReturnValue;
            bool struct_ret = bb.HasStructReturnValue;
            bool has_this = bb.HasThis;
            bool is_catch = bb.IsCatch;
            int locals = bb.StackNumberOfLocals;
            // Use predecessor information to get initial stack size.
            if (bb.IsEntry)
            {
                in_level = bb.StackNumberOfLocals + bb.StackNumberOfArguments;
            }
            else
            {
                foreach (CFG.Vertex pred in bb._graph.PredecessorNodes(bb))
                {
                    // Do not consider interprocedural edges when computing stack size.
                    if (pred._method_reference != bb._method_reference)
                        throw new Exception("Interprocedural edge should not exist.");
                    // If predecessor has not been visited, warn and do not consider.
                    // Warn if predecessor does not concur with another predecessor.
                    if (in_level != -1 && states_out.ContainsKey(pred) && states_out[pred]._stack.Count != in_level)
                        throw new Exception("Miscalculation in stack size "
                                            + "for basic block " + bb
                                            + " or predecessor " + pred);
                    if (states_out.ContainsKey(pred))
                        in_level = states_out[pred]._stack.Count;
                }
            }

            if (in_level == -1)
            {
                throw new Exception("Predecessor edge computation screwed up.");
            }

            int level = in_level;
            // State depends on predecessors. Unlike in compilation, we are
            // only propagating type information. So, chose the first predecessor.
            if (bb._graph.PredecessorNodes(bb).Count() == 0)
            {
                if (!bb.IsEntry) throw new Exception("Cannot handle dead code blocks.");
                if (has_this)
                    state._stack.Push(bb._method_reference.DeclaringType);

                for (int i = 0; i < bb._method_definition.Parameters.Count; ++i)
                {
                    var par = bb._method_reference.Parameters[i];
                    var type = par.ParameterType;
                    if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                        System.Console.WriteLine(par);
                    state._stack.Push(type);
                }

                int offset = 0;
                state._struct_ret = state._stack.Section(struct_ret ? offset++ : offset, struct_ret ? 1 : 0);
                state._this = state._stack.Section(has_this ? offset++ : offset, has_this ? 1 : 0);
                // Set up args. NB: arg 0 is "this" pointer according to spec!!!!!
                state._arguments = state._stack.Section(
                    has_this ? offset - 1 : offset,
                    args - (struct_ret ? 1 : 0));

                // Set up locals. I'm making an assumption that there is a 
                // one to one and in order mapping of the locals with that
                // defined for the method body by Mono.
                Collection<VariableDefinition> vars = bb._method_reference.Resolve().Body.Variables;
                var variables = vars.ToArray();
                state._locals = state._stack.Section((int) state._stack.Count, locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i].VariableType.InstantiateGeneric(bb._method_reference);
                    state._stack.Push(tr);
                }

                // Set up any thing else.
                for (int i = state._stack.Size(); i < level; ++i)
                {
                    var value = typeof(System.Int32).ToMonoTypeReference();
                    state._stack.Push(value);
                }
            }
            else if (bb._graph.Predecessors(bb).Count() == 1)
            {
                // We don't need phi functions--and can't with LLVM--
                // if there is only one predecessor. If it hasn't been
                // converted before this node, just create basic state.

                var pred = bb._graph.PredecessorEdges(bb).ToList()[0].From;
                var other = states_out[pred];
                var size = other._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    var vx = other._stack[i];
                    state._stack.Push(vx);
                }
                if (is_catch)
                {
                    state._stack.Push(bb.CatchType);
                }
                state._struct_ret = state._stack.Section(other._struct_ret.Base, other._struct_ret.Len);
                state._this = state._stack.Section(other._this.Base, other._this.Len);
                state._arguments = state._stack.Section(other._arguments.Base, other._arguments.Len);
                state._locals = state._stack.Section(other._locals.Base, other._locals.Len);
            }
            else // node._Predecessors.Count > 0
            {
                var pred = bb._graph.PredecessorEdges(bb).ToList()[0].From;
                for (int pred_ind = 0; pred_ind < bb._graph.Predecessors(bb).ToList().Count; ++pred_ind)
                {
                    var to_check = bb._graph.PredecessorEdges(bb).ToList()[pred_ind].From;
                    CFG.Vertex check_llvm_node = to_check;
                    if (!states_out.ContainsKey(check_llvm_node))
                        continue;
                    if (states_out[check_llvm_node] == null)
                        continue;
                    if (states_out[check_llvm_node]._stack == null)
                        continue;
                    pred = to_check;
                    break;
                }

                CFG.Vertex p_llvm_node = pred;
                int size = states_out[pred]._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    {
                        var f = typeof(System.Int32).ToMonoTypeReference();
                        state._stack.Push(f);
                    }
                    var count = bb._graph.Predecessors(bb).Count();
                    var value = states_out[pred]._stack[i];
                    state._stack[i] = value;
                }
                if (is_catch)
                {
                    state._stack.Push(bb.CatchType);
                }
                var other = states_out[p_llvm_node];
                state._struct_ret = state._stack.Section(other._struct_ret.Base, other._struct_ret.Len);
                state._this = state._stack.Section(other._this.Base, other._this.Len);
                state._arguments = state._stack.Section(other._arguments.Base, other._arguments.Len);
                state._locals = state._stack.Section(other._locals.Base, other._locals.Len);
            }
        }

        public static void PropagateTypesAndPerformCallClosure(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            if (basic_blocks_to_compile.Count == 0)
                return;

            var _mcfg = basic_blocks_to_compile.First()._graph;

            // Get a list of nodes to compile.
            List<CFG.Vertex> work = new List<CFG.Vertex>(basic_blocks_to_compile);

            // Get a list of the name of nodes to compile.
            var work_names = work.Select(v => v.Name);

            // Get a Tarjan DFS/SCC order of the nodes. Reverse it because we want to
            // proceed from entry basic block.
            //var ordered_list = new TarjanNoBackEdges<int>(_mcfg).GetEnumerable().Reverse();
            var ordered_list = new TarjanNoBackEdges<CFG.Vertex, CFG.Edge>(_mcfg, work).ToList();
            ordered_list.Reverse();

            // Eliminate all node names not in the work list.
            //var order = ordered_list.Where(v => work_names.Contains(v.Name)).ToList();
            var order = ordered_list;

            Dictionary<CFG.Vertex, bool> visited = new Dictionary<CFG.Vertex, bool>();
            Dictionary<CFG.Vertex, STATE<TypeReference, SafeStackQueue<TypeReference>>> states_in = new Dictionary<CFG.Vertex, STATE<TypeReference, SafeStackQueue<TypeReference>>>();
            Dictionary<CFG.Vertex, STATE<TypeReference, SafeStackQueue<TypeReference>>>
                states_out = new Dictionary<CFG.Vertex, STATE<TypeReference, SafeStackQueue<TypeReference>>>();

            try
            {

                // propagate type information and create new basic blocks for nodes that have
                // specific generic type information.
                foreach (var bb in order)
                {
                    if (Campy.Utils.Options.IsOn("overview_import_computation_trace"))
                        System.Console.Write(bb.Name + " ");

                    // Create new stack state with predecessor information, basic block/function
                    // information.
                    var state_in = new STATE<TypeReference, SafeStackQueue<TypeReference>>(visited, states_in, states_out, bb, InitStateGenerics);

                    if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                    {
                        System.Console.WriteLine("state in");
                        state_in.OutputTrace(new String(' ', 4));
                    }

                    var state_out = new STATE<TypeReference, SafeStackQueue<TypeReference>>(state_in);
                    states_in[bb] = state_in;
                    states_out[bb] = state_out;

                    if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                    {
                        bb.OutputEntireNode();
                        state_in.OutputTrace(new String(' ', 4));
                    }

                    INST last_inst = null;
                    for (int i = 0; i < bb.Instructions.Count; ++i)
                    {
                        var inst = bb.Instructions[i];
                        if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                            System.Console.WriteLine(inst);
                        inst.CallClosure(state_out);
                        if (Campy.Utils.Options.IsOn("detailed_import_computation_trace"))
                            state_out.OutputTrace(new String(' ', 4));
                    }

                    visited[bb] = true;
                }
            }
            catch (Exception e)
            {
                _mcfg.OutputEntireGraph();
                throw e;
            }
            if (Campy.Utils.Options.IsOn("overview_import_computation_trace"))
                System.Console.WriteLine();
        }

        public static void AddCctors(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            if (basic_blocks_to_compile.Count == 0)
                return;

            var _mcfg = basic_blocks_to_compile.First()._graph;

            // It is also important to go through all data types and gather
            // cctor constructors for static fields. These have to be executed before
            // execution of the main kernel program.
            Stack<TypeReference> stack = new Stack<TypeReference>();
            foreach (var bb in basic_blocks_to_compile)
            {
                var dt = bb._method_reference.DeclaringType;
                stack.Push(dt);
            }
            do
            {
                var dt = stack.Pop();
                var parent = dt.DeclaringType;
                if (parent != null) stack.Push(parent);
                var dtr = dt.Resolve();
                var methods = dtr.Methods;
                foreach (var method in methods)
                {
                    if (method.Name == ".cctor")
                    {
                        var method_fixed = method.Deresolve(dt, null);
                        IMPORTER.Singleton().Add(method_fixed);
                    }
                }
            } while (stack.Any());
        }

        // Method to denormalize information about the method this block is associated with,
        // placing that information into the block for easier access.
        public static List<CFG.Vertex> ComputeBasicMethodProperties(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            foreach (CFG.Vertex bb in basic_blocks_to_compile)
            {
                Mono.Cecil.MethodReturnType rt = bb._method_definition.MethodReturnType;
                Mono.Cecil.TypeReference tr = rt.ReturnType;
                var ret = tr.FullName != "System.Void";
                bb.HasScalarReturnValue = ret && !tr.IsStruct();
                bb.HasStructReturnValue = ret && tr.IsStruct();
                bb.HasThis = bb._method_definition.HasThis;
                bb.StackNumberOfArguments = bb._method_definition.Parameters.Count
                                            + (bb.HasThis ? 1 : 0)
                                            + (bb.HasStructReturnValue ? 1 : 0);
                Mono.Cecil.MethodReference mr = bb._method_reference;
                int locals = mr.Resolve().Body.Variables.Count;
                bb.StackNumberOfLocals = locals;
            }

            return basic_blocks_to_compile;
        }

        public static List<CFG.Vertex> SetUpLLVMEntries(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            foreach (CFG.Vertex bb in basic_blocks_to_compile)
            {
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("Compile part 1, node " + bb);

                // Skip all but entry blocks for now.
                if (!bb.IsEntry)
                {
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine("skipping -- not an entry.");
                    continue;
                }

                MethodReference method = bb._method_reference;
                List<ParameterDefinition> parameters = method.Parameters.ToList();
                List<ParameterReference> instantiated_parameters = new List<ParameterReference>();

                ModuleRef mod = RUNTIME.global_llvm_module; // LLVM.ModuleCreateWithName(mn);
                bb.LlvmInfo.Module = mod;

                uint count = (uint) bb.StackNumberOfArguments;
                TypeRef[] param_types = new TypeRef[count];
                int current = 0;
                if (count > 0)
                {
                    if (bb.HasStructReturnValue)
                    {
                        TYPE t = new TYPE(method.ReturnType);
                        param_types[current++] = LLVM.PointerType(t.StorageTypeLLVM, 0);
                    }

                    if (bb.HasThis)
                    {
                        TYPE t = new TYPE(method.DeclaringType);
                        if (method.DeclaringType.IsValueType)
                        {
                            // Parameter "this" is a struct, but code in body of method assumes
                            // a pointer is passed. Make the parameter a pointer. For example,
                            // Int32.ToString().
                            param_types[current++] = LLVM.PointerType(t.StorageTypeLLVM, 0);
                        }
                        else
                        {
                            param_types[current++] = t.IntermediateTypeLLVM;
                        }
                    }

                    foreach (var p in parameters)
                    {
                        TypeReference type_reference_of_parameter = p.ParameterType;
                        if (method.DeclaringType.IsGenericInstance && method.ContainsGenericParameter)
                        {
                            var git = method.DeclaringType as GenericInstanceType;
                            type_reference_of_parameter = METAHELPER.FromGenericParameterToTypeReference(
                                type_reference_of_parameter, git);
                        }
                        TYPE t = new TYPE(type_reference_of_parameter);
                        param_types[current] = t.StorageTypeLLVM;
                        if (type_reference_of_parameter.IsValueType
                            && type_reference_of_parameter.Resolve().IsStruct()
                            && !type_reference_of_parameter.Resolve().IsEnum)
                            param_types[current] = LLVM.PointerType(param_types[current], 0);
                        current++;
                    }

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                    {
                        System.Console.WriteLine("Params for block " + bb.Name + " " +
                                                 bb._method_reference.FullName);
                        System.Console.WriteLine("(" + bb._method_definition.FullName + ")");
                        foreach (var pp in param_types)
                        {
                            string a = LLVM.PrintTypeToString(pp);
                            System.Console.WriteLine(" " + a);
                        }
                    }
                }

                //mi2 = FromGenericParameterToTypeReference(typeof(void).ToMonoTypeReference(), null);
                TYPE t_ret = new TYPE(METAHELPER.FromGenericParameterToTypeReference(method.ReturnType,
                    method.DeclaringType as GenericInstanceType));
                if (bb.HasStructReturnValue)
                {
                    t_ret = new TYPE(typeof(void).ToMonoTypeReference());
                }

                TypeRef ret_type = t_ret.StorageTypeLLVM;
                TypeRef method_type = LLVM.FunctionType(ret_type, param_types, false);
                string method_name = METAHELPER.RenameToLegalLLVMName(COMPILER.MethodName(method));
                ValueRef fun = LLVM.AddFunction(mod, method_name, method_type);

                var glob = LLVM.AddGlobal(mod, LLVM.PointerType(method_type, 0), "p_" + method_name);
                LLVM.SetGlobalConstant(glob, true);
                LLVM.SetInitializer(glob, fun);

                BasicBlockRef entry = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                bb.LlvmInfo.BasicBlock = entry;
                bb.LlvmInfo.MethodValueRef = fun;
                var t_fun = LLVM.TypeOf(fun);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");
                //////////LLVM.VerifyFunction(fun, VerifierFailureAction.PrintMessageAction);
                BuilderRef builder = LLVM.CreateBuilder();
                bb.LlvmInfo.Builder = builder;
                LLVM.PositionBuilderAtEnd(builder, entry);
            }

            return basic_blocks_to_compile;
        }

        public static List<CFG.Vertex> SetupLLVMNonEntries(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            if (!basic_blocks_to_compile.Any()) return basic_blocks_to_compile;
            var _mcfg = basic_blocks_to_compile.First()._graph;

            foreach (var bb in basic_blocks_to_compile)
            {
                IEnumerable<CFG.Vertex> successors = _mcfg.SuccessorNodes(bb);
                if (!bb.IsEntry)
                {
                    var ent = bb.Entry;
                    var lvv_ent = ent;
                    var fun = lvv_ent.LlvmInfo.MethodValueRef;
                    var t_fun = LLVM.TypeOf(fun);
                    var t_fun_con = LLVM.GetTypeContext(t_fun);
                    var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                    if (t_fun_con != context) throw new Exception("not equal");
                    //LLVM.VerifyFunction(fun, VerifierFailureAction.PrintMessageAction);
                    var llvm_bb = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                    bb.LlvmInfo.BasicBlock = llvm_bb;
                    bb.LlvmInfo.MethodValueRef = lvv_ent.LlvmInfo.MethodValueRef;
                    BuilderRef builder = LLVM.CreateBuilder();
                    bb.LlvmInfo.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, llvm_bb);
                }
            }

            return basic_blocks_to_compile;
        }

        private static void InitState(STATE<VALUE, StackQueue<VALUE>> state,
            Dictionary<CFG.Vertex, STATE<VALUE, StackQueue<VALUE>>> states_in,
            Dictionary<CFG.Vertex, STATE<VALUE, StackQueue<VALUE>>> states_out,
            CFG.Vertex bb)
        {
            int in_level = -1;
            int args = bb.StackNumberOfArguments;
            bool scalar_ret = bb.HasScalarReturnValue;
            bool struct_ret = bb.HasStructReturnValue;
            bool has_this = bb.HasThis;
            bool is_catch = bb.IsCatch;
            int locals = bb.StackNumberOfLocals;
            // Use predecessor information to get initial stack size.
            if (bb.IsEntry)
            {
                in_level = bb.StackNumberOfLocals + bb.StackNumberOfArguments;
            }
            else
            {
                foreach (CFG.Vertex pred in bb._graph.PredecessorNodes(bb))
                {
                    // Do not consider interprocedural edges when computing stack size.
                    if (pred._method_reference != bb._method_reference)
                        throw new Exception("Interprocedural edge should not exist.");
                    // If predecessor has not been visited, warn and do not consider.
                    // Warn if predecessor does not concur with another predecessor.
                    if (in_level != -1 && states_out.ContainsKey(pred) && states_out[pred]._stack.Count != in_level)
                        throw new Exception("Miscalculation in stack size "
                                            + "for basic block " + bb
                                            + " or predecessor " + pred);
                    if (states_out.ContainsKey(pred))
                        in_level = states_out[pred]._stack.Count;
                }
            }

            if (in_level == -1)
            {
                throw new Exception("Predecessor edge computation screwed up.");
            }

            int level = in_level;
            // State depends on predecessors. To handle this without updating state
            // until a fix point is found while converting to LLVM IR, we introduce
            // SSA phi functions.
            if (bb._graph.PredecessorNodes(bb).Count() == 0)
            {
                if (!bb.IsEntry) throw new Exception("Cannot handle dead code blocks.");
                var fun = bb.LlvmInfo.MethodValueRef;
                var t_fun = LLVM.TypeOf(fun);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(RUNTIME.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");

                // Args are store exactly as the type defined, _cil_type in LLVM.
                // When place on the "value stack" via CIL ldarg/starg, it changes to IntermediateType.
                for (uint i = 0; i < args; ++i)
                {
                    var v = LLVM.GetParam(fun, i);
                    VALUE value = new VALUE(v);
                    TYPE type = value.T;
                    TypeRef tr = LLVM.TypeOf(v);
                    if (bb.CheckArgsAlloc((int)i))
                    {
                        // Parameters are call by value. To handle ldarga, create a stack
                        // temporary, and set it with value.
                        var new_obj = LLVM.BuildAlloca(bb.LlvmInfo.Builder,
                            tr,
                            "i" + INST.instruction_id++);
                        LLVM.SetAlignment(new_obj, 8);
                        LLVM.BuildStore(bb.LlvmInfo.Builder, v, new_obj);
                        value = new VALUE(new_obj);
                    }
                    else
                    {
                        if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.PointerTypeKind)
                            value = new VALUE(LLVM.ConstPointerNull(type.CilTypeLLVM));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.DoubleTypeKind)
                            value = new VALUE(LLVM.ConstReal(LLVM.DoubleType(), 0));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.FloatTypeKind)
                            value = new VALUE(LLVM.ConstReal(LLVM.FloatType(), 0));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.IntegerTypeKind)
                            value = new VALUE(LLVM.ConstInt(type.CilTypeLLVM, (ulong)0, true));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.StructTypeKind)
                        {
                            var entry = bb.Entry.LlvmInfo.BasicBlock;
                            //var beginning = LLVM.GetFirstInstruction(entry);
                            //LLVM.PositionBuilderBefore(basic_block.Builder, beginning);
                            var new_obj = LLVM.BuildAlloca(bb.LlvmInfo.Builder, type.CilTypeLLVM,
                                "i" + INST.instruction_id++); // Allocates struct on stack, but returns a pointer to struct.
                                                              //LLVM.PositionBuilderAtEnd(bb.LlvmInfo.Builder, bb.BasicBlock);
                            value = new VALUE(new_obj);
                        }
                        else
                            throw new Exception("Unhandled type");
                    }
                    // Note, functions defined in SetUpLLVMEntries(), jiter.cs.
                    // Look for LLVM.FunctionType(
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(value);
                    state._stack.Push(value);
                }

                int offset = 0;
                state._struct_ret = state._stack.Section(struct_ret ? offset++ : offset, struct_ret ? 1 : 0);
                state._this = state._stack.Section(has_this ? offset++ : offset, has_this ? 1 : 0);
                // Set up args. NB: arg 0 is "this" pointer according to spec!!!!!
                state._arguments = state._stack.Section(
                    has_this ? offset - 1 : offset,
                    args - (struct_ret ? 1 : 0));

                // Set up locals. I'm making an assumption that the locals here
                // correspond exactly with those reported by Mono.
                Collection<VariableDefinition> vars = bb._method_reference.Resolve().Body.Variables;
                var variables = vars.Select(vd => vd.VariableType).ToArray();
                // Convert any generic parameters to generic instance reference.
                for (int i = 0; i < variables.Length; ++i)
                {
                    variables[i] = variables[i].InstantiateGeneric(bb._method_reference);
                    if (variables[i].ContainsGenericParameter)
                        throw new Exception("Uninstantiated generic parameter.");
                }
                bb.Entry._locals = variables;
                state._locals = state._stack.Section((int) state._stack.Count, locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i];
                    TYPE type = new TYPE(tr);
                    VALUE value;
                    bool use_alloca = bb.CheckLocalsAlloc(i);
                    if (use_alloca)
                    {
                        var new_obj = LLVM.BuildAlloca(bb.LlvmInfo.Builder,
                            type.CilTypeLLVM,
                            "i" + INST.instruction_id++);
                        LLVM.SetAlignment(new_obj, 64);
                        value = new VALUE(new_obj);
                    }
                    else
                    {
                        if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.PointerTypeKind)
                            value = new VALUE(LLVM.ConstPointerNull(type.CilTypeLLVM));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.DoubleTypeKind)
                            value = new VALUE(LLVM.ConstReal(LLVM.DoubleType(), 0));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.FloatTypeKind)
                            value = new VALUE(LLVM.ConstReal(LLVM.FloatType(), 0));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.IntegerTypeKind)
                            value = new VALUE(LLVM.ConstInt(type.CilTypeLLVM, (ulong)0, true));
                        else if (LLVM.GetTypeKind(type.CilTypeLLVM) == TypeKind.StructTypeKind)
                        {
                            var entry = bb.Entry.LlvmInfo.BasicBlock;
                            //var beginning = LLVM.GetFirstInstruction(entry);
                            //LLVM.PositionBuilderBefore(basic_block.Builder, beginning);
                            var new_obj = LLVM.BuildAlloca(bb.LlvmInfo.Builder, type.CilTypeLLVM,
                                "i" + INST.instruction_id++); // Allocates struct on stack, but returns a pointer to struct.
                                                              //LLVM.PositionBuilderAtEnd(bb.LlvmInfo.Builder, bb.BasicBlock);
                            value = new VALUE(new_obj);
                        }
                        else
                            throw new Exception("Unhandled type");
                    }
                    state._stack.Push(value);
                }

                // Set up any thing else.
                for (int i = state._stack.Size(); i < level; ++i)
                {
                    VALUE value = new VALUE(LLVM.ConstInt(LLVM.Int32Type(), (ulong) 0, true));
                    state._stack.Push(value);
                }

                if (is_catch)
                {
                    state._stack.Push(new VALUE(LLVM.ConstPointerNull(bb.CatchType.ToTypeRef())));
                }

            }
            else if (bb._graph.Predecessors(bb).Count() == 1)
            {
                // We don't need phi functions--and can't with LLVM--
                // if there is only one predecessor. If it hasn't been
                // converted before this node, just create basic state.

                var pred = bb._graph.PredecessorEdges(bb).ToList()[0].From;
                var p_llvm_node = pred;
                var other = states_out[p_llvm_node];
                var size = other._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    var vx = other._stack[i];
                    state._stack.Push(vx);
                }
                if (is_catch)
                {
                    state._stack.Push(new VALUE(LLVM.ConstPointerNull(bb.CatchType.ToTypeRef())));
                }

                state._struct_ret = state._stack.Section(other._struct_ret.Base, other._struct_ret.Len);
                state._this = state._stack.Section(other._this.Base, other._this.Len);
                state._arguments = state._stack.Section(other._arguments.Base, other._arguments.Len);
                state._locals = state._stack.Section(other._locals.Base, other._locals.Len);
            }
            else // node._Predecessors.Count > 0
            {
                // As we cannot guarantee whether all predecessors are fulfilled,
                // make up something so we don't have problems.
                // Now, for every arg, local, stack, set up for merge.
                // Find a predecessor that has some definition.
                var pred = bb._graph.PredecessorEdges(bb).ToList()[0].From;
                for (int pred_ind = 0; pred_ind < bb._graph.Predecessors(bb).ToList().Count; ++pred_ind)
                {
                    var to_check = bb._graph.PredecessorEdges(bb).ToList()[pred_ind].From;
                    //         if (!visited.ContainsKey(to_check)) continue;
                    CFG.Vertex check_llvm_node = to_check;
                    if (!states_out.ContainsKey(check_llvm_node))
                        continue;
                    if (states_out[check_llvm_node] == null)
                        continue;
                    if (states_out[check_llvm_node]._stack == null)
                        continue;
                    pred = to_check;
                    break;
                }

                CFG.Vertex p_llvm_node = pred;
                int size = states_out[p_llvm_node]._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    {
                        VALUE f = new VALUE(LLVM.ConstInt(LLVM.Int32Type(), (ulong) 0, true));
                        state._stack.Push(f);
                    }
                    var count = bb._graph.Predecessors(bb).Count();
                    var value = states_out[p_llvm_node]._stack[i];
                    var v = value.V;
                    TypeRef tr = LLVM.TypeOf(v);
                    ValueRef res = LLVM.BuildPhi(bb.LlvmInfo.Builder, tr, "i" + INST.instruction_id++);
                    bb.LlvmInfo.Phi.Add(res);
                    state._stack[i] = new VALUE(res);
                }
                if (is_catch)
                {
                    state._stack.Push(new VALUE(LLVM.ConstPointerNull(bb.CatchType.ToTypeRef())));
                }

                var other = states_out[p_llvm_node];
                state._struct_ret = state._stack.Section(other._struct_ret.Base, other._struct_ret.Len);
                state._this = state._stack.Section(other._this.Base, other._this.Len);
                state._arguments = state._stack.Section(other._arguments.Base, other._arguments.Len);
                state._locals = state._stack.Section(other._locals.Base, other._locals.Len);
            }
        }

        public static void TranslateToLLVMInstructions(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            if (!basic_blocks_to_compile.Any())
                return;

            var _mcfg = basic_blocks_to_compile.First()._graph;

            // Get a list of nodes to compile.
            List<CFG.Vertex> work = new List<CFG.Vertex>(basic_blocks_to_compile);

            // Get a list of the name of nodes to compile.
            var work_names = work.Select(v => v.Name);

            // Get a Tarjan DFS/SCC order of the nodes. Reverse it because we want to
            // proceed from entry basic block.
            //var ordered_list = new TarjanNoBackEdges<int>(_mcfg).GetEnumerable().Reverse();
            var ordered_list = new TarjanNoBackEdges<CFG.Vertex, CFG.Edge>(_mcfg).ToList();
            ordered_list.Reverse();

            // Eliminate all node names not in the work list.
            var order = ordered_list.Where(v => work_names.Contains(v.Name)).ToList();

            Dictionary<CFG.Vertex, bool> visited = new Dictionary<CFG.Vertex, bool>();
            Dictionary<CFG.Vertex, STATE<VALUE, StackQueue<VALUE>>> states_in = new Dictionary<CFG.Vertex, STATE<VALUE, StackQueue<VALUE>>>();
            Dictionary<CFG.Vertex, STATE<VALUE, StackQueue<VALUE>>> states_out = new Dictionary<CFG.Vertex, STATE<VALUE, StackQueue<VALUE>>>();

            // Emit LLVM IR code, based on state and per-instruction simulation on that state.
            foreach (var ob in order)
            {
                CFG.Vertex bb = ob;

                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    System.Console.WriteLine("State computations for node " + bb.Name);

                // Create new stack state with predecessor information, basic block/function
                // information.
                var state_in = new STATE<VALUE, StackQueue<VALUE>>(visited, states_in, states_out, bb, InitState);

                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                {
                    System.Console.WriteLine("state in");
                    state_in.OutputTrace(new String(' ', 4));
                }

                var state_out = new STATE<VALUE, StackQueue<VALUE>>(state_in);
                states_in[bb] = state_in;
                states_out[bb] = state_out;

                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                {
                    bb.OutputEntireNode();
                    state_in.OutputTrace(new String(' ', 4));
                }

                INST last_inst = null;
                for (int i = 0; i < bb.Instructions.Count; ++i)
                {
                    var inst = bb.Instructions[i];
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(inst);
                    inst.DebuggerInfo();
                    inst.Convert(state_out);
                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                        state_out.OutputTrace(new String(' ', 4));
                    last_inst = inst;
                }

                if (last_inst != null
                    && (
                        last_inst.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Next
                        || last_inst.OpCode.FlowControl == FlowControl.Call
                        || last_inst.OpCode.FlowControl == FlowControl.Throw))
                {
                    // Need to insert instruction to branch to fall through.
                    var edge = bb._graph.SuccessorEdges(bb).FirstOrDefault();
                    var s = edge.To;
                    var br = LLVM.BuildBr(bb.LlvmInfo.Builder, s.LlvmInfo.BasicBlock);
                }

                visited[ob] = true;
            }

            // Finally, update phi functions with "incoming" information from predecessors.
            foreach (var ob in order)
            {
                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    System.Console.WriteLine("Working on phis for node " + ob.Name);
                CFG.Vertex node = ob;
                CFG.Vertex llvm_node = node;
                int size = states_in[llvm_node]._stack.Count;
                for (int i = 0; i < size; ++i)
                {
                    var count = llvm_node._graph.Predecessors(llvm_node).Count();
                    if (count < 2) continue;
                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                        System.Console.WriteLine("phi nodes need for "
                                                 + ob.Name + " for stack depth " + i);
                    ValueRef res;
                    res = states_in[llvm_node]._stack[i].V;
                    if (!llvm_node.LlvmInfo.Phi.Contains(res))
                        continue;
                    ValueRef[] phi_vals = new ValueRef[count];
                    for (int c = 0; c < count; ++c)
                    {
                        var p = llvm_node._graph.PredecessorEdges(llvm_node).ToList()[c].From;
                        if (Campy.Utils.Options.IsOn("state_computation_trace"))
                            System.Console.WriteLine("Adding in phi for pred state "
                                                     + p.Name);
                        var plm = p;
                        var vr = states_out[plm]._stack[i];
                        phi_vals[c] = vr.V;
                    }

                    BasicBlockRef[] phi_blocks = new BasicBlockRef[count];
                    for (int c = 0; c < count; ++c)
                    {
                        var p = llvm_node._graph.PredecessorEdges(llvm_node).ToList()[c].From;
                        var plm = p;
                        phi_blocks[c] = plm.LlvmInfo.BasicBlock;
                    }

                    //System.Console.WriteLine();
                    //System.Console.WriteLine("Node " + llvm_node.Name + " stack slot " + i + " types:");
                    for (int c = 0; c < count; ++c)
                    {
                        var vr = phi_vals[c];
                        //System.Console.WriteLine(GetStringTypeOf(vr));
                    }

                    LLVM.AddIncoming(res, phi_vals, phi_blocks);
                }
            }

            if (Campy.Utils.Options.IsOn("state_computation_trace"))
            {
                foreach (var ob in order)
                {
                    CFG.Vertex node = ob;
                    CFG.Vertex llvm_node = node;

                    node.OutputEntireNode();
                    states_in[llvm_node].OutputTrace(new String(' ', 4));
                    states_out[llvm_node].OutputTrace(new String(' ', 4));
                }
            }
        }

        public static string TranslateToPTX(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            var module = RUNTIME.global_llvm_module;
            // Get basic block of entry.
            if (!basic_blocks_to_compile.Any())
                return "";

            // The first block on the list is assumed to be the entry for the kernel.
            // Probably should pass that into this method. Oh well.
            CFG.Vertex basic_block = basic_blocks_to_compile.First();

            if (Campy.Utils.Options.IsOn("llvm-output"))
                LLVM.DumpModule(module);

            if (!Campy.Utils.Options.IsOn("debug_info_off"))
                LLVM.DIBuilderFinalize(INST.dib);

            MyString error = new MyString();
            LLVM.VerifyModule(module, VerifierFailureAction.PrintMessageAction, error);
            if (error.ToString() != "")
            {
                System.Console.WriteLine(error);
                throw new Exception("Error in JIT compilation.");
            }

            string triple = "nvptx64-nvidia-cuda";
            var b = LLVM.GetTargetFromTriple(triple, out TargetRef t2, error);
            if (error.ToString() != "")
            {
                System.Console.WriteLine(error);
                throw new Exception("Error in JIT compilation.");
            }

            TargetMachineRef tmr = LLVM.CreateTargetMachine(t2, triple, "", "", CodeGenOptLevel.CodeGenLevelDefault,
                RelocMode.RelocDefault, CodeModel.CodeModelKernel);
//ContextRef context_ref = LLVM.ContextCreate();
            ContextRef context_ref = LLVM.GetModuleContext(RUNTIME.global_llvm_module);

            // Add kernel to "global" space so it can be called by CUDA Driver API.
            ValueRef kernelMd = LLVM.MDNodeInContext(
                context_ref, new ValueRef[3]
                {
                    basic_block.LlvmInfo.MethodValueRef,
                    LLVM.MDStringInContext(context_ref, "kernel", 6),
                    LLVM.ConstInt(LLVM.Int32TypeInContext(context_ref), 1, false)
                });
            LLVM.AddNamedMetadataOperand(module, "nvvm.annotations", kernelMd);

            // In addition, go through all cctors here and make sure to tag them
            // as well. Otherwise, they're device only. If you try to call,
            // you get a "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" error. Not very informative!
            foreach (var cctor in COMPILER.Singleton.AllCctorBasicBlocks())
            {
                ValueRef mark = LLVM.MDNodeInContext(
                    context_ref, new ValueRef[3]
                    {
                        cctor.LlvmInfo.MethodValueRef,
                        LLVM.MDStringInContext(context_ref, "kernel", 6),
                        LLVM.ConstInt(LLVM.Int32TypeInContext(context_ref), 1, false)
                    });
                LLVM.AddNamedMetadataOperand(module, "nvvm.annotations", mark);
            }

            try
            {
            LLVM.TargetMachineEmitToMemoryBuffer(tmr, module, Swigged.LLVM.CodeGenFileType.AssemblyFile,
                error, out MemoryBufferRef buffer);
            string ptx = null;
            try
            {
                ptx = LLVM.GetBufferStart(buffer);
                uint length = LLVM.GetBufferSize(buffer);

// Modify the version number of the ISA PTX source generated to be the
// most up to date.
                ptx = ptx.Replace(".version 3.2", ".version 6.0");

                // Make sure the target machine is set to sm_30 because we assume that as
                // a minimum, and it's compatible with GPU BCL runtime. Besides, older versions
                // are deprecated.
                // sm_35 needed for declaring pointer to function, e.g.,
                // .visible .global .align 8 .u64 p_nn_3 = nn_3;
                ptx = ptx.Replace(".target sm_20", ".target sm_35");

                // Make sure to fix the stupid end-of-line delimiters to be for Windows.
                ptx = ptx.Replace("\n", "\r\n");

                //ptx = ptx + System_String_get_Chars;

                if (Campy.Utils.Options.IsOn("ptx_trace"))
                    System.Console.WriteLine(ptx);
            }
            finally
            {
                LLVM.DisposeMemoryBuffer(buffer);
            }

            return ptx;
        }
        catch (Exception)
        {
            Console.WriteLine();
            throw;
        }
        }
    }

    public class COMPILER
    {
        private IMPORTER _importer;
        private CFG _mcfg;
        public int _start_index;
        private static bool init;
        public static bool using_cuda = true;
        private Dictionary<MethodDefinition, IntPtr> method_to_image;
        private bool done_major_init;
        private static COMPILER _singleton;
        private UInt64 _options;

        public static COMPILER Singleton
        {
            get
            {
                if (_singleton == null) _singleton = new COMPILER();
                return _singleton;
            }
        }

        private COMPILER()
        {
            InitCuda();
            RUNTIME.global_llvm_module = default(ModuleRef);
            RUNTIME.all_llvm_modules = new List<ModuleRef>();
            RUNTIME._bcl_runtime_csharp_internal_to_valueref = new Dictionary<string, ValueRef>();
            method_to_image = new Dictionary<MethodDefinition, IntPtr>(
                new LambdaComparer<MethodDefinition>((MethodDefinition a, MethodDefinition b) => a.FullName == b.FullName));
            done_major_init = false;

            _importer = IMPORTER.Singleton();
            _mcfg = _importer.Cfg;
            RUNTIME.global_llvm_module = CreateModule("global");
            LLVM.EnablePrettyStackTrace();
            var triple = LLVM.GetDefaultTargetTriple();
            LLVM.SetTarget(RUNTIME.global_llvm_module, triple);
            LLVM.InitializeAllTargets();
            LLVM.InitializeAllTargetMCs();
            LLVM.InitializeAllTargetInfos();
            LLVM.InitializeAllAsmPrinters();

            //basic_llvm_types_created.Add(
            //    typeof(string).ToMonoTypeReference(),
            //    LLVM.PointerType(LLVM.VoidType(), 0));



            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.tid.x",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.tid.y",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.tid.z",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.ctaid.x",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.ctaid.y",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.ctaid.z",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.ntid.x",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.ntid.y",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.ntid.z",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.nctaid.x",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.nctaid.y",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add("llvm.nvvm.read.ptx.sreg.nctaid.z",
                LLVM.AddFunction(
                    RUNTIME.global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            RUNTIME.Initialize();

            InitBCL();
        }

        public static void Options(UInt64 options)
        {
            Singleton._options = options;
        }

        private void InitCuda()
        {
            // Initialize CUDA if it hasn't been done before.
            if (init) return;
            try
            {
                Utils.CudaHelpers.CheckCudaError(Cuda.cuInit(0));
                Utils.CudaHelpers.CheckCudaError(Cuda.cuDevicePrimaryCtxReset(0));
                Utils.CudaHelpers.CheckCudaError(Cuda.cuCtxCreate_v2(out CUcontext pctx, 0, 0));
            }
            catch (Exception e)
            {
                using_cuda = false;
            }
            init = true;
        }

        private ModuleRef CreateModule(string name)
        {
            var new_module = LLVM.ModuleCreateWithName(name);
            var context = LLVM.GetModuleContext(new_module);
            RUNTIME.all_llvm_modules.Add(new_module);
            return new_module;
        }

        public static string MethodName(MethodReference mr)
        {
            return mr.FullName;
        }

        private string CilToPtx(List<CFG.Vertex> basic_blocks_to_compile)
        {
            basic_blocks_to_compile = basic_blocks_to_compile
                .RemoveBasicBlocksAlreadyCompiled()
                .ComputeBasicMethodProperties()
                .SetUpLLVMEntries()
                .SetupLLVMNonEntries()
                ;

            basic_blocks_to_compile
                .TranslateToLLVMInstructions();
            
            if (Utils.Options.IsOn("name_trace"))
                NameTableTrace();

            return basic_blocks_to_compile.TranslateToPTX();
        }

        public void Add(Type type)
        {
            _importer.Add(type);
        }

        static bool done_stack = false;

        public List<MethodReference> AllCctors()
        {
            var bb_list = _mcfg.Entries.Where(v =>
                v.IsEntry && v._method_reference.Name == ".cctor").Select(bb => bb._method_reference);
            return bb_list.ToList();
        }

        public List<CFG.Vertex> AllCctorBasicBlocks()
        {
            var bb_list = _mcfg.Entries.Where(v =>
                v.IsEntry && v._method_reference.Name == ".cctor");
            return bb_list.ToList();
        }

        public void ImportOnlyCompile(MethodReference kernel_method, object kernel_target)
        {
            List<CFG.Vertex> cs = null;
            CFG.Vertex bb;

            Campy.Utils.TimePhase.Time("discovery     ", () =>
            {
                // Parse kernel instructions to determine basic block representation of all the code to compile.
                int change_set_id = _mcfg.StartChangeSet();
                _importer.AnalyzeMethod(kernel_method);
                if (_importer.Failed)
                {
                    throw new Exception("Failure to find all methods in GPU code. Cannot continue.");
                }
                cs = _mcfg.PopChangeSet(change_set_id);
                if (!cs.Any())
                {
                    bb = _mcfg.Entries.Where(v =>
                        v.IsEntry && v._method_reference.FullName == kernel_method.FullName).FirstOrDefault();
                }
                else
                {
                    bb = cs.First();
                }
            });

            int num_instructions = 0;
            int num_blocks = cs.Count();
            int num_entries = 0;
            foreach (var b in cs)
            {
                if (b.IsEntry) ++num_entries;
                num_instructions += b.Instructions.Count();
            }
            System.Console.WriteLine("Number of blocks       " + num_blocks);
            System.Console.WriteLine("Number of methods      " + num_entries);
            System.Console.WriteLine("Number of instructions " + num_instructions);
        }

        public IntPtr Compile(MethodReference kernel_method, object kernel_target)
        {
            if (method_to_image.TryGetValue(kernel_method.Resolve(), out IntPtr value))
            {
                return value;
            }

            List<CFG.Vertex> cs = null;
            CFG.Vertex bb;

            Campy.Utils.TimePhase.Time("discovery     ", () =>
            {
                // Parse kernel instructions to determine basic block representation of all the code to compile.
                int change_set_id = _mcfg.StartChangeSet();
                _importer.AnalyzeMethod(kernel_method);
                if (_importer.Failed)
                {
                    throw new Exception("Failure to find all methods in GPU code. Cannot continue.");
                }
                cs = _mcfg.PopChangeSet(change_set_id);
                if (!cs.Any())
                {
                    bb = _mcfg.Entries.Where(v =>
                        v.IsEntry && v._method_reference.FullName == kernel_method.FullName).FirstOrDefault();
                }
                else
                {
                    bb = cs.First();
                }
            });

            int num_instructions = 0;
            int num_blocks = cs.Count();
            int num_entries = 0;
            foreach (var b in cs)
            {
                if (b.IsEntry) ++num_entries;
                num_instructions += b.Instructions.Count();
            }
            System.Console.WriteLine("Number of blocks       " + num_blocks);
            System.Console.WriteLine("Number of methods      " + num_entries);
            System.Console.WriteLine("Number of instructions " + num_instructions);

            string ptx = null;

            Campy.Utils.TimePhase.Time("compiler      ", () =>
            {
                // Compile methods.
                ptx = CilToPtx(cs);
            });

            if (ptx == "") throw new Exception(
                    "Change set for compilation empty, which means we compiled this before. But, it wasn't recorded.");

            if (Campy.Utils.Options.IsOn("ptx-output"))
                System.Console.WriteLine(ptx);

            IntPtr image = IntPtr.Zero;
            
            Campy.Utils.TimePhase.Time("linker        ", () =>
            {
                var current_directory = Directory.GetCurrentDirectory();
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("Current directory " + current_directory);

                CudaHelpers.CheckCudaError(Cuda.cuMemGetInfo_v2(out ulong free_memory, out ulong total_memory));
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("total memory " + total_memory + " free memory " + free_memory);
                if (!done_stack)
                {
                    CudaHelpers.CheckCudaError(Cuda.cuCtxGetLimit(out ulong pvalue, CUlimit.CU_LIMIT_STACK_SIZE));
                    CudaHelpers.CheckCudaError(Cuda.cuCtxSetLimit(CUlimit.CU_LIMIT_STACK_SIZE, (uint)pvalue * 4));
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine("Stack size " + pvalue);
                    done_stack = true;
                }
                // Add in all of the GPU BCL runtime required.
                uint num_ops_link = 5;
                var op_link = new CUjit_option[num_ops_link];
                ulong[] op_values_link = new ulong[num_ops_link];

                int size = 1024 * 100;
                op_link[0] = CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
                op_values_link[0] = (ulong) size;

                op_link[1] = CUjit_option.CU_JIT_INFO_LOG_BUFFER;
                byte[] info_log_buffer = new byte[size];
                var info_log_buffer_handle = GCHandle.Alloc(info_log_buffer, GCHandleType.Pinned);
                var info_log_buffer_intptr = info_log_buffer_handle.AddrOfPinnedObject();
                op_values_link[1] = (ulong) info_log_buffer_intptr;

                op_link[2] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
                op_values_link[2] = (ulong) size;

                op_link[3] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER;
                byte[] error_log_buffer = new byte[size];
                var error_log_buffer_handle = GCHandle.Alloc(error_log_buffer, GCHandleType.Pinned);
                var error_log_buffer_intptr = error_log_buffer_handle.AddrOfPinnedObject();
                op_values_link[3] = (ulong) error_log_buffer_intptr;

                op_link[4] = CUjit_option.CU_JIT_LOG_VERBOSE;
                op_values_link[4] = (ulong) 1;

                //op_link[5] = CUjit_option.CU_JIT_TARGET;
                //op_values_link[5] = (ulong)CUjit_target.CU_TARGET_COMPUTE_35;

                var op_values_link_handle = GCHandle.Alloc(op_values_link, GCHandleType.Pinned);
                var op_values_link_intptr = op_values_link_handle.AddrOfPinnedObject();
                var res = Cuda.cuLinkCreate_v2(num_ops_link, op_link, op_values_link_intptr, out CUlinkState linkState);
                Utils.CudaHelpers.CheckCudaError(res);

                IntPtr ptr = Marshal.StringToHGlobalAnsi(ptx);
                CUjit_option[] op = new CUjit_option[0];
                ulong[] op_values = new ulong[0];
                var op_values_handle = GCHandle.Alloc(op_values, GCHandleType.Pinned);
                var op_values_intptr = op_values_handle.AddrOfPinnedObject();
                res = Cuda.cuLinkAddData_v2(linkState, CUjitInputType.CU_JIT_INPUT_PTX, ptr, (uint) ptx.Length, "", 0,
                    op,
                    op_values_intptr);
                if (res != CUresult.CUDA_SUCCESS)
                {
                    string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                    System.Console.WriteLine(info);
                    string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                    System.Console.WriteLine(error);
                }

                Utils.CudaHelpers.CheckCudaError(res);

                // Go to directory for Campy.
                uint num_ops = 0;
                res = Cuda.cuLinkAddFile_v2(linkState, CUjitInputType.CU_JIT_INPUT_LIBRARY,
                    RUNTIME.FindNativeCoreLib(), num_ops, op, op_values_intptr);

                if (res != CUresult.CUDA_SUCCESS)
                {
                    string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                    System.Console.WriteLine(info);
                    string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                    System.Console.WriteLine(error);
                }

                Utils.CudaHelpers.CheckCudaError(res);

                res = Cuda.cuLinkComplete(linkState, out image, out ulong sz);
                if (res != CUresult.CUDA_SUCCESS)
                {
                    string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                    System.Console.WriteLine(info);
                    string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                    System.Console.WriteLine(error);
                }

                Utils.CudaHelpers.CheckCudaError(res);

                method_to_image[kernel_method.Resolve()] = image;
            });

            return image;
        }

        public CFG.Vertex GetBasicBlock(string block_id)
        {
            return _mcfg.Vertices.Where(i => i.IsEntry && i.Name == block_id).FirstOrDefault();
        }

        public CFG.Vertex GetBasicBlock(MethodReference kernel_method)
        {
            CFG.Vertex bb = _mcfg.Entries.Where(v =>
                v.IsEntry && v._method_reference.FullName == kernel_method.FullName).FirstOrDefault();
            return bb;
        }

        public CUmodule SetModule(MethodReference kernel_method, IntPtr image)
        {
            // Compiled previously. Look for basic block of entry.
            CFG.Vertex bb = _mcfg.Entries.Where(v =>
                v.IsEntry && v._method_reference.FullName == kernel_method.FullName).FirstOrDefault();
            string basic_block_id = bb.Name;
            CUmodule module = RUNTIME.InitializeModule(image);
            RUNTIME.RuntimeModule = module;
            SetBCLForModule(module);
            return module;
        }

        public void StoreJits(CUmodule module)
        {
            foreach (var v in _mcfg.Entries)
            {
                var normalized_method_name = METAHELPER.RenameToLegalLLVMName(COMPILER.MethodName(v._method_reference));
                var res = Cuda.cuModuleGetFunction(out CUfunction helloWorld, module, normalized_method_name);
                // Not every entry is going to be in module, so this isn't a problem if not found.
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuModuleGetGlobal_v2(out IntPtr hw, out ulong z, module, "p_" + normalized_method_name);
                var bcl_type = RUNTIME.MonoBclMap_GetBcl(v._method_reference.DeclaringType);
                RUNTIME.BclMetaDataSetMethodJit(hw,
                    bcl_type,
                    (int)v._method_reference.MetadataToken.RID | 0x06000000);
            }
            // Some of the BCL have virtual functions, particularly internal methods. Make sure these
            // are added.
            foreach (var caller in INST.CallInstructions)
            {
                MethodReference callee = caller.CallTarget();
                if (callee == null)
                    continue;
                CFG.Vertex n = caller.Block;
                var operand = caller.Operand;
                var mr = callee;
                var md = mr.Resolve();
                if (md == null) continue;
                if (!md.IsVirtual)
                    continue;
                foreach (RUNTIME.BclNativeMethod ci in RUNTIME.BclNativeMethods)
                {
                    if (ci._full_name == mr.FullName)
                    {
                        string the_name = ci._native_name;
                        var res = Cuda.cuModuleGetFunction(out CUfunction helloWorld, module, ci._native_name);
                        // Not every entry is going to be in module, so this isn't a problem if not found.
                        if (res != CUresult.CUDA_SUCCESS) continue;
                        res = Cuda.cuModuleGetGlobal_v2(out IntPtr hw, out ulong z, module, "p_" + ci._short_name);
                        var bcl_type = RUNTIME.MonoBclMap_GetBcl(mr.DeclaringType);
                        RUNTIME.BclMetaDataSetMethodJit(hw,
                            bcl_type,
                            (int)mr.MetadataToken.RID | 0x06000000);
                        break;
                    }
                }
            }
        }

        public CUfunction GetCudaFunction(MethodReference kernel_method, CUmodule module)
        {
            CFG.Vertex bb = _mcfg.Entries.Where(v =>
            v.IsEntry && v._method_reference.FullName == kernel_method.FullName).FirstOrDefault();
            var normalized_method_name = METAHELPER.RenameToLegalLLVMName(COMPILER.MethodName(bb._method_reference));
            var res = Cuda.cuModuleGetFunction(out CUfunction helloWorld, module, normalized_method_name);
            Utils.CudaHelpers.CheckCudaError(res);
            return helloWorld;
        }

        private HashSet<string> _added_module_already = new HashSet<string>();

        public void AddAssemblyToFileSystem(string full_path_assem)
        {
            if (_added_module_already.Contains(full_path_assem))
                return;
            if (Campy.Utils.Options.IsOn("runtime_trace"))
                System.Console.WriteLine("Adding to GPU file system " + full_path_assem);
            _added_module_already.Add(full_path_assem);
            // Set up corlib.dll in file system.
            string assem = Path.GetFileName(full_path_assem);
            Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
            var corlib_bytes_handle_len = stream.Length;
            var corlib_bytes = new byte[corlib_bytes_handle_len];
            stream.Read(corlib_bytes, 0, (int)corlib_bytes_handle_len);
            var corlib_bytes_handle = GCHandle.Alloc(corlib_bytes, GCHandleType.Pinned);
            var corlib_bytes_intptr = corlib_bytes_handle.AddrOfPinnedObject();
            stream.Close();
            stream.Dispose();
            var ptrx = Marshal.StringToHGlobalAnsi(assem);
            BUFFERS buffers = new BUFFERS();
            RUNTIME.BclCheckHeap();
            IntPtr pointer1 = buffers.New(assem.Length + 1);
            RUNTIME.BclCheckHeap();
            BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
            var pointer4 = buffers.New(sizeof(int));
            RUNTIME.BclAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);
        }

        public void InitBCL()
        {
            if (!done_major_init)
            {
                BUFFERS buffers = new BUFFERS();
                int the_size = 2 * 536870912;
                IntPtr b = buffers.New(the_size);
                RUNTIME.BclPtr = b;
                RUNTIME.BclPtrSize = (ulong) the_size;
                int max_threads = 16;
                RUNTIME.InitTheBcl(b, the_size, 2 * 16777216, max_threads);
                RUNTIME.BclSetOptions(_options);
                RUNTIME.BclInitFileSystem();
                // Set up corlib.dll in file system.
                string full_path_assem = RUNTIME.FindCoreLib();
                string assem = Path.GetFileName(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                var corlib_bytes_handle_len = stream.Length;
                var corlib_bytes = new byte[corlib_bytes_handle_len];
                stream.Read(corlib_bytes, 0, (int) corlib_bytes_handle_len);
                var corlib_bytes_handle = GCHandle.Alloc(corlib_bytes, GCHandleType.Pinned);
                var corlib_bytes_intptr = corlib_bytes_handle.AddrOfPinnedObject();
                stream.Close();
                stream.Dispose();
                var ptrx = Marshal.StringToHGlobalAnsi(assem);
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
                var pointer4 = buffers.New(sizeof(int));
                RUNTIME.BclAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);
                RUNTIME.BclContinueInit();
                done_major_init = true;
            }
        }

        public void SetBCLForModule(CUmodule mod)
        {
            CUresult res;

            // Instead of full initialization, just set pointer to BCL globals area.
            Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                out Campy.Utils.CudaHelpers.dim3 tile_size,
                out Campy.Utils.CudaHelpers.dim3 tiles);

            // Set up pointer to bcl.
            unsafe
            {
                // Set up parameters.
                IntPtr bcl_ptr = RUNTIME.BclPtr;
                IntPtr[] x1 = new IntPtr[] {bcl_ptr};
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                IntPtr parm1 = handle1.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] {parm1};

                CUfunction _Z15Set_BCL_GlobalsP6_BCL_t = RUNTIME._Z15Set_BCL_GlobalsP6_BCL_t(mod);
                fixed (IntPtr* kernelParams = kp)
                {
                    res = Cuda.cuLaunchKernel(
                        _Z15Set_BCL_GlobalsP6_BCL_t,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr) kernelParams,
                        (IntPtr) IntPtr.Zero
                    );
                }
                Utils.CudaHelpers.CheckCudaError(res);
                res = Cuda.cuCtxSynchronize();
                Utils.CudaHelpers.CheckCudaError(res);
            }
        }

        public static string GetStringTypeOf(ValueRef v)
        {
            TypeRef stype = LLVM.TypeOf(v);
            if (stype == LLVM.Int64Type())
                return "Int64Type";
            else if (stype == LLVM.Int32Type())
                return "Int32Type";
            else if (stype == LLVM.Int16Type())
                return "Int16Type";
            else if (stype == LLVM.Int8Type())
                return "Int8Type";
            else if (stype == LLVM.DoubleType())
                return "DoubleType";
            else if (stype == LLVM.FloatType())
                return "FloatType";
            else return "unknown";
        }


        public void NameTableTrace()
        {
            System.Console.WriteLine("Name mapping table.");
            foreach (var tuple in METAHELPER._rename_to_legal_llvm_name_cache)
            {
                System.Console.WriteLine(tuple.Key);
                System.Console.WriteLine(tuple.Value);
                System.Console.WriteLine();
            }
        }
    }
}
