using Campy.CIL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Campy.Graphs;
using Campy.Types.Utils;
using Campy.Utils;
using Mono.Cecil;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public class Converter
    {
        private CIL_CFG _mcfg;
        private LLVMCFG _lcfg;

        private Dictionary<CIL_CFG.Vertex, LLVMCFG.Vertex> _cil_to_llvm_node_map =
            new Dictionary<CIL_CFG.Vertex, LLVMCFG.Vertex>();

        public Converter(CIL_CFG mcfg, LLVMCFG lcfg)
        {
            _mcfg = mcfg;
            _lcfg = lcfg;
        }

        public void ConvertToLLVM(IEnumerable<int> change_set)
        {
            // Map all basic blocks in CIL to LLVM.
            IEnumerable<CIL_CFG.Vertex> mono_bbs =
                change_set.Select(i =>
                {
                    CIL_CFG.Vertex r = _mcfg.NameToVertex(i);
                    return r;
                });

            foreach (CIL_CFG.Vertex mv in mono_bbs)
            {
                GraphLinkedList<int, LLVMCFG.Vertex, LLVMCFG.Edge>.Vertex lv = _lcfg.AddVertex(mv.Name);
                LLVMCFG.Vertex lvv = lv as LLVMCFG.Vertex;
                _cil_to_llvm_node_map[mv] = (LLVMCFG.Vertex)lv;

                if (mv.IsEntry)
                {
                    MethodDefinition method = mv.Method;
                    System.Reflection.MethodBase mb =
                        ReflectionCecilInterop.ConvertToSystemReflectionMethodInfo(method);
                    string mn = mb.DeclaringType.Assembly.GetName().Name;
                    ModuleRef mod = LLVM.ModuleCreateWithName(mn);
                    uint count = (uint) mb.GetParameters().Count();
                    TypeRef[] param_types = new TypeRef[count];
                    int current = 0;
                    if (count > 0)
                        foreach (ParameterInfo p in mb.GetParameters())
                        {
                            if (p.ParameterType.IsValueType && p.ParameterType == typeof(Int16))
                            {
                                param_types[current++] = LLVM.Int16Type();
                            }
                            else if (p.ParameterType.IsValueType && p.ParameterType == typeof(Int32))
                            {
                                param_types[current++] = LLVM.Int32Type();
                            }
                            else if (p.ParameterType.IsValueType && p.ParameterType == typeof(Int64))
                            {
                                param_types[current++] = LLVM.Int64Type();
                            }
                            else if (p.ParameterType.IsValueType && p.ParameterType == typeof(Boolean))
                            {
                                param_types[current++] = LLVM.Int1Type();
                            }
                            else if (p.ParameterType.IsArray)
                            {

                            }
                        }
                    TypeRef ret_type = default(TypeRef);
                    if (mb is System.Reflection.MethodInfo mi2 && mi2.ReturnType == typeof(int))
                    {
                        ret_type = LLVM.Int32Type();
                    }
                    else if (mb is System.Reflection.MethodInfo mi3 && mi3.ReturnType == typeof(Int16))
                    {
                        ret_type = LLVM.Int16Type();
                    }
                    else if (mb is System.Reflection.MethodInfo mi4 && mi4.ReturnType == typeof(Int32))
                    {
                        ret_type = LLVM.Int32Type();
                    }
                    else if (mb is System.Reflection.MethodInfo mi5 && mi5.ReturnType == typeof(Int64))
                    {
                        ret_type = LLVM.Int64Type();
                    }
                    else if (mb is System.Reflection.MethodInfo mi6 && mi6.ReturnType == typeof(Boolean))
                    {
                        ret_type = LLVM.Int1Type();
                    }
                    else if (mb is System.Reflection.MethodInfo mi7 && mi7.ReturnType.IsArray)
                    {
                    }
                    TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                    ValueRef fun = LLVM.AddFunction(mod, mb.Name, met_type);
                    BasicBlockRef entry = LLVM.AppendBasicBlock(fun, mv.Name.ToString());
                    lvv.BasicBlock = entry;
                    lvv.Function = fun;
                    BuilderRef builder = LLVM.CreateBuilder();
                    lvv.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, entry);
                }
            }
            foreach (CIL_CFG.Vertex mv in mono_bbs)
            {
                LLVMCFG.Vertex fv = _cil_to_llvm_node_map[mv];
                IEnumerable<CIL_CFG.Vertex> successors = _mcfg.SuccessorNodes(mv);
                foreach (var s in successors)
                {
                    var tv = _cil_to_llvm_node_map[s];
                    _lcfg.AddEdge(fv, tv);
                }
                if (!mv.IsEntry)
                {
                    var ent = mv.Entry;
                    var lvv_ent = _cil_to_llvm_node_map[ent];
                    var fun = lvv_ent.Function;
                    var bb = LLVM.AppendBasicBlock(fun, mv.Name.ToString());
                    fv.BasicBlock = bb;
                    fv.Function = lvv_ent.Function;
                    fv.Builder = lvv_ent.Builder;
                }
            }
            foreach (CIL_CFG.Vertex mv in mono_bbs)
            {
                LLVMCFG.Vertex fv = _cil_to_llvm_node_map[mv];
                foreach (var j in mv.Instructions)
                {
                    var i = Inst.Wrap(j, fv);
                    fv.Instructions.Add(i);
                }
            }

            //******************************************************
            //
            // STEP 1.
            //
            // Create a list of entries.
            // 
            //      Go through all vertices, determine if it's a
            //      an entry node (i.e., the beginning of the method),
            //      and add it to a list.
            //
            //******************************************************
            var entries = _mcfg.VertexNodes.Where(node => node.IsEntry).ToList();

            //******************************************************
            //
            // STEP 3.
            //
            // Set the number of arguments, locals, and the return
            // value count for each of the nodes in the changed set.
            //
            //      Examine the node method (Mono) and using properties
            //      from Mono for the method, compute the attributes
            //      for the nodes.
            //
            //******************************************************
            foreach (CIL_CFG.Vertex node in mono_bbs)
            {
                int args = 0;
                Mono.Cecil.MethodDefinition md = node.Method;
                Mono.Cecil.MethodReference mr = node.Method;
                if (mr.HasThis) args++;
                args += mr.Parameters.Count;
                node.NumberOfArguments = args;
                int locals = md.Body.Variables.Count;
                node.NumberOfLocals = locals;
                int ret = 0;
                if (mr.MethodReturnType != null)
                {
                    Mono.Cecil.MethodReturnType rt = mr.MethodReturnType;
                    Mono.Cecil.TypeReference tr = rt.ReturnType;
                    // Get type, may contain modifiers.
                    // Note, the return type must be examined in order
                    // to really determine if it returns a value--"void"
                    // means that it doesn't return a value.
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
                node.HasReturnValue = ret > 0;
            }

            //******************************************************
            //
            // STEP 4.
            //
            // Compute list of unreachable nodes. These will be removed
            // from further consideration.
            //
            //******************************************************
            List<CIL_CFG.Vertex> unreachable = new List<CIL_CFG.Vertex>();
            {
                // Create DFT order of all nodes.
                IEnumerable<int> objs = entries.Select(x => x.Name);
                GraphAlgorithms.DepthFirstPreorderTraversal<int>
                    dfs = new GraphAlgorithms.DepthFirstPreorderTraversal<int>(
                        _mcfg,
                        objs
                        );
                List<CIL_CFG.Vertex> visited = new List<CIL_CFG.Vertex>();
                foreach (int ob in dfs)
                {
                    CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    visited.Add(node);
                }
                foreach (CIL_CFG.Vertex v in mono_bbs)
                {
                    if (!visited.Contains(v))
                        unreachable.Add(v);
                }
            }

            //******************************************************
            //
            // STEP 5.
            //
            // Compute list of change set minus unreachable nodes.
            // Most of these nodes are "catch" or "finally" blocks,
            // which we aren't supporting for this CFA.
            //
            //******************************************************
            List<CIL_CFG.Vertex> change_set_minus_unreachable = new List<CIL_CFG.Vertex>(mono_bbs);
            foreach (CIL_CFG.Vertex v in unreachable)
            {
                if (change_set_minus_unreachable.Contains(v))
                {
                    change_set_minus_unreachable.Remove(v);
                }
            }

            //******************************************************
            //
            // STEP 6.
            //
            // Compute stack sizes for change set.
            //
            //******************************************************
            {
                List<CIL_CFG.Vertex> work = new List<CIL_CFG.Vertex>(change_set_minus_unreachable);
                while (work.Count != 0)
                {
                    // Create DFT order of all nodes.
                    IEnumerable<int> objs = entries.Select(x => x.Name);
                    GraphAlgorithms.DepthFirstPreorderTraversal<int>
                        dfs = new GraphAlgorithms.DepthFirstPreorderTraversal<int>(
                            _mcfg,
                            objs
                            );

                    List<CIL_CFG.Vertex> visited = new List<CIL_CFG.Vertex>();
                    // Compute stack size for each basic block, processing nodes on work list
                    // in DFT order.
                    foreach (var ob in dfs)
                    {
                        CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                        var llvm_node = _cil_to_llvm_node_map[node];
                        visited.Add(node);
                        if (!(work.Contains(node)))
                        {
                            continue;
                        }
                        work.Remove(node);

                        // Use predecessor information to get initial stack size.
                        if (node.IsEntry)
                        {
                            node.StackLevelIn = node.NumberOfLocals + node.NumberOfArguments;
                        }
                        else
                        {
                            int in_level = -1;
                            foreach (CIL_CFG.Vertex pred in _mcfg.PredecessorNodes(node))
                            {
                                // Do not consider interprocedural edges when computing stack size.
                                if (pred.Method != node.Method)
                                    continue;
                                // If predecessor has not been visited, warn and do not consider.
                                var llvm_pred = _cil_to_llvm_node_map[pred];
                                if (pred.StackLevelOut == null)
                                {
                                    continue;
                                }
                                // Warn if predecessor does not concur with another predecessor.
                                node.StackLevelIn = pred.StackLevelOut;
                                in_level = (int)node.StackLevelIn;
                            }
                            // Warn if no predecessors have been visited.
                            if (in_level == -1)
                            {
                                continue;
                            }
                        }

                        int level_after = (int)node.StackLevelIn;
                        int level_pre = level_after;
                        foreach (CIL_Inst i in node.Instructions)
                        {
                            level_pre = level_after;
                            i.ComputeStackLevel(ref level_after);
                            //System.Console.WriteLine("after inst " + i);
                            //System.Console.WriteLine("level = " + level_after);
                            Debug.Assert(level_after >= node.NumberOfLocals + node.NumberOfArguments);
                        }
                        node.StackLevelOut = level_after;
                        // Verify return node that it makes sense.
                        if (node.IsReturn && !unreachable.Contains(node))
                        {
                            if (node.StackLevelOut ==
                                node.NumberOfArguments +
                                node.NumberOfLocals +
                                (node.HasReturnValue ? 1 : 0))
                                ;
                            else
                            {
                                throw new Exception("Failed stack level out check");
                            }
                        }
                        node.StackLevelPreLastInstruction = level_pre;
                        foreach (CIL_CFG.Vertex succ in node._Graph.SuccessorNodes(node))
                        {
                            // If it's an interprocedural edge, nothing to pass on.
                            if (succ.Method != node.Method)
                                continue;
                            // If it's recursive, nothing more to do.
                            if (succ.IsEntry)
                                continue;
                            // If it's a return, nothing more to do also.
                            if (node.Instructions.Last() as CIL.i_ret != null)
                                continue;
                            // Nothing to update if no change.
                            if (succ.StackLevelIn > level_after)
                            {
                                continue;
                            }
                            else if (succ.StackLevelIn == level_after)
                            {
                                continue;
                            }
                            if (!work.Contains(succ))
                            {
                                work.Add(succ);
                            }
                        }
                    }
                }
            }

            //******************************************************
            //
            // STEP 7.
            //
            // Convert change set to basic SSA representation.
            // Each node is converted by itself, without any predecessor
            // information. To do that, each stack must have unique variables.
            //
            //******************************************************
            {
                List<CIL_CFG.Vertex> work = new List<CIL_CFG.Vertex>(change_set_minus_unreachable);
                StackQueue<CIL_CFG.Vertex> worklist = new StackQueue<CIL_CFG.Vertex>();
                while (work.Count != 0)
                {
                    // Create DFT order of all nodes.
                    IEnumerable<int> objs = entries.Select(x => x.Name);
                    GraphAlgorithms.DepthFirstPreorderTraversal<int>
                        dfs = new GraphAlgorithms.DepthFirstPreorderTraversal<int>(
                            _mcfg,
                            objs
                            );

                    List<CIL_CFG.Vertex> visited = new List<CIL_CFG.Vertex>();
                    foreach (int ob in dfs)
                    {
                        CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                        visited.Add(node);
                        if (!(work.Contains(node)))
                        {
                            continue;
                        }
                        work.Remove(node);

                        // Check if stack levels computed.
                        if (node.StackLevelIn == null)
                        {
                            continue;
                        }

                        int level_in = (int)node.StackLevelIn;
                        LLVMCFG.Vertex llvm_node = _cil_to_llvm_node_map[node];
                        llvm_node.StateIn = new State(node, llvm_node, level_in);
                        State state_after = new State(llvm_node.StateIn);
                        foreach (var i in llvm_node.Instructions)
                        {
                            i.StateIn = new State(state_after);

                            i.Convert(ref state_after);

                            i.StateOut = new State(state_after);
                        }
                        llvm_node.StateOut = state_after;
                    }
                }
            }

            //******************************************************
            //
            // STEP 8.
            //
            // Set up phi functions for parallel For method calls.
            // This is where the "magic" comes from!
            // Every For-function call looks something like the following:
            //  IL_0056: ldloc.3 [v100:= v93] // "e"
            //  IL_0057: ldloc.0 [v101:= v81] // delegate class
            //  IL_0058: ldftn System.Void Test.Program /<> c__DisplayClass0_0::< Main > b__0(Campy.Types.Index) [v102:= node 12]
            //  IL_005e: newobj System.Void Campy.Parallel / _Kernel_type::.ctor(System.Object, System.IntPtr) [v103:= s4]
            //  IL_0063: call System.Void Campy.Parallel::For(Campy.Types.Extent, Campy.Parallel / _Kernel_type)
            //
            // Note the few previous instructions prior to the function call to For.
            // The first instruction to note pushes the Extent variable for the For method.
            // The next three create a delegate with the given class and method. The result
            // is the creation of a delegate with target for the For method.
            // We now transfer control to the For method, but essentially we can consider
            // this a transfer of control to the delegate itself!
            // So, in the entry for the the delegate, set up the stack states to contain phi functions
            // with the basic information containing the target of the delegate, and the index.
            //
            // In this SSA, there will be only one instance of the target type. 
            //
            //******************************************************
            //{
            //    foreach (Inst inst in Inst.CallInstructions)
            //    {
            //        object operand = inst.Operand;
            //        Mono.Cecil.MethodReference call_to = operand as Mono.Cecil.MethodReference;
            //        Mono.Cecil.MethodDefinition call_to_def = call_to != null ? call_to.Resolve() : null;

            //        if (call_to_def == null)
            //            continue;

            //        // The following is big time special case for Campy calls,
            //        // where the function is called indirectly. If it's the parallel.for
            //        // call, find out what we're calling.
            //        if (!(call_to_def != null && call_to_def.Name.Equals("For")
            //              && call_to_def.DeclaringType != null &&
            //              call_to_def.DeclaringType.FullName.Equals("Campy.Parallel")))
            //            continue;

            //        System.Console.WriteLine("Campy.Parallel::For caller/callee matching.");
            //        System.Console.WriteLine("Caller: " + inst);
            //        var node = inst.Block;
            //        int index_of_call = node._instructions
            //            .Select((f, i) => new { Field = f, Index = i })
            //            .Where(x => x.Field == inst)
            //            .Select(x => x.Index)
            //            .DefaultIfEmpty(-1)
            //            .FirstOrDefault();

            //        if (index_of_call < 3)
            //            continue;

            //        // Back up one instruction, grab delegate method.
            //        int index_of_ldftn = index_of_call - 2;
            //        var ldftn_instruction = node._instructions[index_of_ldftn];

            //        OpCode op = ldftn_instruction.Instruction.OpCode;
            //        if (op.Code != Mono.Cecil.Cil.Code.Ldftn)
            //            continue;

            //        object operand_ldftn = ldftn_instruction.Operand;
            //        Mono.Cecil.MethodReference call_to_ldftn = operand_ldftn as Mono.Cecil.MethodReference;
            //        Mono.Cecil.MethodDefinition call_to_def_ldftn = call_to_ldftn?.Resolve();

            //        if (call_to_def_ldftn == null)
            //            continue;

            //        // Find a straight-forward match of the call to an entry,
            //        // using the method definition. This works much of the time,
            //        // but not always.
            //        CIL_CFG.CFGVertex callee = entries.Where(v => v.Method == call_to_def_ldftn)
            //            .FirstOrDefault();

            //        if (callee == null)
            //            continue;

            //        System.Console.WriteLine("callee " + callee);

            //        // Set up ssa to note target, which is essentially the
            //        // only thing that matters.
            //        Mono.Cecil.TypeDefinition td = call_to_def_ldftn.DeclaringType;
            //        SSA.Obj target = new SSA.Obj(td.FullName);
            //        SSA.Value vx = callee.StateIn._stack[0];
            //        SSA.Phi phi = null;
            //        if (!ssa.phi_functions.TryGetValue(vx, out phi))
            //        {
            //            phi = new SSA.Phi();
            //            ssa.phi_functions.Add(vx, phi);
            //            List<SSA.Value> list = new List<SSA.Value>();
            //            phi._merge = list;
            //            phi._v = vx;
            //            phi._block = node;
            //        }
            //        phi._merge.Add(target);
            //    }
            //}
            //******************************************************
            //
            // STEP 9.
            //
            // Set up phi functions for change set.
            //
            //******************************************************
            //{
            //    List<CIL_CFG.CFGVertex> work = new List<CIL_CFG.CFGVertex>(change_set_minus_unreachable);
            //    while (work.Count != 0)
            //    {
            //        CIL_CFG.CFGVertex node = work.First();
            //        work.Remove(node);
            //        if (Options.Singleton.Get(Options.OptionType.DisplaySSAComputation))
            //            System.Console.WriteLine("Compute phi-function for node " + node);
            //        if (Options.Singleton.Get(Options.OptionType.DisplaySSAComputation))
            //            System.Console.WriteLine("predecessors " +
            //                _cfg.PredecessorNodes(node).Aggregate(
            //                    new StringBuilder(),
            //                    (sb, v) =>
            //                        sb.Append(v).Append(", "),
            //                    sb =>
            //                    {
            //                        if (0 < sb.Length)
            //                            sb.Length -= 2;
            //                        return sb.ToString();
            //                    }));

            //        // Verify all predecessors have identical stack sizes.
            //        IEnumerable<int?> levels = _cfg.PredecessorNodes(node).Select(
            //            (v) =>
            //            {
            //                return v.StackLevelOut;
            //            }
            //        );
            //        int? previous = null;
            //        bool first = true;
            //        bool cannot_check = false;
            //        foreach (int? l in levels)
            //        {
            //            if (first)
            //            {
            //                first = false;
            //                previous = l;
            //            }
            //            else
            //            {
            //                if (l != previous)
            //                {
            //                    if (Options.Singleton.Get(Options.OptionType.DisplaySSAComputation))
            //                        System.Console.WriteLine(
            //                            "Predecessor stack sizes do not agree.");
            //                    cannot_check = true;
            //                    break;
            //                }
            //            }
            //        }
            //        if (cannot_check)
            //            continue;
            //        // Every block has a stack that contains variables
            //        // defined as phi function with all predecessors.
            //        //if (!node.IsEntry)
            //        {
            //            for (int i = 0; i < node.StackLevelIn; ++i)
            //            {
            //                SSA.Value vx = node.StateIn._stack[i];
            //                SSA.Phi phi;
            //                if (!ssa.phi_functions.TryGetValue(vx, out phi))
            //                {
            //                    phi = new SSA.Phi();
            //                    ssa.phi_functions.Add(vx, phi);
            //                }
            //                List<SSA.Value> list = _cfg.PredecessorNodes(node).Select(
            //                    (v) =>
            //                    {
            //                        return v.StateOut._stack[i];
            //                    }
            //                ).ToList();
            //                phi._merge = list;
            //                phi._v = vx;
            //                phi._block = node;
            //            }
            //        }
            //    }
            //}

            //            //System.Console.WriteLine("Final graph:");
            //            //_cfg.Dump();

            //            // Dump SSA phi functions.
            //            //System.Console.WriteLine("Phi functions");
            //            //foreach (KeyValuePair<SSA.Value, SSA.Phi> p in ssa.phi_functions)
            //            //{
            //            //    System.Console.WriteLine(p.Key + " "
            //            //        + p.Value._merge.Aggregate(
            //            //                new StringBuilder(),
            //            //                (sb, x) =>
            //            //                   sb.Append(x).Append(", "),
            //            //               sb =>
            //            //                {
            //            //                    if (0 < sb.Length)
            //            //                        sb.Length -= 2;
            //            //                    return sb.ToString();
            //            //                }));
            //            //}

        }
    }
}
