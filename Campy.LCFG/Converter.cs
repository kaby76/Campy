using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Campy.GraphAlgorithms;
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

        public Converter(CIL_CFG mcfg)
        {
            _mcfg = mcfg;
        }

        public void ConvertToLLVM(IEnumerable<CIL_CFG.Vertex> change_set)
        {
            // Map all basic blocks in CIL to LLVM.
            IEnumerable<CIL_CFG.Vertex> mono_bbs = change_set;
            foreach (var lv in mono_bbs)
            {
                if (lv.IsEntry)
                {
                    MethodDefinition method = lv.Method;
                    System.Reflection.MethodBase mb =
                        ReflectionCecilInterop.ConvertToSystemReflectionMethodInfo(method);
                    string mn = mb.DeclaringType.Assembly.GetName().Name;
                    ModuleRef mod = LLVM.ModuleCreateWithName(mn);
                    lv.Module = mod;
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
                        throw new Exception("Cannot handle parameter type.");
                    }
                    else
                    {
                        throw new Exception("Cannot handle parameter type.");
                    }
                    TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                    ValueRef fun = LLVM.AddFunction(mod, mb.Name, met_type);
                    BasicBlockRef entry = LLVM.AppendBasicBlock(fun, lv.Name.ToString());
                    lv.BasicBlock = entry;
                    lv.Function = fun;
                    BuilderRef builder = LLVM.CreateBuilder();
                    lv.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, entry);
                }
            }
            foreach (var mv in mono_bbs)
            {
                IEnumerable<CIL_CFG.Vertex> successors = _mcfg.SuccessorNodes(mv);
                if (!mv.IsEntry)
                {
                    var ent = mv.Entry;
                    var lvv_ent = ent;
                    var fun = lvv_ent.Function;
                    var bb = LLVM.AppendBasicBlock(fun, mv.Name.ToString());
                    mv.BasicBlock = bb;
                    mv.Function = lvv_ent.Function;
                    BuilderRef builder = LLVM.CreateBuilder();
                    mv.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, bb);
                }
            }
	        foreach (CIL_CFG.Vertex mv in mono_bbs)
	        {
		        Inst prev = null;
		        foreach (var j in mv.Instructions)
		        {
		            j.Block = mv;
			        if (prev != null) prev.Next = j;
			        prev = j;
		        }
	        }

            var entries = _mcfg.VertexNodes.Where(node => node.IsEntry).ToList();

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

            foreach (CIL_CFG.Vertex node in _mcfg.VertexNodes)
            {
                if (node.IsEntry) continue;
                CIL_CFG.Vertex e = node.Entry;
                node.HasReturnValue = e.HasReturnValue;
                node.NumberOfArguments = e.NumberOfArguments;
                node.NumberOfLocals = e.NumberOfLocals;
            }

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

            List<CIL_CFG.Vertex> change_set_minus_unreachable = new List<CIL_CFG.Vertex>(mono_bbs);
            foreach (CIL_CFG.Vertex v in unreachable)
            {
                if (change_set_minus_unreachable.Contains(v))
                {
                    change_set_minus_unreachable.Remove(v);
                }
            }

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
                    foreach (int ob in dfs)
                    {
                        CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                        var llvm_node = node;
                        visited.Add(node);
                        if (!(work.Contains(node)))
                        {
                            continue;
                        }
                        work.Remove(node);

                        // Use predecessor information to get initial stack size.
                        if (node.IsEntry)
                        {
                            CIL_CFG.Vertex llvm_nodex = node;
                            llvm_nodex.StackLevelIn = node.NumberOfLocals + node.NumberOfArguments;
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
                                var llvm_pred = pred;
                                if (llvm_pred.StackLevelOut == null)
                                {
                                    continue;
                                }
                                // Warn if predecessor does not concur with another predecessor.
                                CIL_CFG.Vertex llvm_nodex = node;
                                llvm_nodex.StackLevelIn = llvm_pred.StackLevelOut;
                                in_level = (int) llvm_nodex.StackLevelIn;
                            }
                            // Warn if no predecessors have been visited.
                            if (in_level == -1)
                            {
                                continue;
                            }
                        }
                        CIL_CFG.Vertex llvm_nodez = node;
                        int level_after = (int) llvm_nodez.StackLevelIn;
                        int level_pre = level_after;
                        foreach (var i in llvm_nodez.Instructions)
                        {
                            level_pre = level_after;
                            i.ComputeStackLevel(ref level_after);
                            //System.Console.WriteLine("after inst " + i);
                            //System.Console.WriteLine("level = " + level_after);
                            Debug.Assert(level_after >= node.NumberOfLocals + node.NumberOfArguments);
                        }
                        llvm_nodez.StackLevelOut = level_after;
                        // Verify return node that it makes sense.
                        if (node.IsReturn && !unreachable.Contains(node))
                        {
                            if (llvm_nodez.StackLevelOut ==
                                node.NumberOfArguments +
                                node.NumberOfLocals +
                                (node.HasReturnValue ? 1 : 0))
                                ;
                            else
                            {
                                throw new Exception("Failed stack level out check");
                            }
                        }
                        foreach (CIL_CFG.Vertex succ in node._Graph.SuccessorNodes(node))
                        {
                            // If it's an interprocedural edge, nothing to pass on.
                            if (succ.Method != node.Method)
                                continue;
                            // If it's recursive, nothing more to do.
                            if (succ.IsEntry)
                                continue;
                            // If it's a return, nothing more to do also.
                            if (node.Instructions.Last() as i_ret != null)
                                continue;
                            // Nothing to update if no change.
                            CIL_CFG.Vertex llvm_succ = node;
                            if (llvm_succ.StackLevelIn > level_after)
                            {
                                continue;
                            }
                            else if (llvm_succ.StackLevelIn == level_after)
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

            {
                // Get a list of nodes to compile.
                List<CIL_CFG.Vertex> work = new List<CIL_CFG.Vertex>(change_set_minus_unreachable);

                // Get a list of the name of nodes to compile.
                IEnumerable<int> work_names = work.Select(v => v.Name);

                // Get a Tarjan DFS/SCC order of the nodes. Reverse it because we want to
                // proceed from entry basic block.
                var ordered_list = new Tarjan<int>(_mcfg).GetEnumerable().Reverse();

                // Eliminate all node names not in the work list.
                var order = ordered_list.Where(v => work_names.Contains(v)).ToList();

                // Set up the initial states associated with each node, that is, state into and state out of.
                foreach (int ob in order)
                {
                    CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CIL_CFG.Vertex llvm_node = node;
                    llvm_node.StateIn = new State(node.Method, llvm_node.NumberOfArguments, llvm_node.NumberOfLocals,
                        (int) llvm_node.StackLevelIn);
                    llvm_node.StateOut = new State(node.Method, llvm_node.NumberOfArguments, llvm_node.NumberOfLocals,
                        (int) llvm_node.StackLevelOut);
                }

                Dictionary<int, bool> visited = new Dictionary<int, bool>();

                // Emit LLVM IR code, based on state and per-instruction simulation on that state.
                foreach (int ob in order)
                {
                    CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CIL_CFG.Vertex llvm_node = node;

                    var state_in = new State(visited, llvm_node);
                    llvm_node.StateIn = state_in;
                    llvm_node.StateOut = new State(state_in);

                    node.OutputEntireNode();
                    state_in.Dump();

                    Inst last_inst = null;
                    for (Inst i = llvm_node.Instructions.First(); i != null;)
                    {
                        System.Console.WriteLine(i);
                        last_inst = i;
                        i = i.Convert(llvm_node.StateOut);
                        llvm_node.StateOut.Dump();

                    }
                    if (last_inst != null && last_inst.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Next)
                    {
                        // Need to insert instruction to branch to fall through.
                        GraphLinkedList<int, CIL_CFG.Vertex, CIL_CFG.Edge>.Edge edge = llvm_node._Successors[0];
                        int succ = edge.To;
                        var s = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(succ)];
                        var br = LLVM.BuildBr(llvm_node.Builder, s.BasicBlock);
                    }
                    visited[ob] = true;
                }

                // Finally, update phi functions with "incoming" information from predecessors.
                foreach (int ob in order)
                {
                    CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CIL_CFG.Vertex llvm_node = node;
                    int size = llvm_node.StateIn._stack.Count;
                    for (int i = 0; i < size; ++i)
                    {
                        var count = llvm_node._Predecessors.Count;
                        if (count < 2) continue;
                        ValueRef res;
                        res = llvm_node.StateIn._stack[i].V;
                        if (!llvm_node.StateIn._phi.Contains(res)) continue;
                        ValueRef[] phi_vals = new ValueRef[count];
                        for (int c = 0; c < count; ++c)
                        {
                            var p = llvm_node._Predecessors[c].From;
                            var plm = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(p)];
                            var vr = plm.StateOut._stack[i];
                            phi_vals[c] = vr.V;
                        }
                        BasicBlockRef[] phi_blocks = new BasicBlockRef[count];
                        for (int c = 0; c < count; ++c)
                        {
                            var p = llvm_node._Predecessors[c].From;
                            var plm = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(p)];
                            phi_blocks[c] = plm.BasicBlock;
                        }
                        System.Console.WriteLine();
                        System.Console.WriteLine("Node " + llvm_node.Name + " stack slot " + i + " types:");
                        for (int c = 0; c < count; ++c)
                        {
                            var vr = phi_vals[c];
                            System.Console.WriteLine(GetStringTypeOf(vr));
                        }

                        LLVM.AddIncoming(res, phi_vals, phi_blocks);
                    }
                }
                System.Console.WriteLine("===========");
                foreach (int ob in order)
                {
                    CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CIL_CFG.Vertex llvm_node = node;

                    node.OutputEntireNode();
                    llvm_node.StateIn.Dump();
                    llvm_node.StateOut.Dump();
                }
            }
        }

        public string GetStringTypeOf(ValueRef v)
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

        public IntPtr GetPtr(int block_number)
        {
            CIL_CFG.Vertex here = null;
            foreach (var xxx in _mcfg.VertexNodes)
            {
                if (xxx.IsEntry && xxx.Name == block_number)
                {
                    here = xxx;
                    break;
                }
            }
            CIL_CFG.Vertex lvv = here;
            var mod = lvv.Module;
            MyString error = new MyString();
            LLVM.VerifyModule(mod, VerifierFailureAction.ReturnStatusAction, error);
            System.Console.WriteLine(error.ToString());
            ExecutionEngineRef engine;
            LLVM.DumpModule(mod);
            LLVM.LinkInMCJIT();
            LLVM.InitializeNativeTarget();
            LLVM.InitializeNativeAsmPrinter();
            MCJITCompilerOptions options = new MCJITCompilerOptions();
            var optionsSize = (4 * sizeof(int)) + IntPtr.Size; // LLVMMCJITCompilerOptions has 4 ints and a pointer
            LLVM.InitializeMCJITCompilerOptions(options, (uint) optionsSize);
            LLVM.CreateMCJITCompilerForModule(out engine, mod, options, (uint) optionsSize, error);
            var ptr = LLVM.GetPointerToGlobal(engine, lvv.Function);
            IntPtr p = (IntPtr) ptr;

            return p;
        }

        public static TypeRef MapMonoTypeToLLVMType(TypeDefinition td)
        {
            System.Type sys_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToBasicSystemReflectionType(td);
            TypeRef type;
            if (sys_type == typeof(System.Int16))
            {
                type = LLVM.Int16Type();
            }
            else if (sys_type == typeof(System.Int32))
            {
                type = LLVM.Int32Type();
            }
            else if (sys_type == typeof(System.Int64))
            {
                type = LLVM.Int64Type();
            }
            else if (sys_type == typeof(System.Boolean))
            {
                type = LLVM.Int32Type();
            }
            else throw new Exception("Cannot handle type.");
            return type;
        }
    }
}
