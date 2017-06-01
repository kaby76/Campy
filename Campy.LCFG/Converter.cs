using Campy.CIL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

        public void ConvertToLLVM(IEnumerable<CIL_CFG.Vertex> change_set)
        {
            // Map all basic blocks in CIL to LLVM.
            IEnumerable<CIL_CFG.Vertex> mono_bbs = change_set;
            foreach (var mv in mono_bbs)
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
                    lvv.Module = mod;
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
            foreach (var mv in mono_bbs)
            {
                LLVMCFG.Vertex fv = _cil_to_llvm_node_map[mv];
                IEnumerable<CIL_CFG.Vertex> successors = _mcfg.SuccessorNodes(mv);
                if (!mv.IsEntry)
                {
                    var ent = mv.Entry;
                    var lvv_ent = _cil_to_llvm_node_map[ent];
                    var fun = lvv_ent.Function;
                    var bb = LLVM.AppendBasicBlock(fun, mv.Name.ToString());
                    fv.BasicBlock = bb;
                    fv.Function = lvv_ent.Function;
                    BuilderRef builder = LLVM.CreateBuilder();
                    fv.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, bb);
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
            }
            foreach (CIL_CFG.Vertex mv in mono_bbs)
            {
                LLVMCFG.Vertex fv = _cil_to_llvm_node_map[mv];
                Inst prev = null;
                foreach (var j in mv.Instructions)
                {
                    var i = Inst.Wrap(j);
                    i.Block = fv;
                    fv.Instructions.Add(i);
                    if (prev != null) prev.Next = i;
                    prev = i;
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
                            LLVMCFG.Vertex llvm_nodex = _cil_to_llvm_node_map[node];
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
                                var llvm_pred = _cil_to_llvm_node_map[pred];
                                if (llvm_pred.StackLevelOut == null)
                                {
                                    continue;
                                }
                                // Warn if predecessor does not concur with another predecessor.
                                LLVMCFG.Vertex llvm_nodex = _cil_to_llvm_node_map[node];
                                llvm_nodex.StackLevelIn = llvm_pred.StackLevelOut;
                                in_level = (int)llvm_nodex.StackLevelIn;
                            }
                            // Warn if no predecessors have been visited.
                            if (in_level == -1)
                            {
                                continue;
                            }
                        }
                        LLVMCFG.Vertex llvm_nodez = _cil_to_llvm_node_map[node];
                        int level_after = (int)llvm_nodez.StackLevelIn;
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
                        llvm_nodez.StackLevelPreLastInstruction = level_pre;
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
                            LLVMCFG.Vertex llvm_succ = _cil_to_llvm_node_map[node];
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
                List<CIL_CFG.Vertex> work = new List<CIL_CFG.Vertex>(change_set_minus_unreachable);
                StackQueue<CIL_CFG.Vertex> worklist = new StackQueue<CIL_CFG.Vertex>();
                while (work.Count != 0)
                {
                    // Create DFT order of all nodes.
                    IEnumerable<int> objs = entries.Select(x => x.Name);
                    GraphAlgorithms.TSort<int>
                        tSort = new GraphAlgorithms.TSort<int>(
                            _mcfg,
                            objs
                            );

                    List<CIL_CFG.Vertex> visited = new List<CIL_CFG.Vertex>();
                    foreach (int ob in tSort)
                    {
                        CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                        visited.Add(node);
                        System.Console.WriteLine(node);
                    }
                    visited = new List<CIL_CFG.Vertex>();
                    foreach (int ob in tSort)
                    {
                        CIL_CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                        visited.Add(node);
                        if (!(work.Contains(node)))
                        {
                            continue;
                        }
                        work.Remove(node);
                        LLVMCFG.Vertex llvm_nodez = _cil_to_llvm_node_map[node];

                        // Check if stack levels computed.
                        if (llvm_nodez.StackLevelIn == null)
                        {
                            continue;
                        }

                        int level_in = (int)llvm_nodez.StackLevelIn;
                        LLVMCFG.Vertex llvm_node = _cil_to_llvm_node_map[node];
                        llvm_node.StateIn = new State(node, llvm_node, level_in);
                        llvm_node.StateOut = new State(llvm_node.StateIn);
                        for (Inst i = llvm_node.Instructions.First(); i != null; )
                        {
                            i = i.Convert(llvm_node.StateOut);
                        }
                    }
                }
            }
        }

        public IntPtr GetPtr(int block_number)
        {
            KeyValuePair<CIL_CFG.Vertex, LLVMCFG.Vertex> here = default(KeyValuePair<CIL_CFG.Vertex, LLVMCFG.Vertex>);

            foreach (KeyValuePair<CIL_CFG.Vertex, LLVMCFG.Vertex> xxx in this._cil_to_llvm_node_map)
            {
                if (xxx.Key.IsEntry && xxx.Key.Name == block_number)
                {
                    here = xxx;
                    break;
                }
            }
            CIL_CFG.Vertex mv = here.Key;
            LLVMCFG.Vertex lvv = here.Value;
            var mod = lvv.Module;
            MyString error = new MyString();
            LLVM.VerifyModule(mod, VerifierFailureAction.AbortProcessAction, error);
            System.Console.WriteLine(error.ToString());
            ExecutionEngineRef engine;
            LLVM.DumpModule(mod);
            LLVM.LinkInMCJIT();
            LLVM.InitializeNativeTarget();
            LLVM.InitializeNativeAsmPrinter();
            MCJITCompilerOptions options = new MCJITCompilerOptions();
            var optionsSize = (4 * sizeof(int)) + IntPtr.Size; // LLVMMCJITCompilerOptions has 4 ints and a pointer
            LLVM.InitializeMCJITCompilerOptions(options, (uint)optionsSize);
            LLVM.CreateMCJITCompilerForModule(out engine, mod, options, (uint)optionsSize, error);
            var ptr = LLVM.GetPointerToGlobal(engine, lvv.Function);
            IntPtr p = (IntPtr)ptr;

            return p;
        }
    }
}
