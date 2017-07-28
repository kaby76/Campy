using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Campy.GraphAlgorithms;
using Campy.Graphs;
using Campy.Types.Utils;
using Campy.Utils;
using Mono.Cecil;
using Swigged.LLVM;

namespace Campy.ControlFlowGraph
{
    public class Converter
    {
        private CFG _mcfg;

        public Converter(CFG mcfg)
        {
            _mcfg = mcfg;
        }

        // Finally, we need a mapping of node to rewrites.
        Dictionary<CFG.Vertex, MultiMap<Mono.Cecil.TypeReference, System.Type>> map =
            new Dictionary<CFG.Vertex, MultiMap<TypeReference, System.Type>>();

        public class Comparer : IEqualityComparer<Tuple<CFG.Vertex, Mono.Cecil.TypeReference, System.Type>>
        {
            bool IEqualityComparer<Tuple<CFG.Vertex, TypeReference, System.Type>>.Equals(Tuple<CFG.Vertex, TypeReference, System.Type> x, Tuple<CFG.Vertex, TypeReference, System.Type> y)
            {
                // Order by vertex id, typereference string, type string.
                if (x.Item1.Name != y.Item1.Name)
                    return false;
                // Equal vertex name.
                if (x.Item2.Name != y.Item2.Name)
                    return false;
                // Equal TypeReference.
                if (x.Item3.Name != y.Item3.Name)
                    return false;

                return true;
            }

            int IEqualityComparer<Tuple<CFG.Vertex, TypeReference, System.Type>>.GetHashCode(Tuple<CFG.Vertex, TypeReference, System.Type> obj)
            {
                int result = 0;
               // result = obj.Item1.GetHashCode() + obj.Item2.GetHashCode() + obj.Item3.GetHashCode();
                return result;
            }
        }

        Dictionary<Tuple<CFG.Vertex, Mono.Cecil.TypeReference, System.Type>, CFG.Vertex> mmap
            = new Dictionary<Tuple<CFG.Vertex, TypeReference, System.Type>, CFG.Vertex>(new Comparer());


        private CFG.Vertex FindInstantiatedBasicBlock(CFG.Vertex current, Mono.Cecil.TypeReference generic_type, System.Type value)
        {
            var k = new Tuple<CFG.Vertex, TypeReference, System.Type>(current, generic_type, value);

            // Find vertex that maps from base vertex via symbol.
            if (!mmap.ContainsKey(k))
                return null;

            var v = mmap[k];
            return v;
        }

        private void EnterInstantiatedBasicBlock(CFG.Vertex current, Mono.Cecil.TypeReference generic_type, System.Type value, CFG.Vertex xx)
        {
            var k = new Tuple<CFG.Vertex, TypeReference, System.Type>(current, generic_type, value);
            mmap[k] = xx;
        }

        private CFG.Vertex Eval(CFG.Vertex current, List<Tuple<TypeReference, System.Type>> ops)
        {
            // Start at current vertex, and find transition state given ops.
            var copy = new List<Tuple<TypeReference, System.Type>>(ops);
            while (copy.Count != 0)
            {
                bool found = false;
                for (int i = 0; i < copy.Count; ++i)
                {
                    var t = copy[i];
                    var x = FindInstantiatedBasicBlock(current, t.Item1, t.Item2);
                    if (x != null)
                    {
                        current = x;
                        copy.RemoveAt(i);
                        found = true;
                        break;
                    }
                }
                if (! found) throw new Exception("Cannot transition.");
            }
            return current;
        }

        public void InstantiateGenerics(IEnumerable<CFG.Vertex> change_set, List<System.Type> list_of_data_types_used, List<Mono.Cecil.TypeDefinition> list_of_mono_data_types_used)
        {
            // Start a new change set so we can update edges and other properties for the new nodes
            // in the graph.
            int change_set_id2 = _mcfg.StartChangeSet();

            // We need to do bookkeeping of what nodes to consider.
            Stack<CFG.Vertex> instantiated_nodes = new Stack<CFG.Vertex>(change_set);

            while (instantiated_nodes.Count > 0)
            {
                CFG.Vertex lv = instantiated_nodes.Pop();

                MethodDefinition method = lv.Method;
                var declaring_type = method.DeclaringType;
                var method_has_generics = method.HasGenericParameters;
                var method_contains_generics = method.ContainsGenericParameter;
                var method_generic_instance = method.IsGenericInstance;

                System.Console.WriteLine("Considering " + lv.Name);

                // If a block associated with method contains generics,
                // we need to duplicate the node and add in type information
                // about the generic type with that is actually used.
                // So, for example, if the method contains a parameter of type
                // "T", then we add in a mapping of T to the actual data type
                // used, e.g., Integer, or what have you. When it is compiled,
                // to LLVM, the mapped data type will be used!
                if (method_contains_generics)
                {
                    // Let's first consider the parameters to the function.
                    var parameters = method.Parameters;
                    for (int k = 0; k < parameters.Count; ++k)
                    {
                        ParameterDefinition par = parameters[k];
                        var type_to_consider = par.ParameterType;
                        if (type_to_consider.ContainsGenericParameter)
                        {
                            var declaring_type_of_considered_type = type_to_consider.DeclaringType;

                            // "type_to_consider" is generic, so find matching
                            // type, make mapping, and node copy.
                            for (int i = 0; i < list_of_data_types_used.Count; ++i)
                            {
                                var data_type_used = list_of_mono_data_types_used[i];
                                var sys_data_type_used = list_of_data_types_used[i];

                                var data_type_used_has_generics = data_type_used.HasGenericParameters;
                                var data_type_used_contains_generics = data_type_used.ContainsGenericParameter;
                                var data_type_used_generic_instance = data_type_used.IsGenericInstance;

                                var sys_data_type_used_is_generic_type = sys_data_type_used.IsGenericType;
                                var sys_data_type_used_is_generic_parameter = sys_data_type_used.IsGenericParameter;
                                var sys_data_type_used_contains_generics = sys_data_type_used.ContainsGenericParameters;
                                if (sys_data_type_used_is_generic_type)
                                {
                                    var sys_data_type_used_get_generic_type_def = sys_data_type_used.GetGenericTypeDefinition();
                                }

                                if (declaring_type_of_considered_type.FullName.Equals(data_type_used.FullName))
                                {
                                    // Find generic parameter corresponding to par.ParameterType
                                    System.Type xx = null;
                                    for (int l = 0; l < sys_data_type_used.GetGenericArguments().Count(); ++l)
                                    {
                                        var pp = declaring_type.GenericParameters;
                                        var ppp = pp[l];
                                        if (ppp.Name == type_to_consider.Name)
                                            xx = sys_data_type_used.GetGenericArguments()[l];
                                    }

                                    // Match. First find node if it exists.
                                    var old_node = FindInstantiatedBasicBlock(lv, type_to_consider, xx);
                                    if (old_node != null)
                                        continue;

                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new_node_id);
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = lv.Instructions;
                                    new_cfg_node.Method = lv.Method;
                                    if (lv.OriginalVertex == null) new_cfg_node.OriginalVertex = lv;
                                    else new_cfg_node.OriginalVertex = lv.OriginalVertex;
                                    // Add in rewrites.
                                    new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    new_cfg_node.node_type_map.Add(type_to_consider, xx);
                                    EnterInstantiatedBasicBlock(lv, type_to_consider, xx, new_cfg_node);
                                    System.Console.WriteLine("Adding new node " + new_cfg_node.Name);

                                    // Push this node back on the stack.
                                    instantiated_nodes.Push(new_cfg_node);
                                }
                            }
                        }
                    }

                    // Next, consider the return value.
                    {
                        var return_type = method.ReturnType;
                        var type_to_consider = return_type;
                        if (type_to_consider.ContainsGenericParameter)
                        {
                            var declaring_type_of_considered_type = type_to_consider.DeclaringType;

                            // "type_to_consider" is generic, so find matching
                            // type, make mapping, and node copy.
                            for (int i = 0; i < list_of_data_types_used.Count; ++i)
                            {
                                var data_type_used = list_of_mono_data_types_used[i];
                                var sys_data_type_used = list_of_data_types_used[i];

                                var data_type_used_has_generics = data_type_used.HasGenericParameters;
                                var data_type_used_contains_generics = data_type_used.ContainsGenericParameter;
                                var data_type_used_generic_instance = data_type_used.IsGenericInstance;

                                var sys_data_type_used_is_generic_type = sys_data_type_used.IsGenericType;
                                var sys_data_type_used_is_generic_parameter = sys_data_type_used.IsGenericParameter;
                                var sys_data_type_used_contains_generics = sys_data_type_used.ContainsGenericParameters;
                                if (sys_data_type_used_is_generic_type)
                                {
                                    var sys_data_type_used_get_generic_type_def = sys_data_type_used.GetGenericTypeDefinition();
                                }

                                if (declaring_type_of_considered_type.FullName.Equals(data_type_used.FullName))
                                {
                                    // Find generic parameter corresponding to par.ParameterType
                                    System.Type xx = null;
                                    for (int l = 0; l < sys_data_type_used.GetGenericArguments().Count(); ++l)
                                    {
                                        var pp = declaring_type.GenericParameters;
                                        var ppp = pp[l];
                                        if (ppp.Name == type_to_consider.Name)
                                            xx = sys_data_type_used.GetGenericArguments()[l];
                                    }

                                    // Match. First find node if it exists.
                                    var old_node = FindInstantiatedBasicBlock(lv, type_to_consider, xx);
                                    if (old_node != null)
                                        continue;

                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new_node_id);
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = lv.Instructions;
                                    new_cfg_node.Method = lv.Method;
                                    if (lv.OriginalVertex == null) new_cfg_node.OriginalVertex = lv;
                                    else new_cfg_node.OriginalVertex = lv.OriginalVertex;
                                    // Add in rewrites.
                                    new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    new_cfg_node.node_type_map.Add(type_to_consider, xx);
                                    EnterInstantiatedBasicBlock(lv, type_to_consider, xx, new_cfg_node);
                                    System.Console.WriteLine("Adding new node " + new_cfg_node.Name);

                                    // Push this node back on the stack.
                                    instantiated_nodes.Push(new_cfg_node);
                                }
                            }
                        }
                    }
                }
            }

            // Get new nodes.
            List<CFG.Vertex> cs2 = _mcfg.PopChangeSet(change_set_id2);

            // Set up entry flag for every block.
            foreach (var v in cs2)
            {

            }

        }
        

        public void CompileToLLVM(IEnumerable<CFG.Vertex> change_set, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            //
            // Create a basic block and module in LLVM for entry blocks in the CIL graph.
            // Note, we are going to create a basic block unique for each generic type instantiated.
            //
            IEnumerable<CFG.Vertex> mono_bbs = change_set;
            foreach (var lv in mono_bbs)
            {
                // Skip all but entry blocks for now.
                if (!lv.IsEntry)
                    continue;                

                MethodDefinition method = lv.Method;
                var parameters = method.Parameters;

                System.Reflection.MethodBase mb =
                    ReflectionCecilInterop.ConvertToSystemReflectionMethodInfo(method);
                string mn = mb.DeclaringType.Assembly.GetName().Name;
                ModuleRef mod = LLVM.ModuleCreateWithName(mn);
                lv.Module = mod;

                // Further, do not compile nodes for methods that are generic and uninstantiated.
                if (method.HasGenericParameters && lv.node_type_map != null
                    && !lv.node_type_map.Any())
                    continue;
                if (method.ContainsGenericParameter && lv.node_type_map != null
                    && !lv.node_type_map.Any())
                    continue;

                uint count = (uint) mb.GetParameters().Count();
                TypeRef[] param_types = new TypeRef[count];
                int current = 0;
                if (count > 0)
                    foreach (var p in parameters)
                    {
                        param_types[current++] =
                            ConvertMonoTypeToLLVM(
                                p.ParameterType,
                                lv,
                                false);
                    }
                TypeRef ret_type = default(TypeRef);
                var mi2 = method.ReturnType;
                ret_type = ConvertMonoTypeToLLVM(
                    mi2,
                    lv,
                    false);
                TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                ValueRef fun = LLVM.AddFunction(mod, mb.Name, met_type);
                BasicBlockRef entry = LLVM.AppendBasicBlock(fun, lv.Name.ToString());
                lv.BasicBlock = entry;
                lv.Function = fun;
                BuilderRef builder = LLVM.CreateBuilder();
                lv.Builder = builder;
                LLVM.PositionBuilderAtEnd(builder, entry);
            }


            foreach (var mv in mono_bbs)
            {
                IEnumerable<CFG.Vertex> successors = _mcfg.SuccessorNodes(mv);
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
            foreach (CFG.Vertex mv in mono_bbs)
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

            foreach (CFG.Vertex node in mono_bbs)
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

            foreach (CFG.Vertex node in _mcfg.VertexNodes)
            {
                if (node.IsEntry) continue;
                CFG.Vertex e = node.Entry;
                node.HasReturnValue = e.HasReturnValue;
                node.NumberOfArguments = e.NumberOfArguments;
                node.NumberOfLocals = e.NumberOfLocals;
            }

            List<CFG.Vertex> unreachable = new List<CFG.Vertex>();
            {
                // Create DFT order of all nodes.
                IEnumerable<int> objs = entries.Select(x => x.Name);
                GraphAlgorithms.DFSPreorder<int>
                    dfs = new GraphAlgorithms.DFSPreorder<int>(
                        _mcfg,
                        objs
                    );
                List<CFG.Vertex> visited = new List<CFG.Vertex>();
                foreach (int ob in dfs)
                {
                    CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    visited.Add(node);
                }
                foreach (CFG.Vertex v in mono_bbs)
                {
                    if (!visited.Contains(v))
                        unreachable.Add(v);
                }
            }

            List<CFG.Vertex> change_set_minus_unreachable = new List<CFG.Vertex>(mono_bbs);
            foreach (CFG.Vertex v in unreachable)
            {
                if (change_set_minus_unreachable.Contains(v))
                {
                    change_set_minus_unreachable.Remove(v);
                }
            }

            {
                List<CFG.Vertex> work = new List<CFG.Vertex>(change_set_minus_unreachable);
                while (work.Count != 0)
                {
                    // Create DFT order of all nodes.
                    IEnumerable<int> objs = entries.Select(x => x.Name);
                    GraphAlgorithms.DFSPreorder<int>
                        dfs = new GraphAlgorithms.DFSPreorder<int>(
                            _mcfg,
                            objs
                        );

                    List<CFG.Vertex> visited = new List<CFG.Vertex>();
                    // Compute stack size for each basic block, processing nodes on work list
                    // in DFT order.
                    foreach (int ob in dfs)
                    {
                        CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
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
                            CFG.Vertex llvm_nodex = node;
                            llvm_nodex.StackLevelIn = node.NumberOfLocals + node.NumberOfArguments;
                        }
                        else
                        {
                            int in_level = -1;
                            foreach (CFG.Vertex pred in _mcfg.PredecessorNodes(node))
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
                                CFG.Vertex llvm_nodex = node;
                                llvm_nodex.StackLevelIn = llvm_pred.StackLevelOut;
                                in_level = (int) llvm_nodex.StackLevelIn;
                            }
                            // Warn if no predecessors have been visited.
                            if (in_level == -1)
                            {
                                continue;
                            }
                        }
                        CFG.Vertex llvm_nodez = node;
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
                        foreach (CFG.Vertex succ in node._Graph.SuccessorNodes(node))
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
                            CFG.Vertex llvm_succ = node;
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
                List<CFG.Vertex> work = new List<CFG.Vertex>(change_set_minus_unreachable);

                // Get a list of the name of nodes to compile.
                IEnumerable<int> work_names = work.Select(v => v.Name);

                // Get a Tarjan DFS/SCC order of the nodes. Reverse it because we want to
                // proceed from entry basic block.
                //var ordered_list = new Tarjan<int>(_mcfg).GetEnumerable().Reverse();
                var ordered_list = new Tarjan<int>(_mcfg).Reverse();

                // Eliminate all node names not in the work list.
                var order = ordered_list.Where(v => work_names.Contains(v)).ToList();

                // Set up the initial states associated with each node, that is, state into and state out of.
                foreach (int ob in order)
                {
                    CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CFG.Vertex llvm_node = node;
                    llvm_node.StateIn = new State(node, node.Method, llvm_node.NumberOfArguments, llvm_node.NumberOfLocals,
                        (int) llvm_node.StackLevelIn);
                    llvm_node.StateOut = new State(node, node.Method, llvm_node.NumberOfArguments, llvm_node.NumberOfLocals,
                        (int) llvm_node.StackLevelOut);
                }

                Dictionary<int, bool> visited = new Dictionary<int, bool>();

                // Emit LLVM IR code, based on state and per-instruction simulation on that state.
                foreach (int ob in order)
                {
                    CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CFG.Vertex llvm_node = node;

                    var state_in = new State(visited, llvm_node, list_of_data_types_used);
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
                        GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge = llvm_node._Successors[0];
                        int succ = edge.To;
                        var s = llvm_node._Graph.VertexSpace[llvm_node._Graph.NameSpace.BijectFromBasetype(succ)];
                        var br = LLVM.BuildBr(llvm_node.Builder, s.BasicBlock);
                    }
                    visited[ob] = true;
                }

                // Finally, update phi functions with "incoming" information from predecessors.
                foreach (int ob in order)
                {
                    CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CFG.Vertex llvm_node = node;
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
                    CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    CFG.Vertex llvm_node = node;

                    node.OutputEntireNode();
                    llvm_node.StateIn.Dump();
                    llvm_node.StateOut.Dump();
                }
            }

            foreach (var lv in mono_bbs)
            {
                if (lv.IsEntry)
                {
                    ModuleRef mod = lv.Module;
                    LLVM.DumpModule(mod);
                }
            }
        }

        public List<System.Type> FindAllTargets(Delegate obj)
        {
            List<System.Type> data_used = new List<System.Type>();

            Dictionary<Delegate, object> delegate_to_instance = new Dictionary<Delegate, object>();

            Delegate lambda_delegate = (Delegate)obj;

            BindingFlags findFlags = BindingFlags.NonPublic |
                                     BindingFlags.Public |
                                     BindingFlags.Static |
                                     BindingFlags.Instance |
                                     BindingFlags.InvokeMethod |
                                     BindingFlags.OptionalParamBinding |
                                     BindingFlags.DeclaredOnly;

            List<object> processed = new List<object>();

            // Construct list of generic methods with types that will be JIT'ed.
            StackQueue<object> stack = new StackQueue<object>();
            stack.Push(lambda_delegate);

            while (stack.Count > 0)
            {
                object node = stack.Pop();
                if (processed.Contains(node)) continue;

                processed.Add(node);

                // Case 1: object is multicast delegate.
                // A multicast delegate is a list of delegates called in the order
                // they appear in the list.
                System.MulticastDelegate multicast_delegate = node as System.MulticastDelegate;
                if (multicast_delegate != null)
                {
                    foreach (System.Delegate node2 in multicast_delegate.GetInvocationList())
                    {
                        if ((object) node2 != (object) node)
                        {
                            stack.Push(node2);
                        }
                    }
                }

                // Case 2: object is plain delegate.
                System.Delegate plain_delegate = node as System.Delegate;
                if (plain_delegate != null)
                {
                    object target = plain_delegate.Target;
                    if (target == null)
                    {
                        // If target is null, then the delegate is a function that
                        // uses either static data, or does not require any additional
                        // data. If target isn't null, then it's probably a class.
                        target = Activator.CreateInstance(plain_delegate.Method.DeclaringType);
                        if (target != null)
                        {
                            stack.Push(target);
                        }
                    }
                    else
                    {
                        // Target isn't null for delegate. Most likely, the method
                        // is part of the target, so let's assert that.
                        bool found = false;
                        foreach (System.Reflection.MethodInfo mi in target.GetType().GetMethods(findFlags))
                        {
                            if (mi == plain_delegate.Method)
                            {
                                found = true;
                                break;
                            }
                        }
                        Debug.Assert(found);
                        stack.Push(target);
                    }
                    continue;
                }

                if (node != null && (multicast_delegate == null || plain_delegate == null))
                {
                    // This is just a closure object, represented as a class. Go through
                    // the class and record instances of generic types.
                    data_used.Add(node.GetType());

                    // Case 3: object is a class, and potentially could point to delegate.
                    // Examine all fields, looking for list_of_targets.

                    System.Type target_type = node.GetType();

                    FieldInfo[] target_type_fieldinfo = target_type.GetFields();
                    foreach (var field in target_type_fieldinfo)
                    {
                        var value = field.GetValue(node);
                        if (value != null)
                        {
                            if (field.FieldType.IsValueType)
                                continue;
                            // chase pointer type.
                            stack.Push(value);
                        }
                    }
                }
            }

            return data_used;
        }


        public IntPtr GetPtr(int block_number)
        {
            CFG.Vertex here = null;
            foreach (var xxx in _mcfg.VertexNodes)
            {
                if (xxx.IsEntry && xxx.Name == block_number)
                {
                    here = xxx;
                    break;
                }
            }
            CFG.Vertex lvv = here;
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

        //public static TypeRef ConvertMonoTypeToLLVM(TypeDefinition td)
        //{
        //    System.Type sys_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToBasicSystemReflectionType(td);
        //    TypeRef type;
        //    if (sys_type == typeof(System.Int16))
        //    {
        //        type = LLVM.Int16Type();
        //    }
        //    else if (sys_type == typeof(System.Int32))
        //    {
        //        type = LLVM.Int32Type();
        //    }
        //    else if (sys_type == typeof(System.Int64))
        //    {
        //        type = LLVM.Int64Type();
        //    }
        //    else if (sys_type == typeof(System.Boolean))
        //    {
        //        type = LLVM.Int32Type();
        //    }
        //    else if (sys_type == typeof(System.Char))
        //    {
        //        type = LLVM.Int16Type();
        //    }
        //    else throw new Exception("Cannot handle type.");
        //    return type;
        //}

        private static bool setup = true;
        private static Mono.Cecil.TypeDefinition MonoInt16;
        private static Mono.Cecil.TypeDefinition MonoUInt16;
        private static Mono.Cecil.TypeDefinition MonoInt32;
        private static Mono.Cecil.TypeDefinition MonoUInt32;
        private static Mono.Cecil.TypeDefinition MonoInt64;
        private static Mono.Cecil.TypeDefinition MonoUInt64;
        private static Mono.Cecil.TypeDefinition MonoBoolean;
        private static Mono.Cecil.TypeDefinition MonoChar;
        private static Mono.Cecil.TypeDefinition MonoVoid;
        private static Mono.Cecil.TypeDefinition MonoTypeDef;

        public static TypeRef ConvertMonoTypeToLLVM(
            Mono.Cecil.TypeReference tr,
            CFG.Vertex node,
            bool black_box)
        {
            TypeDefinition td = tr.Resolve();
            if (setup)
            {
                MonoInt16 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Int16));
                MonoUInt16 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(UInt16));
                MonoInt32 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Int32));
                MonoUInt32 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(UInt32));
                MonoInt64 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Int64));
                MonoUInt64 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(UInt64));
                MonoBoolean = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Boolean));
                MonoChar = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Char));
                MonoVoid = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(void));
                MonoTypeDef = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Mono.Cecil.TypeDefinition));
                setup = false;
            }

            if (tr == MonoInt16)
            {
                return LLVM.Int16Type();
            }
            else if (tr == MonoUInt16)
            {
                return LLVM.Int16Type();
            }
            else if (tr == MonoInt32)
            {
                return LLVM.Int32Type();
            }
            else if (tr == MonoUInt32)
            {
                return LLVM.Int32Type();
            }
            else if (tr == MonoInt64)
            {
                return LLVM.Int64Type();
            }
            else if (tr == MonoUInt64)
            {
                return LLVM.Int64Type();
            }
            else if (tr == MonoBoolean)
            {
                return LLVM.Int1Type();
            }
            else if (tr == MonoChar)
            {
                return LLVM.Int8Type();
            }
            else if (tr == MonoVoid)
            {
                return LLVM.VoidType();
            }
            else if (tr == MonoTypeDef)
            {
                // Pass on compiling the system type. Too compilicated. For now, just pass void *.
                var typeref = LLVM.VoidType();
                var s = LLVM.PointerType(typeref, 0);
                return s;
            }
            else if (black_box && tr.IsArray)
            {
                // Pass on compiling the system type. Too compilicated. For now, just pass void *.
                var typeref = LLVM.VoidType();
                var s = LLVM.PointerType(typeref, 0);
                return s;
            }
            else if (black_box && td.IsClass)
            {
                // Pass on compiling the system type. Too compilicated. For now, just pass void *.
                var typeref = LLVM.VoidType();
                var s = LLVM.PointerType(typeref, 0);
                return s;
            }
            else if (tr.IsArray)
            {
                ContextRef c = LLVM.ContextCreate();
                TypeRef s = LLVM.StructCreateNamed(c, tr.ToString());
                LLVM.StructSetBody(s, new TypeRef[2]
                {
                    LLVM.PointerType(ConvertMonoTypeToLLVM(tr.GetElementType(), node, false), 0),
                    LLVM.Int64Type()
                }, true);

                var element_type = tr.GetElementType();
                var e = ConvertMonoTypeToLLVM(element_type, node, false);
                var p = LLVM.PointerType(e, 0);
                var d = LLVM.GetUndef(p);
                return s;
            }
            else if (tr.IsGenericParameter)
            {
                foreach (var kvp in node.node_type_map)
                {
                    var key = kvp.Key;
                    var value = kvp.Value;
                    if (key.Name == tr.Name)
                    {
                        // Match, and substitute.
                        var v = value.First();
                        var mv = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(v);
                        var e = ConvertMonoTypeToLLVM(mv, node, true);
                        return e;
                    }
                }
                throw new Exception("Cannot convert " + tr.Name);
            }
            else if (td.IsClass)
            {
                if (tr.HasGenericParameters)
                {
                    // The type is generic. Loop through all data types used in closure to see
                    // how to compile this type.
                    foreach (var kvp in node.node_type_map)
                    {
                        var key = kvp.Key;
                        var value = kvp.Value;

                        if (key.Name == tr.Name)
                        {
                            // match.
                            // Substitute tt for t.
                            //return ConvertSystemTypeToLLVM(tt, list_of_data_types_used, black_box);
                        }
                    }
                }
                // Create a struct/class type.
                ContextRef c = LLVM.ContextCreate();
                TypeRef s = LLVM.StructCreateNamed(c, tr.ToString());
                // Create array of typerefs as argument to StructSetBody below.
                var fields = td.Fields;
                //(
                //    System.Reflection.BindingFlags.Instance
                //    | System.Reflection.BindingFlags.NonPublic
                //    | System.Reflection.BindingFlags.Public
                //    | System.Reflection.BindingFlags.Static);
                List<TypeRef> list = new List<TypeRef>();
                foreach (var field in fields)
                {
                    if (field.FieldType == tr)
                    {
                        list.Add(s);
                        continue;
                    }
                    var field_converted_type = ConvertMonoTypeToLLVM(field.FieldType, node, true);
                    list.Add(field_converted_type);
                }
                LLVM.StructSetBody(s, list.ToArray(), true);
                return s;
            }
            else
            throw new Exception("Unknown type.");
        }
    }
}
