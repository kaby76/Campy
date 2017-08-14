using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Campy.GraphAlgorithms;
using Campy.Graphs;
using Campy.Types.Utils;
using Campy.Utils;
using Mono.Cecil;
using Swigged.LLVM;
using System;
using System.Runtime.InteropServices;
using Swigged.Cuda;

namespace Campy.ControlFlowGraph
{
    public class Converter
    {
        private CFG _mcfg;

        static Dictionary<CFG, Converter> _converters = new Dictionary<CFG, Converter>();

        public static Converter GetConverter(CFG cfg)
        {
            return _converters[cfg];
        }

        public Converter(CFG mcfg)
        {
            _mcfg = mcfg;
            _converters[mcfg] = this;
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

        private CFG.Vertex Eval(CFG.Vertex current, Dictionary<TypeReference, System.Type> ops)
        {
            // Start at current vertex, and find transition state given ops.
            CFG.Vertex result = current;
            for (;;)
            {
                bool found = false;
                foreach(var t in ops)
                {
                    var x = FindInstantiatedBasicBlock(current, t.Key, t.Value);
                    if (x != null)
                    {
                        current = x;
                        found = true;
                        break;
                    }
                }
                if (!found) break;
            }
            return current;
        }

        private bool TypeUsingGeneric()
        { return false; }

        public List<CFG.Vertex> InstantiateGenerics(IEnumerable<CFG.Vertex> change_set, List<System.Type> list_of_data_types_used, List<Mono.Cecil.TypeDefinition> list_of_mono_data_types_used)
        {
            // Start a new change set so we can update edges and other properties for the new nodes
            // in the graph.
            int change_set_id2 = _mcfg.StartChangeSet();

            // We need to do bookkeeping of what nodes to consider.
            Stack<CFG.Vertex> instantiated_nodes = new Stack<CFG.Vertex>(change_set);

            while (instantiated_nodes.Count > 0)
            {
                CFG.Vertex lv = instantiated_nodes.Pop();

                System.Console.WriteLine("Considering " + lv.Name);

                // If a block associated with method contains generics,
                // we need to duplicate the node and add in type information
                // about the generic type with that is actually used.
                // So, for example, if the method contains a parameter of type
                // "T", then we add in a mapping of T to the actual data type
                // used, e.g., Integer, or what have you. When it is compiled,
                // to LLVM, the mapped data type will be used!
                MethodDefinition method = lv.Method;
                var declaring_type = method.DeclaringType;
                var method_has_generics = method.HasGenericParameters;
                var method_contains_generics = method.ContainsGenericParameter;
                var method_generic_instance = method.IsGenericInstance;

                {
                    // Let's first consider the parameter types to the function.
                    var parameters = method.Parameters;
                    for (int k = 0; k < parameters.Count; ++k)
                    {
                        ParameterDefinition par = parameters[k];
                        var type_to_consider = par.ParameterType;
                        var type_to_consider_system_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(type_to_consider);
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

                                    // Match. First find rewrite node if previous created.
                                    var previous = lv;
                                    for (; previous != null; previous = previous.PreviousVertex)
                                    {
                                        var old_node = FindInstantiatedBasicBlock(previous, type_to_consider, xx);
                                        if (old_node != null)
                                            break;
                                    }
                                    if (previous != null) continue;
                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new_node_id);
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = lv.Instructions;
                                    new_cfg_node.Method = lv.Method;
                                    new_cfg_node.PreviousVertex = lv;
                                    new_cfg_node.OpFromPreviousNode = new Tuple<TypeReference, System.Type>(type_to_consider, xx);
                                    var previous_list = lv.OpsFromOriginal;
                                    if (previous_list != null) new_cfg_node.OpsFromOriginal = new Dictionary<TypeReference, System.Type>(previous_list);
                                    else new_cfg_node.OpsFromOriginal = new Dictionary<TypeReference, System.Type>();
                                    new_cfg_node.OpsFromOriginal.Add(new_cfg_node.OpFromPreviousNode.Item1, new_cfg_node.OpFromPreviousNode.Item2);
                                    if (lv.OriginalVertex == null) new_cfg_node.OriginalVertex = lv;
                                    else new_cfg_node.OriginalVertex = lv.OriginalVertex;

                                    // Add in rewrites.
                                    //new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    //new_cfg_node.node_type_map.Add(type_to_consider, xx);
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
                        var type_to_consider_system_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(type_to_consider);
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

                                    // Match. First find rewrite node if previous created.
                                    var previous = lv;
                                    for (; previous != null; previous = previous.PreviousVertex)
                                    {
                                        var old_node = FindInstantiatedBasicBlock(previous, type_to_consider, xx);
                                        if (old_node != null)
                                            break;
                                    }
                                    if (previous != null) continue;
                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new_node_id);
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = lv.Instructions;
                                    new_cfg_node.Method = lv.Method;
                                    new_cfg_node.PreviousVertex = lv;
                                    new_cfg_node.OpFromPreviousNode = new Tuple<TypeReference, System.Type>(type_to_consider, xx);
                                    var previous_list = lv.OpsFromOriginal;
                                    if (previous_list != null) new_cfg_node.OpsFromOriginal = new Dictionary<TypeReference, System.Type>(previous_list);
                                    else new_cfg_node.OpsFromOriginal = new Dictionary<TypeReference, System.Type>();
                                    new_cfg_node.OpsFromOriginal.Add(new_cfg_node.OpFromPreviousNode.Item1, new_cfg_node.OpFromPreviousNode.Item2);
                                    if (lv.OriginalVertex == null) new_cfg_node.OriginalVertex = lv;
                                    else new_cfg_node.OriginalVertex = lv.OriginalVertex;
                                    
                                    // Add in rewrites.
                                    //new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    //new_cfg_node.node_type_map.Add(type_to_consider, xx);
                                    EnterInstantiatedBasicBlock(lv, type_to_consider, xx, new_cfg_node);
                                    System.Console.WriteLine("Adding new node " + new_cfg_node.Name);

                                    // Push this node back on the stack.
                                    instantiated_nodes.Push(new_cfg_node);
                                }
                            }
                        }
                    }
                }

                {
                    // Let's consider "this" to the function.
                    var has_this = method.HasThis;
                    if (has_this)
                    {
                        var type_to_consider = method.DeclaringType;
                        var type_to_consider_system_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(type_to_consider);
                        if (type_to_consider.ContainsGenericParameter)
                        {
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

                                if (type_to_consider.FullName.Equals(data_type_used.FullName))
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

                                    // Match. First find rewrite node if previous created.
                                    var previous = lv;
                                    for (; previous != null; previous = previous.PreviousVertex)
                                    {
                                        var old_node = FindInstantiatedBasicBlock(previous, type_to_consider, xx);
                                        if (old_node != null)
                                            break;
                                    }
                                    if (previous != null) continue;
                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new_node_id);
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = lv.Instructions;
                                    new_cfg_node.Method = lv.Method;
                                    new_cfg_node.PreviousVertex = lv;
                                    new_cfg_node.OpFromPreviousNode = new Tuple<TypeReference, System.Type>(type_to_consider, xx);
                                    var previous_list = lv.OpsFromOriginal;
                                    if (previous_list != null) new_cfg_node.OpsFromOriginal = new Dictionary<TypeReference, System.Type>(previous_list);
                                    else new_cfg_node.OpsFromOriginal = new Dictionary<TypeReference, System.Type>();
                                    new_cfg_node.OpsFromOriginal.Add(new_cfg_node.OpFromPreviousNode.Item1, new_cfg_node.OpFromPreviousNode.Item2);
                                    if (lv.OriginalVertex == null) new_cfg_node.OriginalVertex = lv;
                                    else new_cfg_node.OriginalVertex = lv.OriginalVertex;

                                    // Add in rewrites.
                                    //new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    //new_cfg_node.node_type_map.Add(type_to_consider, xx);
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

            List<CFG.Vertex> new_change_set = _mcfg.PopChangeSet(change_set_id2);
            Dictionary<CFG.Vertex, CFG.Vertex> map_to_new_block = new Dictionary<CFG.Vertex, CFG.Vertex>();
            foreach (var v in new_change_set)
            {
                if (!IsFullyInstantiatedNode(v)) continue;
                var original = v.OriginalVertex;
                var ops_list = v.OpsFromOriginal;
                // Apply instance information from v onto predecessors and successors, and entry.
                foreach (var vto in _mcfg.SuccessorNodes(original))
                {
                    var vto_mapped = Eval(vto, ops_list);
                    _mcfg.AddEdge(v, vto_mapped);
                }
            }
            foreach (var v in new_change_set)
            {
                if (!IsFullyInstantiatedNode(v)) continue;
                var original = v.OriginalVertex;
                var ops_list = v.OpsFromOriginal;
                if (original.Entry != null)
                    v.Entry = Eval(original.Entry, ops_list);
            }

            this._mcfg.OutputEntireGraph();

            List<CFG.Vertex> result = new List<CFG.Vertex>();
            result.AddRange(change_set);
            result.AddRange(new_change_set);
            return result;
        }

        public bool IsFullyInstantiatedNode(CFG.Vertex node)
        {
            bool result = false;
            // First, go through and mark all nodes that have non-null
            // previous entries.

            Dictionary<CFG.Vertex, bool> instantiated = new Dictionary<CFG.Vertex, bool>();
            foreach (var v in _mcfg.VertexNodes)
            {
                instantiated[v] = true;
            }
            foreach (var v in _mcfg.VertexNodes)
            {
                if (v.PreviousVertex != null) instantiated[v.PreviousVertex] = false;
            }
            result = instantiated[node];
            return result;
        }

        public static ModuleRef global_module = default(ModuleRef);
        private List<ModuleRef> all_modules = new List<ModuleRef>();
        private ModuleRef CreateModule(string name)
        {
            var new_module = LLVM.ModuleCreateWithName(name);
            all_modules.Add(new_module);
            return new_module;
        }

        private void CompilePart1(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            global_module = CreateModule("global");
            LLVM.EnablePrettyStackTrace();
            var triple = LLVM.GetDefaultTargetTriple();
            LLVM.SetTarget(global_module, triple);
            LLVM.InitializeAllTargets();
            LLVM.InitializeAllTargetMCs();
            LLVM.InitializeAllTargetInfos();
            LLVM.InitializeAllAsmPrinters();

            foreach (var bb in basic_blocks_to_compile)
            {
                System.Console.WriteLine("Compile part 1, node " + bb);

                // Skip all but entry blocks for now.
                if (!bb.IsEntry)
                {
                    System.Console.WriteLine("skipping -- not an entry.");
                    continue;
                }

                if (!IsFullyInstantiatedNode(bb))
                {
                    System.Console.WriteLine("skipping -- not fully instantiated block the contains generics.");
                    continue;
                }

                MethodDefinition method = bb.Method;
                bb.HasThis = method.HasThis;
                var parameters = method.Parameters;
                System.Reflection.MethodBase mb = ReflectionCecilInterop.ConvertToSystemReflectionMethodInfo(method);
                string mn = mb.DeclaringType.Assembly.GetName().Name;
                ModuleRef mod = global_module; // LLVM.ModuleCreateWithName(mn);
                bb.Module = mod;
                uint count = (uint)mb.GetParameters().Count();
                if (bb.HasThis) count++;
                TypeRef[] param_types = new TypeRef[count];
                int current = 0;
                if (count > 0)
                {
                    if (bb.HasThis)
                        param_types[current++] = ConvertMonoTypeToLLVM(bb,
                            Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(mb.DeclaringType), bb.LLVMTypeMap, bb.OpsFromOriginal);
                    foreach (var p in parameters)
                        param_types[current++] = ConvertMonoTypeToLLVM(bb,
                            p.ParameterType, bb.LLVMTypeMap, bb.OpsFromOriginal);
                    foreach (var pp in param_types)
                    {
                        string a = LLVM.PrintTypeToString(pp);
                        System.Console.WriteLine(" " + a);
                    }
                }

                TypeRef ret_type = default(TypeRef);
                var mi2 = method.ReturnType;
                ret_type = ConvertMonoTypeToLLVM(bb, mi2, bb.LLVMTypeMap, bb.OpsFromOriginal);
                TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                ValueRef fun = LLVM.AddFunction(mod, Converter.MethodName(method), met_type);
                BasicBlockRef entry = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                bb.BasicBlock = entry;
                bb.Function = fun;
                BuilderRef builder = LLVM.CreateBuilder();
                bb.Builder = builder;
                LLVM.PositionBuilderAtEnd(builder, entry);
            }
        }

        private void CompilePart2(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            foreach (var bb in basic_blocks_to_compile)
            {
                if (!IsFullyInstantiatedNode(bb))
                    continue;

                IEnumerable<CFG.Vertex> successors = _mcfg.SuccessorNodes(bb);
                if (!bb.IsEntry)
                {
                    var ent = bb.Entry;
                    var lvv_ent = ent;
                    var fun = lvv_ent.Function;
                    var llvm_bb = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                    bb.BasicBlock = llvm_bb;
                    bb.Function = lvv_ent.Function;
                    BuilderRef builder = LLVM.CreateBuilder();
                    bb.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, llvm_bb);
                }
            }
        }

        private void CompilePart3(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            foreach (CFG.Vertex bb in basic_blocks_to_compile)
            {
                if (!IsFullyInstantiatedNode(bb))
                    continue;

                Inst prev = null;
                foreach (var j in bb.Instructions)
                {
                    j.Block = bb;
                    if (prev != null) prev.Next = j;
                    prev = j;
                }
            }
        }


        private void CompilePart4(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeDefinition> list_of_data_types_used, List<CFG.Vertex> entries,
            out List<CFG.Vertex> unreachable, out List<CFG.Vertex> change_set_minus_unreachable)
        {
            unreachable = new List<CFG.Vertex>();
            change_set_minus_unreachable = new List<CFG.Vertex>(basic_blocks_to_compile);
            {
                // Create DFT order of all nodes from entries.
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
                    if (!IsFullyInstantiatedNode(node))
                        continue;
                    visited.Add(node);
                }
                foreach (CFG.Vertex v in basic_blocks_to_compile)
                {
                    if (!visited.Contains(v))
                        unreachable.Add(v);
                }

                foreach (CFG.Vertex v in unreachable)
                {
                    if (change_set_minus_unreachable.Contains(v))
                    {
                        change_set_minus_unreachable.Remove(v);
                    }
                }
            }
        }

        public static string MethodName(MethodReference mr)
        {
            // Method names for a method reference are sometimes not the
            // same, even though they are in principle referring to the same
            // method, especially for methods that contain generics. This function
            // returns a normalized name for the method reference so that there
            // is equivalence.
            var declaring_type = mr.DeclaringType;
            if (declaring_type == null) throw new Exception("Cannot get declaring type for method.");
            var r = declaring_type.Resolve();
            var methods = r.Methods;
            foreach (var method in methods)
            {
                if (method.Name == mr.Name)
                    return method.FullName;
            }
            return null;
        }

        public static CFG.Vertex FindFullyInstantiatedMethod(MethodReference mr,
            Dictionary<TypeReference, TypeReference> map)
        {
            return null;
        }

        private void AddExternFunction(CFG.Vertex bb, CFG.Vertex called_bb)
        {
            return;
            var mr = called_bb.Method;
            MethodDefinition method = mr.Resolve();
            var mb = method.DeclaringType;
            var name = Converter.MethodName(method);
            var parameters = method.Parameters;
            uint count = (uint)method.Parameters.Count();
            if (method.HasThis) count++;
            TypeRef[] param_types = new TypeRef[count];
            int current = 0;
            if (count > 0)
            {
                if (method.HasThis)
                    param_types[current++] = ConvertMonoTypeToLLVM(bb,
                        mb, bb.LLVMTypeMap, bb.OpsFromOriginal);
                foreach (var p in parameters)
                    param_types[current++] = ConvertMonoTypeToLLVM(bb,
                        p.ParameterType, bb.LLVMTypeMap, bb.OpsFromOriginal);
            }
            TypeRef ret_type = default(TypeRef);
            var mi2 = method.ReturnType;
            ret_type = ConvertMonoTypeToLLVM(called_bb, mi2, called_bb.LLVMTypeMap, called_bb.OpsFromOriginal);
            TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
            ValueRef fun = LLVM.AddFunction(bb.Module,
                Converter.MethodName(method), met_type);
        }

        private void CompilePart5(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            return;
            // In all entries (that is, a basic block which in an entry and
            // has an LLVM Module created), declare external functions. Note,
            // every function is defined in its own Module. We have to do this
            // due to a restriction in LLVM.
            foreach (CFG.Vertex bb in basic_blocks_to_compile)
            {
                if (!IsFullyInstantiatedNode(bb))
                    continue;

                foreach (Inst caller in Inst.CallInstructions)
                {
                    CFG.Vertex n = caller.Block;
                    if (n != bb) continue;
                    System.Console.WriteLine(caller);
                    object method = caller.Operand;
                    if (method as Mono.Cecil.MethodReference == null) throw new Exception("Cannot cast call instruction operand!");
                    Mono.Cecil.MethodReference mr = method as Mono.Cecil.MethodReference;
                    var name = MethodName(mr);
                    // Find bb entry.
                    CFG.Vertex the_entry = caller.Block._Graph.VertexNodes.Where(node
                        =>
                    {
                        GraphLinkedList<int, CFG.Vertex, CFG.Edge> g = caller.Block._Graph;
                        int k = g.NameSpace.BijectFromBasetype(node.Name);
                        CFG.Vertex v = g.VertexSpace[k];
                        Converter c = Converter.GetConverter((CFG)g);
                        if (v.IsEntry && MethodName(v.Method) == name && c.IsFullyInstantiatedNode(v))
                            return true;
                        else return false;
                    }).ToList().FirstOrDefault();
                    System.Console.WriteLine("Found " + the_entry);

                    // Within the basic block bb, set up declaration of method.
                    AddExternFunction(bb, the_entry);
                }
            }
        }

        public void CompileToLLVM(List<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeDefinition> list_of_data_types_used)
        {
            CompilePart1(basic_blocks_to_compile, list_of_data_types_used);

            CompilePart2(basic_blocks_to_compile, list_of_data_types_used);

            CompilePart3(basic_blocks_to_compile, list_of_data_types_used);

            List<CFG.Vertex> entries = _mcfg.VertexNodes.Where(node => node.IsEntry).ToList();

            foreach (CFG.Vertex node in basic_blocks_to_compile)
            {
                if (!IsFullyInstantiatedNode(node))
                    continue;

                int args = 0;
                Mono.Cecil.MethodDefinition md = node.Method;
                Mono.Cecil.MethodReference mr = node.Method;
                args += mr.Parameters.Count;
                node.NumberOfArguments = args;
                node.HasThis = mr.HasThis;
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
                if (node.IsEntry)
                    continue;
                CFG.Vertex e = node.Entry;
                node.HasReturnValue = e.HasReturnValue;
                node.NumberOfArguments = e.NumberOfArguments;
                node.NumberOfLocals = e.NumberOfLocals;
                node.HasThis = e.HasThis;
            }

            List<CFG.Vertex> unreachable;
            List<CFG.Vertex> change_set_minus_unreachable;
            CompilePart4(basic_blocks_to_compile, list_of_data_types_used, entries, out unreachable, out change_set_minus_unreachable);

            CompilePart5(basic_blocks_to_compile, list_of_data_types_used);


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
                            llvm_nodex.StackLevelIn = node.NumberOfLocals + node.NumberOfArguments + (node.HasThis ? 1 : 0);
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
                            Debug.Assert(level_after >= node.NumberOfLocals + node.NumberOfArguments
                                 + (node.HasThis ? 1 : 0));
                        }
                        llvm_nodez.StackLevelOut = level_after;
                        // Verify return node that it makes sense.
                        if (node.IsReturn && !unreachable.Contains(node))
                        {
                            if (llvm_nodez.StackLevelOut ==
                                node.NumberOfArguments
                                + node.NumberOfLocals
                                + (node.HasThis ? 1 : 0)
                                + (node.HasReturnValue ? 1 : 0))
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
                    CFG.Vertex bb = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                    System.Console.WriteLine("State computations for node " + bb.Name);

                    var state_in = new State(visited, bb, list_of_data_types_used);
                    System.Console.WriteLine("state in output");
                    state_in.Dump();

                    bb.StateIn = state_in;
                    bb.StateOut = new State(state_in);

                    bb.OutputEntireNode();
                    state_in.Dump();

                    Inst last_inst = null;
                    for (int i = 0; i < bb.Instructions.Count; ++i)
                    {
                        var inst = bb.Instructions[i];
                        System.Console.WriteLine(inst);
                        last_inst = inst;
                        inst = inst.Convert(bb.StateOut);
                        bb.StateOut.Dump();
                    }
                    if (last_inst != null && last_inst.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Next)
                    {
                        // Need to insert instruction to branch to fall through.
                        GraphLinkedList<int, CFG.Vertex, CFG.Edge>.Edge edge = bb._Successors[0];
                        int succ = edge.To;
                        var s = bb._Graph.VertexSpace[bb._Graph.NameSpace.BijectFromBasetype(succ)];
                        var br = LLVM.BuildBr(bb.Builder, s.BasicBlock);
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

            foreach (var m in all_modules)
            {
                LLVM.DumpModule(m);
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


        public unsafe IntPtr GetPtr(int block_number)
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

            mod = Converter.global_module;

            MyString error = new MyString();
            LLVM.VerifyModule(mod, VerifierFailureAction.AbortProcessAction, error);
            System.Console.WriteLine(error.ToString());
            ExecutionEngineRef engine;
            LLVM.DumpModule(mod);

            string triple = "nvptx64-nvidia-cuda";
            TargetRef t2;
            var b = LLVM.GetTargetFromTriple(triple, out t2, error);

            string cpu = "";
            string features = "";

            TargetMachineRef tmr = LLVM.CreateTargetMachine(
                t2, triple, cpu, features,
                CodeGenOptLevel.CodeGenLevelDefault,
                RelocMode.RelocDefault,
                CodeModel.CodeModelKernel);
            ContextRef context_ref = LLVM.ContextCreate();
            ValueRef kernelMd = LLVM.MDNodeInContext(
                context_ref, new ValueRef[3]
            {
                lvv.Function,
                LLVM.MDStringInContext(context_ref, "kernel", 6),
                LLVM.ConstInt(LLVM.Int32TypeInContext(context_ref), 1, false)
            });
            LLVM.AddNamedMetadataOperand(mod, "nvvm.annotations", kernelMd);
            var y1 = LLVM.TargetMachineEmitToMemoryBuffer(
                tmr,
                mod,
                Swigged.LLVM.CodeGenFileType.AssemblyFile,
                error,
                out MemoryBufferRef buffer);
            string kernel = null;
            try
            {
                kernel = LLVM.GetBufferStart(buffer);
                uint length = LLVM.GetBufferSize(buffer);
                // Output the PTX assembly code. We can run this using the CUDA Driver API
                System.Console.WriteLine(kernel);
            }
            finally
            {
                LLVM.DisposeMemoryBuffer(buffer);
            }

            // Compile.
            Cuda.cuInit(0);

            // Device api.
            var res = Cuda.cuDeviceGet(out int device, 0);
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            res = Cuda.cuDeviceGetPCIBusId(out string pciBusId, 100, device);
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            res = Cuda.cuDeviceGetName(out string name, 100, device);
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();

            res = Cuda.cuCtxCreate_v2(out CUcontext cuContext, 0, device);
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            IntPtr ptr = Marshal.StringToHGlobalAnsi(kernel);
            res = Cuda.cuModuleLoadData(out CUmodule cuModule, ptr);
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            res = Cuda.cuModuleGetFunction(out CUfunction helloWorld, cuModule, "_Z4kernPi");
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            int[] v = { 'G', 'd', 'k', 'k', 'n', (char)31, 'v', 'n', 'q', 'k', 'c' };
            GCHandle handle = GCHandle.Alloc(v, GCHandleType.Pinned);
            IntPtr pointer = IntPtr.Zero;
            pointer = handle.AddrOfPinnedObject();
            res = Cuda.cuMemAlloc_v2(out IntPtr dptr, 11 * sizeof(int));
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            res = Cuda.cuMemcpyHtoD_v2(dptr, pointer, 11 * sizeof(int));
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();

            IntPtr[] x = new IntPtr[] { dptr };
            GCHandle handle2 = GCHandle.Alloc(x, GCHandleType.Pinned);
            IntPtr pointer2 = IntPtr.Zero;
            pointer2 = handle2.AddrOfPinnedObject();

            IntPtr[] kp = new IntPtr[] { pointer2 };
            fixed (IntPtr* kernelParams = kp)
            {
                res = Cuda.cuLaunchKernel(helloWorld,
                    1, 1, 1, // grid has one block.
                    11, 1, 1, // block has 11 threads.
                    0, // no shared memory
                    default(CUstream),
                    (IntPtr)kernelParams,
                    (IntPtr)IntPtr.Zero
                );
            }
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            res = Cuda.cuMemcpyDtoH_v2(pointer, dptr, 11 * sizeof(int));
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            Cuda.cuCtxDestroy_v2(cuContext);
            return default(IntPtr);

            //LLVM.LinkInMCJIT();
            //LLVM.InitializeNativeTarget();
            //LLVM.InitializeNativeAsmPrinter();
            //MCJITCompilerOptions options = new MCJITCompilerOptions();
            //var optionsSize = (4 * sizeof(int)) + IntPtr.Size; // LLVMMCJITCompilerOptions has 4 ints and a pointer
            //LLVM.InitializeMCJITCompilerOptions(options, (uint) optionsSize);
            //LLVM.CreateMCJITCompilerForModule(out engine, mod, options, (uint) optionsSize, error);
            //var ptr = LLVM.GetPointerToGlobal(engine, lvv.Function);
            //IntPtr p = (IntPtr) ptr;

            //return p;
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
	    private static Mono.Cecil.TypeDefinition MonoSystemType;
	    private static Mono.Cecil.TypeDefinition MonoSystemString;
        private static Dictionary<TypeReference, TypeRef> previous_llvm_types_created_global = new Dictionary<TypeReference, TypeRef>();
        private static Stack<bool> nested = new Stack<bool>();

        public static TypeRef ConvertMonoTypeToLLVM(
            CFG.Vertex node,
            Mono.Cecil.TypeReference tr,
            Dictionary<TypeReference, TypeRef> previous_llvm_types_created,
            Dictionary<TypeReference, System.Type> generic_type_rewrite_rules)
        {
            // Search for type if already converted.
            foreach (var kv in previous_llvm_types_created_global)
            {
                if (kv.Key.Name == tr.Name)
                {
                    if (nested.Any())
                    {
                        // LLVM cannot handle recursive types.. For now, if nested, make it void *.
                        var typeref = LLVM.VoidType();
                        var s = LLVM.PointerType(typeref, 0);
                        return s;
                    }
                    return kv.Value;
                }
            }
            foreach (var kv in previous_llvm_types_created)
            {
                if (kv.Key.Name == tr.Name)
                    return kv.Value;
            }
            TypeDefinition td = tr.Resolve();
            if (setup)
            {
                MonoInt16 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Int16));
                if (MonoInt16 == null) throw new Exception("Bad initialization");
                MonoUInt16 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(UInt16));
                if (MonoUInt16 == null) throw new Exception("Bad initialization");
                MonoInt32 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Int32));
                if (MonoInt32 == null) throw new Exception("Bad initialization");
                MonoUInt32 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(UInt32));
                if (MonoUInt32 == null) throw new Exception("Bad initialization");
                MonoInt64 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Int64));
                if (MonoInt64 == null) throw new Exception("Bad initialization");
                MonoUInt64 = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(UInt64));
                if (MonoUInt64 == null) throw new Exception("Bad initialization");
                MonoBoolean = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Boolean));
                if (MonoBoolean == null) throw new Exception("Bad initialization");
                MonoChar = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Char));
                if (MonoChar == null) throw new Exception("Bad initialization");
                MonoVoid = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(void));
                if (MonoVoid == null) throw new Exception("Bad initialization");
                MonoTypeDef = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(Mono.Cecil.TypeDefinition));
                if (MonoTypeDef == null) throw new Exception("Bad initialization");
		        MonoSystemType = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(System.Type));
		        if (MonoSystemType == null) throw new Exception("Bad initialization");
		        MonoSystemString = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(typeof(System.String));
		        if (MonoSystemString == null) throw new Exception("Bad initialization");
                setup = false;
            }
            // Check basic types using TypeDefinition's found and initialized in the above code.
            if (td != null)
            {
                if (td.FullName == MonoInt16.FullName)
                {
                    return LLVM.Int16Type();
                }
                else if (td.FullName == MonoUInt16.FullName)
                {
                    return LLVM.Int16Type();
                }
                else if (td.FullName == MonoInt32.FullName)
                {
                    return LLVM.Int32Type();
                }
                else if (td.FullName == MonoUInt32.FullName)
                {
                    return LLVM.Int32Type();
                }
                else if (td.FullName == MonoInt64.FullName)
                {
                    return LLVM.Int64Type();
                }
                else if (td.FullName == MonoUInt64.FullName)
                {
                    return LLVM.Int64Type();
                }
                else if (td.FullName == MonoBoolean.FullName)
                {
                    return LLVM.Int1Type();
                }
                else if (td.FullName == MonoChar.FullName)
                {
                    return LLVM.Int8Type();
                }
                else if (td.FullName == MonoVoid.FullName)
                {
                    return LLVM.VoidType();
                }
                else if (td.FullName == MonoTypeDef.FullName)
                {
                    // Pass on compiling the system type. Too compilicated. For now, just pass void *.
                    var typeref = LLVM.VoidType();
                    var s = LLVM.PointerType(typeref, 0);
                    return s;
                }
		        else if (td.FullName == MonoSystemType.FullName)
		        {
			        var typeref = LLVM.VoidType();
			        var s = LLVM.PointerType(typeref, 0);
			        return s;
		        }
		        else if (td.FullName == MonoSystemString.FullName)
		        {
			        var typeref = LLVM.VoidType();
			        var s = LLVM.PointerType(typeref, 0);
			        return s;
		        }
            }
            
            if (tr.IsArray)
            {
                ContextRef c = LLVM.ContextCreate();
                TypeRef s = LLVM.StructCreateNamed(c, tr.ToString());
                previous_llvm_types_created_global.Add(tr, s);
                LLVM.StructSetBody(s, new TypeRef[2]
                {
                    LLVM.PointerType(ConvertMonoTypeToLLVM(node, tr.GetElementType(), previous_llvm_types_created, generic_type_rewrite_rules), 0),
                    LLVM.Int64Type()
                }, true);

                var element_type = tr.GetElementType();
                var e = ConvertMonoTypeToLLVM(node, element_type, previous_llvm_types_created, generic_type_rewrite_rules);
                var p = LLVM.PointerType(e, 0);
                var d = LLVM.GetUndef(p);
                return s;
            }
            else if (tr.IsGenericParameter)
            {
                foreach (var kvp in generic_type_rewrite_rules)
                {
                    var key = kvp.Key;
                    var value = kvp.Value;
                    if (key.Name == tr.Name)
                    {
                        // Match, and substitute.
                        var v = value;
                        var mv = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(v);
                        var e = ConvertMonoTypeToLLVM(node, mv, previous_llvm_types_created, generic_type_rewrite_rules);
                        previous_llvm_types_created_global.Add(tr, e);
                        return e;
                    }
                }
                throw new Exception("Cannot convert " + tr.Name);
            }
            else if (td != null && td.IsClass)
            {
                nested.Push(true); // LLVM cannot handle recursive types.. For now, if nested, make it void *.
                Dictionary<TypeReference, System.Type> additional = new Dictionary<TypeReference, System.Type>();
                var gp = tr.GenericParameters;
                GenericInstanceType git = tr as GenericInstanceType;
                Mono.Collections.Generic.Collection<TypeReference> ga = null;
                if (git != null)
                {
                    ga = git.GenericArguments;
                    Mono.Collections.Generic.Collection<GenericParameter> gg = td.GenericParameters;
                    // Map parameter to instantiated type.
                    for (int i = 0; i < gg.Count; ++i)
                    {
                        var pp = gg[i];
                        var qq = ga[i];
                        TypeReference trrr = pp as TypeReference;
                        var system_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(qq);
                        if (system_type == null) throw new Exception("Failed to convert " + qq);
                        additional[pp] = system_type;
                    }
                }

                if (tr.HasGenericParameters)
                {
                    // The type is generic. Loop through all data types used in closure to see
                    // how to compile this type.
                    foreach (var kvp in generic_type_rewrite_rules)
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
                var p = LLVM.PointerType(s, 0);
                previous_llvm_types_created_global.Add(tr, p);
                // Create array of typerefs as argument to StructSetBody below.
                var fields = td.Fields;
                var new_list = new Dictionary<TypeReference, System.Type>(generic_type_rewrite_rules);
                foreach (var a in additional) new_list.Add(a.Key, a.Value);
                //(
                //    System.Reflection.BindingFlags.Instance
                //    | System.Reflection.BindingFlags.NonPublic
                //    | System.Reflection.BindingFlags.Public
                //    | System.Reflection.BindingFlags.Static);
                List<TypeRef> list = new List<TypeRef>();
                foreach (var field in fields)
                {
                    var field_converted_type = ConvertMonoTypeToLLVM(node, field.FieldType, previous_llvm_types_created, new_list);
                    list.Add(field_converted_type);
                }
                LLVM.StructSetBody(s, list.ToArray(), true);
                System.Console.WriteLine("Created class for node " + node.Name + " :::: " + LLVM.PrintTypeToString(s));
                nested.Pop();
                return p;
            }
            else
                throw new Exception("Unknown type.");
        }
    }
}
