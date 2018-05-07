using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using Campy.Graphs;
using Campy.Utils;
using Mono.Cecil;
using Swigged.LLVM;
using System.Runtime.InteropServices;
using Mono.Cecil.Cil;
using Swigged.Cuda;
using Mono.Cecil.Rocks;
using Mono.Collections.Generic;
using FieldAttributes = Mono.Cecil.FieldAttributes;
using MethodBody = Mono.Cecil.Cil.MethodBody;

namespace Campy.Compiler
{
    public static class JIT_HELPER
    {
        public static TypeRef ToTypeRef(
            this Mono.Cecil.TypeReference tr,
            Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> generic_type_rewrite_rules = null,
            int level = 0)
        {
            if (generic_type_rewrite_rules == null) generic_type_rewrite_rules = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();

            // Search for type if already converted. Note, there are several caches to search, each
            // containing types with different properties.
            // Also, NB: we use full name for the conversion, as types can be similarly named but within
            // different owning classes.
            foreach (var kv in JITER.basic_llvm_types_created)
            {
                if (kv.Key.FullName == tr.FullName)
                {
                    return kv.Value;
                }
            }
            foreach (var kv in JITER.previous_llvm_types_created_global)
            {
                if (kv.Key.FullName == tr.FullName)
                    return kv.Value;
            }
            foreach (var kv in JITER.previous_llvm_types_created_global)
            {
                if (kv.Key.FullName == tr.FullName)
                    return kv.Value;
            }

            tr = RUNTIME.RewriteType(tr);

            try
            {
                // Check basic types using TypeDefinition.
                // I don't know why, but Resolve() of System.Int32[] (an arrary) returns a simple System.Int32, not
                // an array. If always true, then use TypeReference as much as possible.
                // Resolve() also warps pointer types into just the element type. Really bad design in Mono!
                // All I want is just the frigging information about the type. For example, to know if the
                // type is a class, you have to convert it to a TypeDefinition because a TypeReference does
                // not have IsClass property! Really really really poor design for a type system. Sure, keep
                // a basic understanding of applied and defining occurences, but please, keep type information
                // about! I cannot rail enough with this half-baked type system in Mono. It has caused so many
                // problems!!!!!!!

                TypeDefinition td = tr.Resolve();

                var is_pointer = tr.IsPointer;
                var is_reference = tr.IsByReference;
                var is_array = tr.IsArray;
                var is_value_type = tr.IsValueType;


                if (is_pointer)
                {
                    
                }

                if (is_reference)
                {
                    // Convert the base type first.
                    var base_type = ToTypeRef(td, generic_type_rewrite_rules, level + 1);
                    // Add in pointer to type.
                    TypeRef p = LLVM.PointerType(base_type, 0);
                    return p;
                }

                GenericInstanceType git = tr as GenericInstanceType;
                TypeDefinition gtd = tr as TypeDefinition;

                // System.Array is not considered an "array", rather a "class". So, we need to handle
                // this type.
                if (tr.FullName == "System.Array")
                {
                    // Create a basic int[] and call it the day.
                    var original_tr = tr;

                    tr = typeof(int[]).ToMonoTypeReference();
                    var p = tr.ToTypeRef(generic_type_rewrite_rules, level + 1);
                    JITER.previous_llvm_types_created_global.Add(original_tr, p);
                    return p;
                }
                else if (tr.IsArray)
                {
                    // Note: mono_type_reference.GetElementType() is COMPLETELY WRONG! It does not function the same
                    // as system_type.GetElementType(). Use ArrayType.ElementType!
                    var array_type = tr as ArrayType;
                    var element_type = array_type.ElementType;
                    // ContextRef c = LLVM.ContextCreate();
                    ContextRef c = LLVM.GetModuleContext(JITER.global_llvm_module);
                    string type_name = JITER.RenameToLegalLLVMName(tr.ToString());
                    TypeRef s = LLVM.StructCreateNamed(c, type_name);
                    TypeRef p = LLVM.PointerType(s, 0);
                    JITER.previous_llvm_types_created_global.Add(tr, p);
                    var e = ToTypeRef(element_type, generic_type_rewrite_rules, level + 1);
                    LLVM.StructSetBody(s, new TypeRef[3]
                    {
                        LLVM.PointerType(e, 0)
                        , LLVM.Int64Type()
                        , LLVM.Int64Type()
                    }, true);
                    return p;
                }
                else if (tr.IsGenericParameter)
                {
                    foreach (var kvp in generic_type_rewrite_rules)
                    {
                        Tuple<TypeReference, GenericParameter> key = kvp.Key;
                        var value = kvp.Value;
                        if (key.Item1.FullName == tr.FullName // NOT COMPLETE!
                            )
                        {
                            // Match, and substitute.
                            var v = value;
                            var mv = v.ToMonoTypeReference();
                            var e = ToTypeRef(mv, generic_type_rewrite_rules, level + 1);
                            JITER.previous_llvm_types_created_global.Add(tr, e);
                            return e;
                        }
                    }
                    throw new Exception("Cannot convert " + tr.Name);
                }
                else if (td != null && td.IsEnum)
                {
                    // Enums are any underlying type, e.g., one of { bool, char, int8,
                    // unsigned int8, int16, unsigned int16, int32, unsigned int32, int64, unsigned int64, native int,
                    // unsigned native int }.
                    var bas = td.BaseType;
                    Collection<FieldDefinition> fields = td.Fields;
                    if (fields == null)
                        throw new Exception("Cannot convert " + tr.Name);
                    if (fields.Count == 0)
                        throw new Exception("Cannot convert " + tr.Name);
                    FieldDefinition field = fields[0];
                    if (field == null)
                        throw new Exception("Cannot convert " + tr.Name);
                    var field_type = field.FieldType;
                    if (field_type == null)
                        throw new Exception("Cannot convert " + tr.Name);
                    var va = ToTypeRef(field_type, generic_type_rewrite_rules, level + 1);
                    return va;
                }
                else if (td != null && td.IsValueType)
                {
                    // Struct!!!!!
                    Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> additional = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();
                    var gp = tr.GenericParameters;
                    Mono.Collections.Generic.Collection<TypeReference> ga = null;
                    if (git != null)
                    {
                        ga = git.GenericArguments;
                        Mono.Collections.Generic.Collection<GenericParameter> gg = td.GenericParameters;
                        // Map parameter to instantiated type.
                        for (int i = 0; i < gg.Count; ++i)
                        {
                            GenericParameter pp = gg[i];
                            TypeReference qq = ga[i];
                            TypeReference trrr = pp as TypeReference;
                            var system_type = qq.ToSystemType();
                            Tuple<TypeReference, GenericParameter> tr_gp = new Tuple<TypeReference, GenericParameter>(tr, pp);
                            if (system_type == null) throw new Exception("Failed to convert " + qq);
                            additional[tr_gp] = system_type;
                        }
                    }

                    // Create a struct type.
                    ContextRef c = LLVM.GetModuleContext(JITER.global_llvm_module);
                    string llvm_name = JITER.RenameToLegalLLVMName(tr.ToString());

                    TypeRef s = LLVM.StructCreateNamed(c, llvm_name);
                    
                    // Structs are implemented as value types, but if this type is a pointer,
                    // then return one.
                    TypeRef p;
                    if (is_pointer) p = LLVM.PointerType(s, 0);
                    else p = s;

                    JITER.previous_llvm_types_created_global.Add(tr, p);

                    // Create array of typerefs as argument to StructSetBody below.
                    // Note, tr is correct type, but tr.Resolve of a generic type turns the type
                    // into an uninstantiated generic type. E.g., List<int> contains a generic T[] containing the
                    // data. T could be a struct/value type, or T could be a class.

                    var new_list = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>(
                        generic_type_rewrite_rules);
                    foreach (var a in additional)
                        new_list.Add(a.Key, a.Value);

                    // This code should use this:  BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))

                    List<TypeRef> list = new List<TypeRef>();
                    int offset = 0;
                    var fields = td.Fields;
                    foreach (var field in fields)
                    {
                        FieldAttributes attr = field.Attributes;
                        if ((attr & FieldAttributes.Static) != 0)
                        {
                           
                            continue;
                        }

                        TypeReference field_type = field.FieldType;
                        TypeReference instantiated_field_type = field.FieldType;

                        if (git != null)
                        {
                            Collection<TypeReference> generic_args = git.GenericArguments;
                            if (field.FieldType.IsArray)
                            {
                                var field_type_as_array_type = field.FieldType as ArrayType;
                                //var et = field.FieldType.GetElementType();
                                var et = field_type_as_array_type.ElementType;
                                var bbc = et.HasGenericParameters;
                                var bbbbc = et.IsGenericParameter;
                                var array = field.FieldType as ArrayType;
                                int rank = array.Rank;
                                if (bbc)
                                {
                                    instantiated_field_type = et.MakeGenericInstanceType(generic_args.ToArray());
                                    instantiated_field_type = instantiated_field_type.MakeArrayType(rank);
                                }
                                else if (bbbbc)
                                {
                                    instantiated_field_type = generic_args.First();
                                    instantiated_field_type = instantiated_field_type.MakeArrayType(rank);
                                }
                            }
                            else
                            {
                                var et = field.FieldType;
                                var bbc = et.HasGenericParameters;
                                var bbbbc = et.IsGenericParameter;
                                if (bbc)
                                {
                                    instantiated_field_type = et.MakeGenericInstanceType(generic_args.ToArray());
                                }
                                else if (bbbbc)
                                {
                                    instantiated_field_type = generic_args.First();
                                }
                            }
                        }

                        int field_size;
                        int alignment;
                        var ft =
                            instantiated_field_type.ToSystemType();
                        var array_or_class = (instantiated_field_type.IsArray || !instantiated_field_type.IsValueType);
                        if (array_or_class)
                        {
                            field_size = BUFFERS.SizeOf(typeof(IntPtr));
                            alignment = BUFFERS.Alignment(typeof(IntPtr));
                            int padding = BUFFERS.Padding(offset, alignment);
                            offset = offset + padding + field_size;
                            if (padding != 0)
                            {
                                // Add in bytes to effect padding.
                                for (int j = 0; j < padding; ++j)
                                    list.Add(LLVM.Int8Type());
                            }
                            var field_converted_type = ToTypeRef(instantiated_field_type, new_list, level + 1);
                            list.Add(field_converted_type);
                        }
                        else
                        {
                            field_size = BUFFERS.SizeOf(ft);
                            alignment = BUFFERS.Alignment(ft);
                            int padding = BUFFERS.Padding(offset, alignment);
                            offset = offset + padding + field_size;
                            if (padding != 0)
                            {
                                // Add in bytes to effect padding.
                                for (int j = 0; j < padding; ++j)
                                    list.Add(LLVM.Int8Type());
                            }
                            var field_converted_type = ToTypeRef(instantiated_field_type, new_list, level + 1);
                            list.Add(field_converted_type);
                        }
                    }
                    LLVM.StructSetBody(s, list.ToArray(), true);
                    return p;
                }
                else if (td != null && td.IsClass)
                {
                    Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> additional = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();
                    var gp = tr.GenericParameters;
                    Mono.Collections.Generic.Collection<TypeReference> ga = null;
                    if (git != null)
                    {
                        ga = git.GenericArguments;
                        Mono.Collections.Generic.Collection<GenericParameter> gg = td.GenericParameters;
                        // Map parameter to instantiated type.
                        for (int i = 0; i < gg.Count; ++i)
                        {
                            GenericParameter pp = gg[i]; // This is the parameter name, like "T".
                            TypeReference qq = ga[i]; // This is the generic parameter, like System.Int32.
                            var rewr = ToTypeRef(qq);
                            TypeReference trrr = pp as TypeReference;
                            var system_type = qq.ToSystemType();
                            Tuple<TypeReference, GenericParameter> tr_gp = new Tuple<TypeReference, GenericParameter>(tr, pp);
                            if (system_type == null) throw new Exception("Failed to convert " + qq);
                            additional[tr_gp] = system_type;
                        }
                    }

                    // Create a struct/class type.
                    //ContextRef c = LLVM.ContextCreate();
                    ContextRef c = LLVM.GetModuleContext(JITER.global_llvm_module);
                    string llvm_name = JITER.RenameToLegalLLVMName(tr.ToString());
                    TypeRef s = LLVM.StructCreateNamed(c, llvm_name);

                    // Classes are always implemented as pointers.
                    TypeRef p;
                    p = LLVM.PointerType(s, 0);

                    JITER.previous_llvm_types_created_global.Add(tr, p);

                    // Create array of typerefs as argument to StructSetBody below.
                    // Note, tr is correct type, but tr.Resolve of a generic type turns the type
                    // into an uninstantiated generic type. E.g., List<int> contains a generic T[] containing the
                    // data. T could be a struct/value type, or T could be a class.

                    var new_list = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>(
                        generic_type_rewrite_rules);
                    foreach (var a in additional)
                        new_list.Add(a.Key, a.Value);

                    List<TypeRef> list = new List<TypeRef>();
                    int offset = 0;
                    var fields = td.Fields;
                    foreach (var field in fields)
                    {
                        FieldAttributes attr = field.Attributes;
                        if ((attr & FieldAttributes.Static) != 0)
                            continue;

                        TypeReference field_type = field.FieldType;
                        TypeReference instantiated_field_type = field.FieldType;

                        if (git != null)
                        {
                            Collection<TypeReference> generic_args = git.GenericArguments;
                            if (field.FieldType.IsArray)
                            {
                                var field_type_as_array_type = field.FieldType as ArrayType;
                                //var et = field.FieldType.GetElementType();
                                var et = field_type_as_array_type.ElementType;
                                var bbc = et.HasGenericParameters;
                                var bbbbc = et.IsGenericParameter;
                                var array = field.FieldType as ArrayType;
                                int rank = array.Rank;
                                if (bbc)
                                {
                                    instantiated_field_type = et.MakeGenericInstanceType(generic_args.ToArray());
                                    instantiated_field_type = instantiated_field_type.MakeArrayType(rank);
                                }
                                else if (bbbbc)
                                {
                                    instantiated_field_type = generic_args.First();
                                    instantiated_field_type = instantiated_field_type.MakeArrayType(rank);
                                }
                            }
                            else
                            {
                                var et = field.FieldType;
                                var bbc = et.HasGenericParameters;
                                var bbbbc = et.IsGenericParameter;
                                if (bbc)
                                {
                                    instantiated_field_type = et.MakeGenericInstanceType(generic_args.ToArray());
                                }
                                else if (bbbbc)
                                {
                                    instantiated_field_type = generic_args.First();
                                }
                            }
                        }


                        int field_size;
                        int alignment;
                        var array_or_class = (instantiated_field_type.IsArray || !instantiated_field_type.IsValueType);
                        if (array_or_class)
                        {
                            field_size = BUFFERS.SizeOf(typeof(IntPtr));
                            alignment = BUFFERS.Alignment(typeof(IntPtr));
                            int padding = BUFFERS.Padding(offset, alignment);
                            offset = offset + padding + field_size;
                            if (padding != 0)
                            {
                                // Add in bytes to effect padding.
                                for (int j = 0; j < padding; ++j)
                                    list.Add(LLVM.Int8Type());
                            }
                            var field_converted_type = ToTypeRef(instantiated_field_type, new_list, level + 1);
                            list.Add(field_converted_type);
                        }
                        else
                        {
                            var ft =
                                instantiated_field_type.ToSystemType();
                            field_size = BUFFERS.SizeOf(ft);
                            alignment = BUFFERS.Alignment(ft);
                            int padding = BUFFERS.Padding(offset, alignment);
                            offset = offset + padding + field_size;
                            if (padding != 0)
                            {
                                // Add in bytes to effect padding.
                                for (int j = 0; j < padding; ++j)
                                    list.Add(LLVM.Int8Type());
                            }
                            var field_converted_type = ToTypeRef(instantiated_field_type, new_list, level + 1);
                            list.Add(field_converted_type);
                        }
                    }
                    LLVM.StructSetBody(s, list.ToArray(), true);
                    return p;
                }
                else
                    throw new Exception("Unknown type.");
            }
            catch (Exception e)
            {
                throw e;
            }
            finally
            {

            }
        }
    }

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
            }
            return weeded;
        }

        private static void InitStateGenerics(STATE<TypeReference> state,
            Dictionary<CFG.Vertex, STATE<TypeReference>> states_in,
            Dictionary<CFG.Vertex, STATE<TypeReference>> states_out,
            CFG.Vertex bb)
        {
            int in_level = -1;
            int args = bb.StackNumberOfArguments;
            bool scalar_ret = bb.HasScalarReturnValue;
            bool struct_ret = bb.HasStructReturnValue;
            bool has_this = bb.HasThis;
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
                    if (pred._original_method_reference != bb._original_method_reference)
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
                    state._stack.Push(bb._method_definition.DeclaringType);

                for (int i = 0; i < bb._method_definition.Parameters.Count; ++i)
                {
                    var par = bb._method_definition.Parameters[i];
                    var type = par.ParameterType;
                    if (Campy.Utils.Options.IsOn("jit_trace"))
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
                Collection<VariableDefinition> variables = bb.RewrittenCalleeSignature.Resolve().Body.Variables;
                state._locals = state._stack.Section((int) state._stack.Count, locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i].VariableType;
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

                var other = states_out[p_llvm_node];
                state._struct_ret = state._stack.Section(other._struct_ret.Base, other._struct_ret.Len);
                state._this = state._stack.Section(other._this.Base, other._this.Len);
                state._arguments = state._stack.Section(other._arguments.Base, other._arguments.Len);
                state._locals = state._stack.Section(other._locals.Base, other._locals.Len);
            }
        }

        public static void InstantiateGenerics(this List<CFG.Vertex> basic_blocks_to_compile)
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
            var ordered_list = new TarjanNoBackEdges<CFG.Vertex, CFG.Edge>(_mcfg).ToList();
            ordered_list.Reverse();

            // Eliminate all node names not in the work list.
            var order = ordered_list.Where(v => work_names.Contains(v.Name)).ToList();

            Dictionary<CFG.Vertex, bool> visited = new Dictionary<CFG.Vertex, bool>();
            Dictionary<CFG.Vertex, STATE<TypeReference>> states_in = new Dictionary<CFG.Vertex, STATE<TypeReference>>();
            Dictionary<CFG.Vertex, STATE<TypeReference>> states_out = new Dictionary<CFG.Vertex, STATE<TypeReference>>();

            // propagate type information and create new basic blocks for nodes that have
            // specific generic type information.
            foreach (var bb in order)
            {
                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    System.Console.WriteLine("Generic computations for node " + bb.Name);

                // Create new stack state with predecessor information, basic block/function
                // information.
                var state_in = new STATE<TypeReference>(visited, states_in, states_out, bb, InitStateGenerics);

                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                {
                    System.Console.WriteLine("state in");
                    state_in.OutputTrace(new String(' ', 4));
                }

                var state_out = new STATE<TypeReference>(state_in);
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
                    last_inst = inst;
                    inst = inst.GenerateGenerics(state_out);
                    // Rewrite instruction.
                    if (inst != last_inst)
                        bb.Instructions[i] = inst;
                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                        state_out.OutputTrace(new String(' ', 4));
                }
                visited[bb] = true;
            }
        }

        public static List<CFG.Vertex> ThreadInstructions(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            foreach (CFG.Vertex bb in basic_blocks_to_compile)
            {
                INST prev = null;
                foreach (var j in bb.Instructions)
                {
                    j.Block = bb;
                    if (prev != null) prev.Next = j;
                    prev = j;
                }
            }

            return basic_blocks_to_compile;
        }

        // Method to denormalize information about the method this block is associated with,
        // placing that information into the block for easier access.
        public static List<CFG.Vertex> ComputeBasicMethodProperties(this List<CFG.Vertex> basic_blocks_to_compile)
        {
            foreach (CFG.Vertex bb in basic_blocks_to_compile)
            {
                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("Compile part 1, node " + bb);

                Mono.Cecil.MethodReturnType rt = bb._method_definition.MethodReturnType;
                Mono.Cecil.TypeReference tr = rt.ReturnType;
                var ret = tr.FullName != "System.Void";
                bb.HasScalarReturnValue = ret && !tr.IsStruct();
                bb.HasStructReturnValue = ret && tr.IsStruct();
                bb.RewrittenCalleeSignature = bb._method_definition;
                bb.HasThis = bb._method_definition.HasThis;
                bb.StackNumberOfArguments = bb._method_definition.Parameters.Count
                                            + (bb.HasThis ? 1 : 0)
                                            + (bb.HasStructReturnValue ? 1 : 0);
                Mono.Cecil.MethodReference mr = bb.RewrittenCalleeSignature;
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

                if (!JITER.Singleton.IsFullyInstantiatedNode(bb))
                {
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine("skipping -- not fully instantiated block the contains generics.");
                    continue;
                }

                MethodReference method = bb._original_method_reference;
                List<ParameterDefinition> parameters = method.Parameters.ToList();
                List<ParameterReference> instantiated_parameters = new List<ParameterReference>();

                ModuleRef mod = JITER.global_llvm_module; // LLVM.ModuleCreateWithName(mn);
                bb.LlvmInfo.Module = mod;

                uint count = (uint)bb.StackNumberOfArguments;
                TypeRef[] param_types = new TypeRef[count];
                int current = 0;
                if (count > 0)
                {
                    if (bb.HasStructReturnValue)
                    {
                        TYPE t = new TYPE(method.ReturnType);
                        param_types[current++] = LLVM.PointerType(t.IntermediateType, 0);
                    }
                    if (bb.HasThis)
                    {
                        TYPE t = new TYPE(method.DeclaringType);
                        if (method.DeclaringType.IsValueType)
                        {
                            // Parameter "this" is a struct, but code in body of method assumes
                            // a pointer is passed. Make the parameter a pointer.
                            param_types[current++] = LLVM.PointerType(t.IntermediateType, 0);
                        }
                        else
                        {
                            param_types[current++] = t.IntermediateType;
                        }
                    }

                    foreach (var p in parameters)
                    {
                        TypeReference type_reference_of_parameter = p.ParameterType;

                        if (method.DeclaringType.IsGenericInstance && method.ContainsGenericParameter)
                        {
                            var git = method.DeclaringType as GenericInstanceType;
                            type_reference_of_parameter = JITER.FromGenericParameterToTypeReference(
                                type_reference_of_parameter, git);
                        }
                        TYPE t = new TYPE(type_reference_of_parameter);
                        param_types[current++] = t.IntermediateType;
                    }

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                    {
                        System.Console.WriteLine("Params for block " + bb.Name + " " + bb._original_method_reference.FullName);
                        System.Console.WriteLine("(" + bb._method_definition.FullName + ")");
                        foreach (var pp in param_types)
                        {
                            string a = LLVM.PrintTypeToString(pp);
                            System.Console.WriteLine(" " + a);
                        }
                    }
                }

                //mi2 = FromGenericParameterToTypeReference(typeof(void).ToMonoTypeReference(), null);
                TYPE t_ret = new TYPE(JITER.FromGenericParameterToTypeReference(method.ReturnType, method.DeclaringType as GenericInstanceType));
                if (bb.HasStructReturnValue)
                {
                    t_ret = new TYPE(typeof(void).ToMonoTypeReference());
                }
                TypeRef ret_type = t_ret.IntermediateType;
                TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                ValueRef fun = LLVM.AddFunction(mod,
                    JITER.RenameToLegalLLVMName(JITER.MethodName(method)), met_type);
                BasicBlockRef entry = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                bb.LlvmInfo.BasicBlock = entry;
                bb.LlvmInfo.MethodValueRef = fun;
                var t_fun = LLVM.TypeOf(fun);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(JITER.global_llvm_module);
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
            var _mcfg = basic_blocks_to_compile.First()._graph;

            foreach (var bb in basic_blocks_to_compile)
            {
                if (!JITER.Singleton.IsFullyInstantiatedNode(bb))
                    continue;

                IEnumerable<CFG.Vertex> successors = _mcfg.SuccessorNodes(bb);
                if (!bb.IsEntry)
                {
                    var ent = bb.Entry;
                    var lvv_ent = ent;
                    var fun = lvv_ent.LlvmInfo.MethodValueRef;
                    var t_fun = LLVM.TypeOf(fun);
                    var t_fun_con = LLVM.GetTypeContext(t_fun);
                    var context = LLVM.GetModuleContext(JITER.global_llvm_module);
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

        private static void InitState(STATE<VALUE> state,
            Dictionary<CFG.Vertex, STATE<VALUE>> states_in,
            Dictionary<CFG.Vertex, STATE<VALUE>> states_out,
            CFG.Vertex bb)
        {
            int in_level = -1;
            int args = bb.StackNumberOfArguments;
            bool scalar_ret = bb.HasScalarReturnValue;
            bool struct_ret = bb.HasStructReturnValue;
            bool has_this = bb.HasThis;
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
                    if (pred._original_method_reference != bb._original_method_reference)
                        throw new Exception("Interprocedural edge should not exist.");
                    // If predecessor has not been visited, warn and do not consider.
                    // Warn if predecessor does not concur with another predecessor.
                    if (in_level != -1 && states_out[pred]._stack.Count != in_level)
                        throw new Exception("Miscalculation in stack size "
                                            + "for basic block " + bb
                                            + " or predecessor " + pred);
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
                var context = LLVM.GetModuleContext(JITER.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");

                for (uint i = 0; i < args; ++i)
                {
                    var par = new VALUE(LLVM.GetParam(fun, i));
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine(par);
                    state._stack.Push(par);
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
                Collection<VariableDefinition> variables = bb.RewrittenCalleeSignature.Resolve().Body.Variables;
                state._locals = state._stack.Section((int)state._stack.Count, locals);
                for (int i = 0; i < locals; ++i)
                {
                    var tr = variables[i].VariableType;
                    TYPE type = new TYPE(tr);
                    VALUE value;
                    if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.PointerTypeKind)
                        value = new VALUE(LLVM.ConstPointerNull(type.IntermediateType));
                    else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.DoubleTypeKind)
                        value = new VALUE(LLVM.ConstReal(LLVM.DoubleType(), 0));
                    else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.IntegerTypeKind)
                        value = new VALUE(LLVM.ConstInt(type.IntermediateType, (ulong)0, true));
                    else if (LLVM.GetTypeKind(type.IntermediateType) == TypeKind.StructTypeKind)
                    {
                        var entry = bb.Entry.LlvmInfo.BasicBlock;
                        //var beginning = LLVM.GetFirstInstruction(entry);
                        //LLVM.PositionBuilderBefore(basic_block.Builder, beginning);
                        var new_obj = LLVM.BuildAlloca(bb.LlvmInfo.Builder, type.IntermediateType, "i" + INST.instruction_id++); // Allocates struct on stack, but returns a pointer to struct.
                        //LLVM.PositionBuilderAtEnd(bb.LlvmInfo.Builder, bb.BasicBlock);
                        value = new VALUE(new_obj);
                    }
                    else
                        throw new Exception("Unhandled type");
                    state._stack.Push(value);
                }

                // Set up any thing else.
                for (int i = state._stack.Size(); i < level; ++i)
                {
                    VALUE value = new VALUE(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
                    state._stack.Push(value);
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
                        VALUE f = new VALUE(LLVM.ConstInt(LLVM.Int32Type(), (ulong)0, true));
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
                var other = states_out[p_llvm_node];
                state._struct_ret = state._stack.Section(other._struct_ret.Base, other._struct_ret.Len);
                state._this = state._stack.Section(other._this.Base, other._this.Len);
                state._arguments = state._stack.Section(other._arguments.Base, other._arguments.Len);
                state._locals = state._stack.Section(other._locals.Base, other._locals.Len);
            }
        }

        public static void TranslateToLLVMInstructions(this List<CFG.Vertex> basic_blocks_to_compile)
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
            var ordered_list = new TarjanNoBackEdges<CFG.Vertex, CFG.Edge>(_mcfg).ToList();
            ordered_list.Reverse();

            // Eliminate all node names not in the work list.
            var order = ordered_list.Where(v => work_names.Contains(v.Name)).ToList();

            Dictionary<CFG.Vertex, bool> visited = new Dictionary<CFG.Vertex, bool>();
            Dictionary<CFG.Vertex, STATE<VALUE>> states_in = new Dictionary<CFG.Vertex, STATE<VALUE>>();
            Dictionary<CFG.Vertex, STATE<VALUE>> states_out = new Dictionary<CFG.Vertex, STATE<VALUE>>();

            // Emit LLVM IR code, based on state and per-instruction simulation on that state.
            foreach (var ob in order)
            {
                CFG.Vertex bb = ob;

                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    System.Console.WriteLine("State computations for node " + bb.Name);

                // Create new stack state with predecessor information, basic block/function
                // information.
                var state_in = new STATE<VALUE>(visited, states_in, states_out, bb, InitState);

                if (Campy.Utils.Options.IsOn("state_computation_trace"))
                {
                    System.Console.WriteLine("state in");
                    state_in.OutputTrace(new String(' ', 4));
                }

                var state_out = new STATE<VALUE>(state_in);
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
                    last_inst = inst;
                    inst.DebuggerInfo();
                    inst = inst.Convert(state_out);
                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                        state_out.OutputTrace(new String(' ', 4));
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
                    if(!llvm_node.LlvmInfo.Phi.Contains(res))
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
    }


    public class JITER
    {
        private IMPORTER _importer;
        private CFG _mcfg;
        private static int _nn_id = 0;
        public static ModuleRef global_llvm_module;
        private List<ModuleRef> all_llvm_modules;
        public static Dictionary<string, ValueRef> functions_in_internal_bcl_layer;
        Dictionary<Tuple<CFG.Vertex, Mono.Cecil.TypeReference, System.Type>, CFG.Vertex> mmap;
        internal static Dictionary<TypeReference, TypeRef> basic_llvm_types_created;
        internal static Dictionary<TypeReference, TypeRef> previous_llvm_types_created_global;
        internal static Dictionary<string, string> _rename_to_legal_llvm_name_cache;
        public int _start_index;
        private static bool init;
        private Dictionary<MethodInfo, IntPtr> method_to_image;
        private bool done_init;
        private static JITER _singleton;

        public static JITER Singleton
        {
            get
            {
                if (_singleton == null) _singleton = new JITER();
                return _singleton;
            }
        }

        private JITER()
        {
            global_llvm_module = default(ModuleRef);
            all_llvm_modules = new List<ModuleRef>();
            functions_in_internal_bcl_layer = new Dictionary<string, ValueRef>();
            mmap = new Dictionary<Tuple<CFG.Vertex, TypeReference, System.Type>, CFG.Vertex>(new Comparer());
            basic_llvm_types_created = new Dictionary<TypeReference, TypeRef>();
            previous_llvm_types_created_global = new Dictionary<TypeReference, TypeRef>();
            _rename_to_legal_llvm_name_cache = new Dictionary<string, string>();
            method_to_image = new Dictionary<MethodInfo, IntPtr>();
            done_init = false;

            _importer = IMPORTER.Singleton();
            _mcfg = _importer.Cfg;
            global_llvm_module = CreateModule("global");
            LLVM.EnablePrettyStackTrace();
            var triple = LLVM.GetDefaultTargetTriple();
            LLVM.SetTarget(global_llvm_module, triple);
            LLVM.InitializeAllTargets();
            LLVM.InitializeAllTargetMCs();
            LLVM.InitializeAllTargetInfos();
            LLVM.InitializeAllAsmPrinters();

            basic_llvm_types_created.Add(
                typeof(Int16).ToMonoTypeReference(),
                LLVM.Int16Type());

            basic_llvm_types_created.Add(
                typeof(UInt16).ToMonoTypeReference(),
                LLVM.Int16Type());

            basic_llvm_types_created.Add(
                typeof(Int32).ToMonoTypeReference(),
                LLVM.Int32Type());

            basic_llvm_types_created.Add(
                typeof(UInt32).ToMonoTypeReference(),
                LLVM.Int32Type());

            basic_llvm_types_created.Add(
                typeof(Int64).ToMonoTypeReference(),
                LLVM.Int64Type());

            basic_llvm_types_created.Add(
                typeof(UInt64).ToMonoTypeReference(),
                LLVM.Int64Type());

            basic_llvm_types_created.Add(
                typeof(float).ToMonoTypeReference(),
                LLVM.FloatType());

            basic_llvm_types_created.Add(
                typeof(double).ToMonoTypeReference(),
                LLVM.DoubleType());


            basic_llvm_types_created.Add(
                typeof(bool).ToMonoTypeReference(),
                LLVM.Int32Type()); // Asking for trouble if one tries to map directly to 1 bit.

            basic_llvm_types_created.Add(
                typeof(char).ToMonoTypeReference(),
                LLVM.Int8Type());

            basic_llvm_types_created.Add(
                typeof(void).ToMonoTypeReference(),
                LLVM.VoidType());

            basic_llvm_types_created.Add(
                typeof(Mono.Cecil.TypeDefinition).ToMonoTypeReference(),
                LLVM.PointerType(LLVM.VoidType(), 0));

            basic_llvm_types_created.Add(
                typeof(System.Type).ToMonoTypeReference(),
                LLVM.PointerType(LLVM.VoidType(), 0));

            //basic_llvm_types_created.Add(
            //    typeof(string).ToMonoTypeReference(),
            //    LLVM.PointerType(LLVM.VoidType(), 0));



            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.tid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.tid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.tid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.ctaid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.ctaid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.ctaid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.ntid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.ntid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.ntid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.nctaid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.nctaid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            functions_in_internal_bcl_layer.Add("llvm.nvvm.read.ptx.sreg.nctaid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            RUNTIME.Initialize();
        }

        public static void InitCuda()
        {
            // Initialize CUDA if it hasn't been done before.
            if (init) return;
            Utils.CudaHelpers.CheckCudaError(Cuda.cuInit(0));
            Utils.CudaHelpers.CheckCudaError(Cuda.cuDevicePrimaryCtxReset(0));
            Utils.CudaHelpers.CheckCudaError(Cuda.cuCtxCreate_v2(out CUcontext pctx, 0, 0));
            init = true;
        }

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

        private CFG.Vertex FindInstantiatedBasicBlock(CFG.Vertex current, Mono.Cecil.TypeReference generic_type, System.Type value)
        {
            var k = new Tuple<CFG.Vertex, TypeReference, System.Type>(current, generic_type, value);

            // Find vertex that maps from base vertex via symbol.
            if (!mmap.ContainsKey(k))
                return null;

            var v = mmap[k];
            return v;
        }

        private void EnterInstantiatedBasicBlock(CFG.Vertex current, Mono.Cecil.TypeReference generic_type, System.Type value, CFG.Vertex bb)
        {
            var k = new Tuple<CFG.Vertex, TypeReference, System.Type>(current, generic_type, value);
            mmap[k] = bb;
        }

        public CFG.Vertex Eval(CFG.Vertex current, Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> ops)
        {
            // Start at current vertex, and find transition state given ops.
            CFG.Vertex result = current;
            for (;;)
            {
                bool found = false;
                foreach(var t in ops)
                {
                    Tuple<TypeReference, GenericParameter> k = t.Key;
                    var x = FindInstantiatedBasicBlock(current, k.Item1, t.Value);
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

        private bool TypeUsingGeneric() { return false; }

        public bool IsFullyInstantiatedNode(CFG.Vertex node)
        {
            bool result = false;
            // First, go through and mark all nodes that have non-null
            // previous entries.

            Dictionary<CFG.Vertex, bool> instantiated = new Dictionary<CFG.Vertex, bool>();
            foreach (var v in _mcfg.Vertices)
            {
                instantiated[v] = true;
            }
            foreach (var v in _mcfg.Vertices)
            {
                if (v.PreviousVertex != null) instantiated[v.PreviousVertex] = false;
            }
            result = instantiated[node];
            return result;
        }

        private ModuleRef CreateModule(string name)
        {
            var new_module = LLVM.ModuleCreateWithName(name);
            var context = LLVM.GetModuleContext(new_module);
            all_llvm_modules.Add(new_module);
            return new_module;
        }

        public static TypeReference FromGenericParameterToTypeReference(TypeReference type_reference_of_parameter, GenericInstanceType git)
        {
            if (git == null)
                return type_reference_of_parameter;
            Collection<TypeReference> genericArguments = git.GenericArguments;
            TypeDefinition td = git.Resolve();

            // Map parameter to actual type.

            var t1 = type_reference_of_parameter.HasGenericParameters;
            var t2 = type_reference_of_parameter.IsGenericInstance;
            var t3 = type_reference_of_parameter.ContainsGenericParameter;
            var t4 = type_reference_of_parameter.IsGenericParameter;


            if (type_reference_of_parameter.IsGenericParameter)
            {
                var gp = type_reference_of_parameter as GenericParameter;
                var num = gp.Position;
                var yo = genericArguments.ToArray()[num];
                type_reference_of_parameter = yo;
            }
            else if (type_reference_of_parameter.ContainsGenericParameter && type_reference_of_parameter.IsArray)
            {
                var array_type = type_reference_of_parameter as ArrayType;
                var element_type = array_type.ElementType;
                element_type = FromGenericParameterToTypeReference(element_type, git);
                ArrayType art = element_type.MakeArrayType();
                type_reference_of_parameter = art;
            }
            return type_reference_of_parameter;
        }

        public static string MethodName(MethodReference mr)
        {
            return mr.FullName;
        }


        public string CompileToLLVM(List<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeReference> list_of_data_types_used,
            string basic_block_id)
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

            {
                var module = JITER.global_llvm_module;
                var basic_block = GetBasicBlock(basic_block_id);

                if (Campy.Utils.Options.IsOn("module_trace"))
                    LLVM.DumpModule(module);

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
                ContextRef context_ref = LLVM.GetModuleContext(JITER.global_llvm_module);
                ValueRef kernelMd = LLVM.MDNodeInContext(
                    context_ref, new ValueRef[3]
                {
                    basic_block.LlvmInfo.MethodValueRef,
                    LLVM.MDStringInContext(context_ref, "kernel", 6),
                    LLVM.ConstInt(LLVM.Int32TypeInContext(context_ref), 1, false)
                });
                LLVM.AddNamedMetadataOperand(module, "nvvm.annotations", kernelMd);
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
                        ptx = ptx.Replace(".target sm_20", ".target sm_30");

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

        public CFG.Vertex GetBasicBlock(string block_id)
        {
            return _mcfg.Vertices.Where(i => i.IsEntry && i.Name == block_id).FirstOrDefault();
        }

        public IntPtr JitCodeToImage(MethodInfo kernel_method, object kernel_target)
        {
            if (method_to_image.TryGetValue(kernel_method, out IntPtr value))
            {
                return value;
            }

            Stopwatch stopwatch_discovery = new Stopwatch();
            stopwatch_discovery.Reset();
            stopwatch_discovery.Start();

            // Parse kernel instructions to determine basic block representation of all the code to compile.
            int change_set_id = _mcfg.StartChangeSet();
            _importer.AnalyzeMethod(kernel_method);
            if (_importer.Failed)
            {
                throw new Exception("Failure to find all methods in GPU code. Cannot continue.");
            }
            List<CFG.Vertex> cs = _mcfg.PopChangeSet(change_set_id);

            stopwatch_discovery.Stop();
            var elapse_discovery = stopwatch_discovery.Elapsed;
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("discovery     " + elapse_discovery);

            object target = kernel_target;

            // Get basic block of entry.
            CFG.Vertex bb;
            if (!cs.Any())
            {
                // Compiled previously. Look for basic block of entry.
                bb = _mcfg.Entries.Where(v =>
                    v.IsEntry && v._original_method_reference.Name == kernel_method.Name).FirstOrDefault();
            }
            else
            {
                bb = cs.First();
            }

            // Very important note: Although we have the control flow graph of the code that is to
            // be compiled, there is going to be generics used, e.g., ArrayView<int>, within the body
            // of the code and in the called runtime library. We need to record the types for compiling
            // and add that to compilation.
            // https://stackoverflow.com/questions/5342345/how-do-generics-get-compiled-by-the-jit-compiler

            // Create a list of generics called with types passed.
            List<System.Type> list_of_data_types_used = new List<System.Type>();
            list_of_data_types_used.Add(kernel_target.GetType());

            // Convert list into Mono data types.
            List<Mono.Cecil.TypeReference> list_of_mono_data_types_used = new List<TypeReference>();
            foreach (System.Type data_type_used in list_of_data_types_used)
            {
                list_of_mono_data_types_used.Add(
                    data_type_used.ToMonoTypeReference());
            }

            // Associate "this" with entry.
            Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> ops = bb.OpsFromOriginal;

            var stopwatch_compiler = new Stopwatch();
            stopwatch_compiler.Reset();
            stopwatch_compiler.Start();

            // Compile methods with added type information.
            string ptx = CompileToLLVM(cs, list_of_mono_data_types_used,
                bb.Name);

            stopwatch_compiler.Stop();
            var elapse_compiler = stopwatch_compiler.Elapsed;
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("compiler      " + elapse_compiler);
            var current_directory = Directory.GetCurrentDirectory();
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("Current directory " + current_directory);

            CudaHelpers.CheckCudaError(Cuda.cuMemGetInfo_v2(out ulong free_memory, out ulong total_memory));
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("total memory " + total_memory + " free memory " + free_memory);
            CudaHelpers.CheckCudaError(Cuda.cuCtxGetLimit(out ulong pvalue, CUlimit.CU_LIMIT_STACK_SIZE));
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine("Stack size " + pvalue);
            var stopwatch_cuda_compile = new Stopwatch();
            stopwatch_cuda_compile.Reset();
            stopwatch_cuda_compile.Start();

            // GetCudaFunction(string basic_block_id, string ptx)
            var basic_block_id = bb.Name;
            var basic_block = GetBasicBlock(basic_block_id);
            var method = basic_block._original_method_reference;

            var res = Cuda.cuDeviceGet(out int device, 0);
            Utils.CudaHelpers.CheckCudaError(res);
            res = Cuda.cuDeviceGetPCIBusId(out string pciBusId, 100, device);
            Utils.CudaHelpers.CheckCudaError(res);
            res = Cuda.cuDeviceGetName(out string name, 100, device);
            Utils.CudaHelpers.CheckCudaError(res);

            res = Cuda.cuCtxGetCurrent(out CUcontext pctx);
            Utils.CudaHelpers.CheckCudaError(res);

            res = Cuda.cuCtxGetApiVersion(pctx, out uint version);
            Utils.CudaHelpers.CheckCudaError(res);

            // Create context done around cuInit() call.
            //res = Cuda.cuCtxCreate_v2(out CUcontext cuContext, 0, device);
            //CheckCudaError(res);

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
            res = Cuda.cuLinkCreate_v2(num_ops_link, op_link, op_values_link_intptr, out CUlinkState linkState);
            Utils.CudaHelpers.CheckCudaError(res);


            IntPtr ptr = Marshal.StringToHGlobalAnsi(ptx);
            CUjit_option[] op = new CUjit_option[0];
            ulong[] op_values = new ulong[0];
            var op_values_handle = GCHandle.Alloc(op_values, GCHandleType.Pinned);
            var op_values_intptr = op_values_handle.AddrOfPinnedObject();
            res = Cuda.cuLinkAddData_v2(linkState, CUjitInputType.CU_JIT_INPUT_PTX, ptr, (uint) ptx.Length, "", 0, op,
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
            var dir = Path.GetDirectoryName(Path.GetFullPath(System.Reflection.Assembly.GetEntryAssembly().Location));
            uint num_ops = 0;
            res = Cuda.cuLinkAddFile_v2(linkState, CUjitInputType.CU_JIT_INPUT_LIBRARY,
                RUNTIME.FindNativeCoreLib(), num_ops, op, op_values_intptr);
            // static .lib
            // cuLinkAddFile_v2, CU_JIT_INPUT_OBJECT; succeeds cuLinkAddFile_v2, fails cuLinkComplete => truncated .lib?
            // cuLinkAddFile_v2, CU_JIT_INPUT_LIBRARY; fails cuLinkAddFile_v2 => invalid image?
            // 
            // static .lib, device-link = no.
            // cuLinkAddFile_v2, CU_JIT_INPUT_LIBRARY; succeeds.

            if (res != CUresult.CUDA_SUCCESS)
            {
                string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                System.Console.WriteLine(info);
                string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                System.Console.WriteLine(error);
            }
            Utils.CudaHelpers.CheckCudaError(res);

            IntPtr image;
            res = Cuda.cuLinkComplete(linkState, out image, out ulong sz);
            if (res != CUresult.CUDA_SUCCESS)
            {
                string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                System.Console.WriteLine(info);
                string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                System.Console.WriteLine(error);
            }
            Utils.CudaHelpers.CheckCudaError(res);

            method_to_image[kernel_method] = image;

            return image;
        }

        public CFG.Vertex GetBasicBlock(MethodInfo kernel_method)
        {
            CFG.Vertex bb = _mcfg.Entries.Where(v =>
                v.IsEntry && v._original_method_reference.Name == kernel_method.Name).FirstOrDefault();
            return bb;
        }

        public CUfunction GetCudaFunction(MethodInfo kernel_method, IntPtr image)
        {
            // Compiled previously. Look for basic block of entry.
            CFG.Vertex bb = _mcfg.Entries.Where(v =>
                v.IsEntry && v._original_method_reference.Name == kernel_method.Name).FirstOrDefault();
            string basic_block_id = bb.Name;
            CUmodule module = RUNTIME.InitializeModule(image);
            RUNTIME.RuntimeModule = module;
            InitBCL(module);
            var normalized_method_name = JITER.RenameToLegalLLVMName(JITER.MethodName(bb._original_method_reference));
            var res = Cuda.cuModuleGetFunction(out CUfunction helloWorld, module, normalized_method_name);
            Utils.CudaHelpers.CheckCudaError(res);
            return helloWorld;
        }

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitTheBcl")]
        public static extern void InitTheBcl(System.IntPtr a1, long a2, int a3, System.IntPtr a4);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitFileSystem")]
        public static extern void InitFileSystem();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "GfsAddFile")]
        public static extern void GfsAddFile(System.IntPtr name, System.IntPtr file, long length, System.IntPtr result);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitializeBCL1")]
        public static extern void InitializeBCL1();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitializeBCL2")]
        public static extern void InitializeBCL2();

    
        private HashSet<string> _added_module_already = new HashSet<string>();

        public void AddAssemblyToFileSystem(Mono.Cecil.ModuleDefinition module)
        {
            string full_path_assem = module.FileName;
            if (_added_module_already.Contains(full_path_assem))
                return;
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
            IntPtr pointer1 = buffers.New(assem.Length + 1);
            BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
            var pointer4 = buffers.New(sizeof(int));
            GfsAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);
        }

        public void InitBCL(CUmodule mod)
        {
            if (!done_init)
            {
                unsafe
                {
                    BUFFERS buffers = new BUFFERS();
                    int the_size = 536870912;
                    IntPtr b = buffers.New(the_size);
                    RUNTIME.BclPtr = b;
                    RUNTIME.BclPtrSize = (ulong)the_size;
                    int max_threads = 16;
                    IntPtr b2 = buffers.New(sizeof(int*));

                    InitTheBcl(b, the_size, max_threads, b2);
                }

                unsafe
                {
                    InitFileSystem();

                    {
                        // Set up corlib.dll in file system.
                        string full_path_assem = RUNTIME.FindCoreLib();
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
                        IntPtr pointer1 = buffers.New(assem.Length + 1);
                        BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
                        var pointer4 = buffers.New(sizeof(int));
                        GfsAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);
                    }

                    InitializeBCL1();

                    InitializeBCL2();

                    done_init = true;
                }
            }

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
                    IntPtr parm1;
                    var bcl_ptr = RUNTIME.BclPtr;
                    IntPtr[] x1 = new IntPtr[] { bcl_ptr };
                    GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                    parm1 = handle1.AddrOfPinnedObject();

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
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    Utils.CudaHelpers.CheckCudaError(res);
                }

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

        /// <summary>
        /// LLVM has a restriction in the names of methods and types different that the Name field of 
        /// the type. For the moment, we rename to a simple identifier following the usual naming
        /// convesions for variables (simple prefix, followed by underscore, then a whole number).
        /// In addition, cache the name so we can rename consistently.
        /// </summary>
        /// <param name="before"></param>
        /// <returns></returns>
        public static string RenameToLegalLLVMName(string before)
        {
            if (_rename_to_legal_llvm_name_cache.ContainsKey(before))
                return _rename_to_legal_llvm_name_cache[before];
            _rename_to_legal_llvm_name_cache[before] = "nn_" + _nn_id++;
            return _rename_to_legal_llvm_name_cache[before];
        }

        public void NameTableTrace()
        {
            System.Console.WriteLine("Name mapping table.");
            foreach (var tuple in _rename_to_legal_llvm_name_cache)
            {
                System.Console.WriteLine(tuple.Key);
                System.Console.WriteLine(tuple.Value);
                System.Console.WriteLine();
            }
        }
    }
}
