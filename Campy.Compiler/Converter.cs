using System;
using System.Collections.Generic;
using System.Diagnostics;
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

namespace Campy.Compiler
{
    public static class ConverterHelper
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
            foreach (var kv in Converter.basic_llvm_types_created)
            {
                if (kv.Key.FullName == tr.FullName)
                {
                    return kv.Value;
                }
            }
            foreach (var kv in Converter.previous_llvm_types_created_global)
            {
                if (kv.Key.FullName == tr.FullName)
                    return kv.Value;
            }
            foreach (var kv in Converter.previous_llvm_types_created_global)
            {
                if (kv.Key.FullName == tr.FullName)
                    return kv.Value;
            }

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

                var is_pointer = tr.IsPointer;
                var is_array = tr.IsArray;
                var is_value_type = tr.IsValueType;

                if (is_pointer)
                {
                    
                }

                TypeDefinition td = tr.Resolve();
                
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
                    Converter.previous_llvm_types_created_global.Add(original_tr, p);
                    return p;
                }
                else if (tr.IsArray)
                {
                    // Note: mono_type_reference.GetElementType() is COMPLETELY WRONG! It does not function the same
                    // as system_type.GetElementType(). Use ArrayType.ElementType!
                    var array_type = tr as ArrayType;
                    var element_type = array_type.ElementType;
                    // ContextRef c = LLVM.ContextCreate();
                    ContextRef c = LLVM.GetModuleContext(Converter.global_llvm_module);
                    string type_name = Converter.RenameToLegalLLVMName(tr.ToString());
                    TypeRef s = LLVM.StructCreateNamed(c, type_name);
                    TypeRef p = LLVM.PointerType(s, 0);
                    Converter.previous_llvm_types_created_global.Add(tr, p);
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
                            Converter.previous_llvm_types_created_global.Add(tr, e);
                            return e;
                        }
                    }
                    throw new Exception("Cannot convert " + tr.Name);
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
                    ContextRef c = LLVM.GetModuleContext(Converter.global_llvm_module);
                    string llvm_name = Converter.RenameToLegalLLVMName(tr.ToString());

                    TypeRef s = LLVM.StructCreateNamed(c, llvm_name);
                    
                    // Structs are implemented as value types, but if this type is a pointer,
                    // then return one.
                    TypeRef p;
                    if (is_pointer) p = LLVM.PointerType(s, 0);
                    else p = s;

                    Converter.previous_llvm_types_created_global.Add(tr, p);

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
                        var ft =
                            instantiated_field_type.ToSystemType();
                        var array_or_class = (instantiated_field_type.IsArray || !instantiated_field_type.IsValueType);
                        if (array_or_class)
                        {
                            field_size = Buffers.SizeOf(typeof(IntPtr));
                            alignment = Buffers.Alignment(typeof(IntPtr));
                            int padding = Buffers.Padding(offset, alignment);
                            offset = offset + padding + field_size;
                            if (padding != 0)
                            {
                                // Add in bytes to effect padding.
                                for (int j = 0; j < padding; ++j)
                                    list.Add(LLVM.Int8Type());
                            }
                            var field_converted_type = ToTypeRef(instantiated_field_type, new_list, level + 1);
                            field_converted_type = field_converted_type;
                            list.Add(field_converted_type);
                        }
                        else
                        {
                            field_size = Buffers.SizeOf(ft);
                            alignment = Buffers.Alignment(ft);
                            int padding = Buffers.Padding(offset, alignment);
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
                            GenericParameter pp = gg[i];
                            TypeReference qq = ga[i];
                            TypeReference trrr = pp as TypeReference;
                            var system_type = qq.ToSystemType();
                            Tuple<TypeReference, GenericParameter> tr_gp = new Tuple<TypeReference, GenericParameter>(tr, pp);
                            if (system_type == null) throw new Exception("Failed to convert " + qq);
                            additional[tr_gp] = system_type;
                        }
                    }

                    // Create a struct/class type.
                    //ContextRef c = LLVM.ContextCreate();
                    ContextRef c = LLVM.GetModuleContext(Converter.global_llvm_module);
                    string llvm_name = Converter.RenameToLegalLLVMName(tr.ToString());
                    TypeRef s = LLVM.StructCreateNamed(c, llvm_name);

                    // Classes are always implemented as pointers.
                    TypeRef p;
                    p = LLVM.PointerType(s, 0);

                    Converter.previous_llvm_types_created_global.Add(tr, p);

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
                        var ft =
                            instantiated_field_type.ToSystemType();
                        var array_or_class = (instantiated_field_type.IsArray || !instantiated_field_type.IsValueType);
                        if (array_or_class)
                        {
                            field_size = Buffers.SizeOf(typeof(IntPtr));
                            alignment = Buffers.Alignment(typeof(IntPtr));
                            int padding = Buffers.Padding(offset, alignment);
                            offset = offset + padding + field_size;
                            if (padding != 0)
                            {
                                // Add in bytes to effect padding.
                                for (int j = 0; j < padding; ++j)
                                    list.Add(LLVM.Int8Type());
                            }
                            var field_converted_type = ToTypeRef(instantiated_field_type, new_list, level + 1);
                            field_converted_type = field_converted_type;
                            list.Add(field_converted_type);
                        }
                        else
                        {
                            field_size = Buffers.SizeOf(ft);
                            alignment = Buffers.Alignment(ft);
                            int padding = Buffers.Padding(offset, alignment);
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

    public class Converter
    {
        private CFG _mcfg;
        private static int _nn_id = 0;
        public static ModuleRef global_llvm_module = default(ModuleRef);
        private List<ModuleRef> all_llvm_modules = new List<ModuleRef>();
        public static Dictionary<string, ValueRef> built_in_functions = new Dictionary<string, ValueRef>();
        Dictionary<Tuple<CFG.Vertex, Mono.Cecil.TypeReference, System.Type>, CFG.Vertex> mmap
            = new Dictionary<Tuple<CFG.Vertex, TypeReference, System.Type>, CFG.Vertex>(new Comparer());
        internal static Dictionary<TypeReference, TypeRef> basic_llvm_types_created = new Dictionary<TypeReference, TypeRef>();
        internal static Dictionary<TypeReference, TypeRef> previous_llvm_types_created_global = new Dictionary<TypeReference, TypeRef>();
        internal static Dictionary<string, string> _rename_to_legal_llvm_name_cache = new Dictionary<string, string>();
        public int _start_index;

        public Converter(CFG mcfg)
        {
            _mcfg = mcfg;
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
                LLVM.Int1Type());

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

            basic_llvm_types_created.Add(
                typeof(string).ToMonoTypeReference(),
                LLVM.PointerType(LLVM.VoidType(), 0));



            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.tid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.tid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.tid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.tid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.ctaid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.ctaid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.ctaid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ctaid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.ntid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.ntid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.ntid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.ntid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));

            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.nctaid.x",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.x",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.nctaid.y",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.y",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));
            built_in_functions.Add("llvm.nvvm.read.ptx.sreg.nctaid.z",
                LLVM.AddFunction(
                    global_llvm_module,
                    "llvm.nvvm.read.ptx.sreg.nctaid.z",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { }, false)));



            built_in_functions.Add("System_String_get_Chars",
                LLVM.AddFunction(
                    global_llvm_module,
                    "System_String_get_Chars",
                    LLVM.FunctionType(LLVM.Int32Type(),
                        new TypeRef[] { LLVM.PointerType(LLVM.VoidType(), 0),
                            LLVM.Int32Type(),
                            LLVM.PointerType(LLVM.VoidType(), 0)
                        }, false)));

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

        private CFG.Vertex Eval(CFG.Vertex current, Dictionary<Tuple<TypeReference, GenericParameter>, System.Type> ops)
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

        private bool TypeUsingGeneric()
        { return false; }

        public List<CFG.Vertex> InstantiateGenerics(IEnumerable<CFG.Vertex> change_set, List<System.Type> list_of_data_types_used, List<Mono.Cecil.TypeReference> list_of_mono_data_types_used)
        {
            // Start a new change set so we can update edges and other properties for the new nodes
            // in the graph.
            int change_set_id2 = _mcfg.StartChangeSet();

            // Perform in-order traversal to generate instantiated type information.
            IEnumerable<CFG.Vertex> reverse_change_set = change_set.Reverse();

            // We need to do bookkeeping of what nodes to consider.
            Stack<CFG.Vertex> instantiated_nodes = new Stack<CFG.Vertex>(reverse_change_set);

            while (instantiated_nodes.Count > 0)
            {
                CFG.Vertex basic_block = instantiated_nodes.Pop();

                if (Campy.Utils.Options.IsOn("jit_trace"))
                    System.Console.WriteLine("Considering " + basic_block.Name);

                // If a block associated with method contains generics,
                // we need to duplicate the node and add in type information
                // about the generic type with that is actually used.
                // So, for example, if the method contains a parameter of type
                // "T", then we add in a mapping of T to the actual data type
                // used, e.g., Integer, or what have you. When it is compiled,
                // to LLVM, the mapped data type will be used!
                MethodReference method = basic_block.ExpectedCalleeSignature;
                var declaring_type = method.DeclaringType;

                {
                    // Let's first consider the parameter types to the function.
                    var parameters = method.Parameters;
                    for (int k = 0; k < parameters.Count; ++k)
                    {
                        ParameterDefinition par = parameters[k];
                        var type_to_consider = par.ParameterType;
                        type_to_consider = Converter.FromGenericParameterToTypeReference(type_to_consider, method.DeclaringType as GenericInstanceType);
                        if (type_to_consider.ContainsGenericParameter)
                        {
                            var declaring_type_of_considered_type = type_to_consider.DeclaringType;

                            // "type_to_consider" is generic, so find matching
                            // type, make mapping, and node copy.
                            for (int i = 0; i < list_of_data_types_used.Count; ++i)
                            {
                                var data_type_used = list_of_mono_data_types_used[i];
                                if (data_type_used == null) continue;
                                var sys_data_type_used = list_of_data_types_used[i];
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
                                    var previous = basic_block;
                                    for (; previous != null; previous = previous.PreviousVertex)
                                    {
                                        var old_node = FindInstantiatedBasicBlock(previous, type_to_consider, xx);
                                        if (old_node != null)
                                            break;
                                    }
                                    if (previous != null) continue;
                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new CFG.Vertex() { Name = new_node_id.ToString() });
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = basic_block.Instructions;
                                    new_cfg_node.ExpectedCalleeSignature = basic_block.ExpectedCalleeSignature;
                                    new_cfg_node.RewrittenCalleeSignature = basic_block.RewrittenCalleeSignature;
                                    new_cfg_node.PreviousVertex = basic_block;
                                    var b = type_to_consider as GenericParameter;
                                    var bb = new Tuple<TypeReference, GenericParameter>(type_to_consider, b);
                                    new_cfg_node.OpFromPreviousNode = new Tuple<Tuple<TypeReference, GenericParameter>, System.Type>(bb, xx);
                                    var previous_list = basic_block.OpsFromOriginal;
                                    if (previous_list != null) new_cfg_node.OpsFromOriginal = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>(previous_list);
                                    else new_cfg_node.OpsFromOriginal = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();

                                    var a = new_cfg_node.OpFromPreviousNode.Item1 as Tuple<TypeReference, GenericParameter>;

                                    new_cfg_node.OpsFromOriginal.Add(
                                        a,
                                        new_cfg_node.OpFromPreviousNode.Item2);
                                    if (basic_block.OriginalVertex == null) new_cfg_node.OriginalVertex = basic_block;
                                    else new_cfg_node.OriginalVertex = basic_block.OriginalVertex;

                                    // Add in rewrites.
                                    //new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    //new_cfg_node.node_type_map.Add(type_to_consider, xx);
                                    EnterInstantiatedBasicBlock(basic_block, type_to_consider, xx, new_cfg_node);
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
                                var sys_data_type_used_is_generic_type = sys_data_type_used.IsGenericType;
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
                                    var previous = basic_block;
                                    for (; previous != null; previous = previous.PreviousVertex)
                                    {
                                        var old_node = FindInstantiatedBasicBlock(previous, type_to_consider, xx);
                                        if (old_node != null)
                                            break;
                                    }
                                    if (previous != null) continue;
                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new CFG.Vertex(){Name = new_node_id.ToString()});
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = basic_block.Instructions;
                                    new_cfg_node.ExpectedCalleeSignature = basic_block.ExpectedCalleeSignature;
                                    new_cfg_node.RewrittenCalleeSignature = basic_block.RewrittenCalleeSignature;
                                    new_cfg_node.PreviousVertex = basic_block;
                                    var b = type_to_consider as GenericParameter;
                                    var bb = new Tuple<TypeReference, GenericParameter>(type_to_consider, b);
                                    new_cfg_node.OpFromPreviousNode = new Tuple<Tuple<TypeReference, GenericParameter>, System.Type>(bb, xx);
                                    var previous_list = basic_block.OpsFromOriginal;
                                    if (previous_list != null) new_cfg_node.OpsFromOriginal = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>(previous_list);
                                    else new_cfg_node.OpsFromOriginal = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();

                                    var a = new_cfg_node.OpFromPreviousNode.Item1 as Tuple<TypeReference, GenericParameter>;
                                    new_cfg_node.OpsFromOriginal.Add(
                                        a,
                                        new_cfg_node.OpFromPreviousNode.Item2);

                                    if (basic_block.OriginalVertex == null) new_cfg_node.OriginalVertex = basic_block;
                                    else new_cfg_node.OriginalVertex = basic_block.OriginalVertex;
                                    
                                    // Add in rewrites.
                                    //new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    //new_cfg_node.node_type_map.Add(type_to_consider, xx);
                                    EnterInstantiatedBasicBlock(basic_block, type_to_consider, xx, new_cfg_node);
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
                        var type_to_consider_system_type = type_to_consider.ToSystemType();
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
                                    var previous = basic_block;
                                    for (; previous != null; previous = previous.PreviousVertex)
                                    {
                                        var old_node = FindInstantiatedBasicBlock(previous, type_to_consider, xx);
                                        if (old_node != null)
                                            break;
                                    }
                                    if (previous != null) continue;
                                    // Rewrite node
                                    int new_node_id = _mcfg.NewNodeNumber();
                                    var new_node = _mcfg.AddVertex(new CFG.Vertex(){Name = new_node_id.ToString()});
                                    var new_cfg_node = (CFG.Vertex)new_node;
                                    new_cfg_node.Instructions = basic_block.Instructions;
                                    new_cfg_node.ExpectedCalleeSignature = basic_block.ExpectedCalleeSignature;
                                    new_cfg_node.RewrittenCalleeSignature = basic_block.RewrittenCalleeSignature;
                                    new_cfg_node.PreviousVertex = basic_block;
                                    var b = type_to_consider as GenericParameter;
                                    var bb = new Tuple<TypeReference, GenericParameter>(type_to_consider, b);
                                    new_cfg_node.OpFromPreviousNode = new Tuple<Tuple<TypeReference, GenericParameter>, System.Type>(bb, xx);
                                    var previous_list = basic_block.OpsFromOriginal;
                                    if (previous_list != null) new_cfg_node.OpsFromOriginal = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>(previous_list);
                                    else new_cfg_node.OpsFromOriginal = new Dictionary<Tuple<TypeReference, GenericParameter>, System.Type>();

                                    Tuple<TypeReference, GenericParameter> a = new_cfg_node.OpFromPreviousNode.Item1;
                                    new_cfg_node.OpsFromOriginal.Add(
                                        a,
                                        new_cfg_node.OpFromPreviousNode.Item2);
                                    if (basic_block.OriginalVertex == null) new_cfg_node.OriginalVertex = basic_block;
                                    else new_cfg_node.OriginalVertex = basic_block.OriginalVertex;

                                    // Add in rewrites.
                                    //new_cfg_node.node_type_map = new MultiMap<TypeReference, System.Type>(lv.node_type_map);
                                    //new_cfg_node.node_type_map.Add(type_to_consider, xx);
                                    EnterInstantiatedBasicBlock(basic_block, type_to_consider, xx, new_cfg_node);
                                    System.Console.WriteLine("Adding new node " + new_cfg_node.Name);

                                    // Push this node back on the stack.
                                    instantiated_nodes.Push(new_cfg_node);
                                }
                            }
                        }
                    }
                }
            }

            this._mcfg.OutputEntireGraph();

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
                    _mcfg.AddEdge(new CFG.Edge() {From = v, To = vto_mapped});
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

        private void CompilePart1(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeReference> list_of_data_types_used)
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

                if (!IsFullyInstantiatedNode(bb))
                {
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine("skipping -- not fully instantiated block the contains generics.");
                    continue;
                }

                MethodReference method = bb.ExpectedCalleeSignature;
                List<ParameterDefinition> parameters = method.Parameters.ToList();
                List<ParameterReference> instantiated_parameters = new List<ParameterReference>();

                ModuleRef mod = global_llvm_module; // LLVM.ModuleCreateWithName(mn);
                bb.Module = mod;

                uint count = (uint) bb.NumberOfArguments;

                TypeRef[] param_types = new TypeRef[count];
                int current = 0;
                if (count > 0)
                {
                    if (bb.HasStructReturnValue)
                    {
                        Type t = new Type(method.ReturnType);
                        param_types[current++] = LLVM.PointerType(t.IntermediateType, 0);
                    }
                    if (bb.HasThis)
                    {
                        Type t = new Type(method.DeclaringType);
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
                            type_reference_of_parameter = FromGenericParameterToTypeReference(
                                type_reference_of_parameter, git);
                        }
                        Type t = new Type(type_reference_of_parameter);
                        param_types[current++] = t.IntermediateType;
                    }

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                    {
                        foreach (var pp in param_types)
                        {
                            string a = LLVM.PrintTypeToString(pp);
                            System.Console.WriteLine(" " + a);
                        }
                    }
                }

                //mi2 = FromGenericParameterToTypeReference(typeof(void).ToMonoTypeReference(), null);
                Type t_ret = new Type(FromGenericParameterToTypeReference(method.ReturnType, method.DeclaringType as GenericInstanceType));
                if (bb.HasStructReturnValue)
                {
                    t_ret = new Type(typeof(void).ToMonoTypeReference());
                }
                TypeRef ret_type = t_ret.IntermediateType;
                TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                ValueRef fun = LLVM.AddFunction(mod,
                    Converter.RenameToLegalLLVMName(Converter.MethodName(method)), met_type);
                BasicBlockRef entry = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                bb.BasicBlock = entry;
                bb.MethodValueRef = fun;
                var t_fun = LLVM.TypeOf(fun);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(Converter.global_llvm_module);
                if (t_fun_con != context) throw new Exception("not equal");
                //////////LLVM.VerifyFunction(fun, VerifierFailureAction.PrintMessageAction);
                BuilderRef builder = LLVM.CreateBuilder();
                bb.Builder = builder;
                LLVM.PositionBuilderAtEnd(builder, entry);
            }
        }

        private void CompilePart2(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeReference> list_of_data_types_used)
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
                    var fun = lvv_ent.MethodValueRef;
                    var t_fun = LLVM.TypeOf(fun);
                    var t_fun_con = LLVM.GetTypeContext(t_fun);
                    var context = LLVM.GetModuleContext(Converter.global_llvm_module);
                    if (t_fun_con != context) throw new Exception("not equal");
                    //LLVM.VerifyFunction(fun, VerifierFailureAction.PrintMessageAction);
                    var llvm_bb = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                    bb.BasicBlock = llvm_bb;
                    bb.MethodValueRef = lvv_ent.MethodValueRef;
                    BuilderRef builder = LLVM.CreateBuilder();
                    bb.Builder = builder;
                    LLVM.PositionBuilderAtEnd(builder, llvm_bb);
                }
            }
        }

        private void CompilePart3(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeReference> list_of_data_types_used)
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

        private void CompilePart4(IEnumerable<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeReference> list_of_data_types_used, List<CFG.Vertex> entries,
            out List<CFG.Vertex> unreachable, out List<CFG.Vertex> change_set_minus_unreachable)
        {
            unreachable = new List<CFG.Vertex>();
            change_set_minus_unreachable = new List<CFG.Vertex>(basic_blocks_to_compile);
            {
                // Create DFT order of all nodes from entries.
                var objs = entries.Select(x => x.Name);
                var ordered_list =  new TarjanNoBackEdges<CFG.Vertex,CFG.Edge>(_mcfg).ToList();
                ordered_list.Reverse();

                //Graphs.DFSPreorder<int>
                //    ordered_list = new Graphs.DFSPreorder<int>(
                //        _mcfg,
                //        objs
                //    );

                List<CFG.Vertex> visited = new List<CFG.Vertex>();
                foreach (var ob in ordered_list)
                {
                    CFG.Vertex node = ob;
                    if (!IsFullyInstantiatedNode(node))
                        continue;
                    if (visited.Contains(node))
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
            return mr.FullName;

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

        private void CompilePart6(IEnumerable<CFG.Vertex> basic_blocks_to_compile,
            List<Mono.Cecil.TypeReference> list_of_data_types_used, List<CFG.Vertex> entries,
            List<CFG.Vertex> unreachable, List<CFG.Vertex> change_set_minus_unreachable)
        {
            List<CFG.Vertex> work = new List<CFG.Vertex>(change_set_minus_unreachable);
            while (work.Count != 0)
            {
                // Create DFT order of all nodes.
                var ordered_list = new TarjanNoBackEdges<CFG.Vertex,CFG.Edge>(_mcfg).ToList();
                ordered_list.Reverse();

                //IEnumerable<int> objs = entries.Select(x => x.Name);
                //Graphs.DFSPreorder<int>
                //    ordered_list = new Graphs.DFSPreorder<int>(
                //        _mcfg,
                //        objs
                //    );

                List<CFG.Vertex> visited = new List<CFG.Vertex>();
                // Compute stack size for each basic block, processing nodes on work list
                // in DFT order.
                foreach (var ob in ordered_list)
                {
                    CFG.Vertex node = ob;
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
                        node.StackLevelIn =
                            node.NumberOfLocals + node.NumberOfArguments;
                    }
                    else
                    {
                        int in_level = -1;
                        foreach (CFG.Vertex pred in _mcfg.PredecessorNodes(node))
                        {
                            // Do not consider interprocedural edges when computing stack size.
                            if (pred.ExpectedCalleeSignature != node.ExpectedCalleeSignature)
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
                    foreach (var inst in llvm_nodez.Instructions)
                    {
                        level_pre = level_after;
                        inst.ComputeStackLevel(this, ref level_after);
                        if (!(level_after >= node.NumberOfLocals + node.NumberOfArguments))
                            throw new Exception("Stack computation off. Internal error.");
                    }
                    llvm_nodez.StackLevelOut = level_after;
                    // Verify return node that it makes sense.
                    if (node.IsReturn && !unreachable.Contains(node))
                    {
                        if (llvm_nodez.StackLevelOut ==
                            node.NumberOfArguments
                            + node.NumberOfLocals
                            + (node.HasScalarReturnValue ? 1 : 0))
                            ;
                        else
                        {
                            throw new Exception("Failed stack level out check");
                        }
                    }
                    foreach (CFG.Vertex succ in node._graph.SuccessorNodes(node))
                    {
                        // If it's an interprocedural edge, nothing to pass on.
                        if (succ.ExpectedCalleeSignature != node.ExpectedCalleeSignature)
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

        private List<CFG.Vertex> RemoveBasicBlocksAlreadyCompiled(List<CFG.Vertex> basic_blocks_to_compile)
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

        public string CompileToLLVM(List<CFG.Vertex> basic_blocks_to_compile, List<Mono.Cecil.TypeReference> list_of_data_types_used,
            string basic_block_id, int start_index = 0)
        {
            _start_index = start_index;

            basic_blocks_to_compile = RemoveBasicBlocksAlreadyCompiled(basic_blocks_to_compile);

            CompilePart1(basic_blocks_to_compile, list_of_data_types_used);

            CompilePart2(basic_blocks_to_compile, list_of_data_types_used);

            CompilePart3(basic_blocks_to_compile, list_of_data_types_used);

            List<CFG.Vertex> entries = _mcfg.Vertices.Where(node => node.IsEntry).ToList();

            List<CFG.Vertex> unreachable;
            List<CFG.Vertex> change_set_minus_unreachable;
            CompilePart4(basic_blocks_to_compile, list_of_data_types_used, entries, out unreachable, out change_set_minus_unreachable);

            CompilePart6(basic_blocks_to_compile, list_of_data_types_used, entries,
                unreachable, change_set_minus_unreachable);

            {
                // Get a list of nodes to compile.
                List<CFG.Vertex> work = new List<CFG.Vertex>(change_set_minus_unreachable);

                // Get a list of the name of nodes to compile.
                var work_names = work.Select(v => v.Name);

                // Get a Tarjan DFS/SCC order of the nodes. Reverse it because we want to
                // proceed from entry basic block.
                //var ordered_list = new TarjanNoBackEdges<int>(_mcfg).GetEnumerable().Reverse();
                var ordered_list = new TarjanNoBackEdges<CFG.Vertex,CFG.Edge>(_mcfg).ToList();
                ordered_list.Reverse();

                // Eliminate all node names not in the work list.
                var order = ordered_list.Where(v => work_names.Contains(v.Name)).ToList();

                //// Set up the initial states associated with each node, that is, state into and state out of.
                //foreach (int ob in order)
                //{
                //    CFG.Vertex node = _mcfg.VertexSpace[_mcfg.NameSpace.BijectFromBasetype(ob)];
                //    CFG.Vertex llvm_node = node;
                //    llvm_node.StateIn = new State(node, true);
                //    llvm_node.StateOut = new State(node, false);
                //}

                Dictionary<CFG.Vertex, bool> visited = new Dictionary<CFG.Vertex, bool>();

                // Emit LLVM IR code, based on state and per-instruction simulation on that state.
                foreach (var ob in order)
                {
                    CFG.Vertex bb = ob;

                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                        System.Console.WriteLine("State computations for node " + bb.Name);

                    var state_in = new State(visited, bb, list_of_data_types_used);

                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    {
                        System.Console.WriteLine("state in output");
                        state_in.OutputTrace();
                    }

                    bb.StateIn = state_in;
                    bb.StateOut = new State(state_in);

                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    {
                        bb.OutputEntireNode();
                        state_in.OutputTrace();
                    }

                    Inst last_inst = null;
                    for (int i = 0; i < bb.Instructions.Count; ++i)
                    {
                        var inst = bb.Instructions[i];
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine(inst);
                        last_inst = inst;
                        inst = inst.Convert(this, bb.StateOut);
                        if (Campy.Utils.Options.IsOn("state_computation_trace"))
                            bb.StateOut.OutputTrace();
                    }
                    if (last_inst != null && (last_inst.OpCode.FlowControl == Mono.Cecil.Cil.FlowControl.Next
                        || last_inst.OpCode.FlowControl == FlowControl.Call))
                    {
                        // Need to insert instruction to branch to fall through.
                        var edge = bb._graph.SuccessorEdges(bb).FirstOrDefault();
                        var s = edge.To;
                        var br = LLVM.BuildBr(bb.Builder, s.BasicBlock);
                    }
                    visited[ob] = true;
                }

                // Finally, update phi functions with "incoming" information from predecessors.
                foreach (var ob in order)
                {
                    CFG.Vertex node = ob;
                    CFG.Vertex llvm_node = node;
                    int size = llvm_node.StateIn._stack.Count;
                    for (int i = 0; i < size; ++i)
                    {
                        var count = llvm_node._graph.Predecessors(llvm_node).Count();
                        if (count < 2) continue;
                        ValueRef res;
                        res = llvm_node.StateIn._stack[i].V;
                        if (!llvm_node.StateIn._phi.Contains(res)) continue;
                        ValueRef[] phi_vals = new ValueRef[count];
                        for (int c = 0; c < count; ++c)
                        {
                            var p = llvm_node._graph.PredecessorEdges(llvm_node).ToList()[c].From;
                            var plm = p;
                            var vr = plm.StateOut._stack[i];
                            phi_vals[c] = vr.V;
                        }
                        BasicBlockRef[] phi_blocks = new BasicBlockRef[count];
                        for (int c = 0; c < count; ++c)
                        {
                            var p = llvm_node._graph.PredecessorEdges(llvm_node).ToList()[c].From;
                            var plm = p;
                            phi_blocks[c] = plm.BasicBlock;
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
                        llvm_node.StateIn.OutputTrace();
                        llvm_node.StateOut.OutputTrace();
                    }
                }
            }

            if (Utils.Options.IsOn("name_trace"))
                NameTableTrace();


            {
                var module = Converter.global_llvm_module;
                var basic_block = GetBasicBlock(basic_block_id);

                if (Campy.Utils.Options.IsOn("module_trace"))
                    LLVM.DumpModule(module);

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
                ContextRef context_ref = LLVM.GetModuleContext(Converter.global_llvm_module);
                ValueRef kernelMd = LLVM.MDNodeInContext(
                    context_ref, new ValueRef[3]
                {
                    basic_block.MethodValueRef,
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
                        ptx = ptx.Replace("3.2", "5.0");

                        ptx = ptx + "\n" + System_String_get_Chars;

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

                    FieldInfo[] target_type_fieldinfo = target_type.GetFields(
                        System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                        | System.Reflection.BindingFlags.Public
                        //| System.Reflection.BindingFlags.Static
                    );
                    var rffi = target_type.GetRuntimeFields();

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

        public CFG.Vertex GetBasicBlock(string block_id)
        {
            return _mcfg.Vertices.Where(i => i.IsEntry && i.Name == block_id).FirstOrDefault();
        }

        public CUfunction GetCudaFunction(string basic_block_id, string ptx)
        {
            var basic_block = GetBasicBlock(basic_block_id);
            var method = basic_block.ExpectedCalleeSignature;

            var res = Cuda.cuDeviceGet(out int device, 0);
            CheckCudaError(res);
            res = Cuda.cuDeviceGetPCIBusId(out string pciBusId, 100, device);
            CheckCudaError(res);
            res = Cuda.cuDeviceGetName(out string name, 100, device);
            CheckCudaError(res);
            res = Cuda.cuCtxCreate_v2(out CUcontext cuContext, 0, device);
            CheckCudaError(res);
            IntPtr ptr = Marshal.StringToHGlobalAnsi(ptx);
            res = Cuda.cuModuleLoadData(out CUmodule cuModule, ptr);
            CheckCudaError(res);
            var normalized_method_name = Converter.RenameToLegalLLVMName(Converter.MethodName(basic_block.ExpectedCalleeSignature));
            res = Cuda.cuModuleGetFunction(out CUfunction helloWorld, cuModule, normalized_method_name);
            CheckCudaError(res);
            return helloWorld;
        }

        public static void CheckCudaError(Swigged.Cuda.CUresult res)
        {
            if (res != CUresult.CUDA_SUCCESS)
            {
                Cuda.cuGetErrorString(res, out IntPtr pStr);
                var cuda_error = Marshal.PtrToStringAnsi(pStr);
                throw new Exception("CUDA error: " + cuda_error);
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

	private static string System_String_get_Chars = @"
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-22781540
// Cuda compilation tools, release 9.0, V9.0.176
// Based on LLVM 3.4svn
//

.version 6.0
.target sm_30
.address_size 64

	// .globl	_Z28System_String_ctor_CharInt32PhS_S_
.extern .func  (.param .b64 func_retval0) _Z10Heap_AllocP12tMD_TypeDef_j
(
	.param .b64 _Z10Heap_AllocP12tMD_TypeDef_j_param_0,
	.param .b32 _Z10Heap_AllocP12tMD_TypeDef_j_param_1
)
;
.extern .func  (.param .b32 func_retval0) _Z9gpumemcmpPKvS0_y
(
	.param .b64 _Z9gpumemcmpPKvS0_y_param_0,
	.param .b64 _Z9gpumemcmpPKvS0_y_param_1,
	.param .b64 _Z9gpumemcmpPKvS0_y_param_2
)
;
.extern .func  (.param .b64 func_retval0) _Z22MetaData_GetUserStringP10tMetaData_jPj
(
	.param .b64 _Z22MetaData_GetUserStringP10tMetaData_jPj_param_0,
	.param .b32 _Z22MetaData_GetUserStringP10tMetaData_jPj_param_1,
	.param .b64 _Z22MetaData_GetUserStringP10tMetaData_jPj_param_2
)
;
.extern .func  (.param .b64 func_retval0) _Z9gpustrlenPKc
(
	.param .b64 _Z9gpustrlenPKc_param_0
)
;
.extern .global .align 8 .b64 types;

.visible .func  (.param .b64 func_retval0) _Z28System_String_ctor_CharInt32PhS_S_(
	.param .b64 _Z28System_String_ctor_CharInt32PhS_S__param_0,
	.param .b64 _Z28System_String_ctor_CharInt32PhS_S__param_1,
	.param .b64 _Z28System_String_ctor_CharInt32PhS_S__param_2
)
{
	.reg .pred 	%p<7>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<23>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd4, [_Z28System_String_ctor_CharInt32PhS_S__param_1];
	ld.param.u64 	%rd3, [_Z28System_String_ctor_CharInt32PhS_S__param_2];
	ld.u16 	%rs1, [%rd4];
	ld.u32 	%r1, [%rd4+4];
	shl.b32 	%r10, %r1, 1;
	add.s32 	%r11, %r10, 4;
	ld.global.u64 	%rd5, [types];
	ld.u64 	%rd6, [%rd5+72];
	// Callseq Start 0
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd6;
	.param .b32 param1;
	st.param.b32	[param1+0], %r11;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd1, [retval0+0];
	
	//{
	}// Callseq End 0
	st.u32 	[%rd1], %r1;
	setp.eq.s32	%p1, %r1, 0;
	@%p1 bra 	BB0_9;

	and.b32  	%r2, %r1, 3;
	setp.eq.s32	%p2, %r2, 0;
	mov.u32 	%r22, 0;
	@%p2 bra 	BB0_7;

	setp.eq.s32	%p3, %r2, 1;
	mov.u32 	%r20, 0;
	@%p3 bra 	BB0_6;

	setp.eq.s32	%p4, %r2, 2;
	mov.u32 	%r19, 0;
	@%p4 bra 	BB0_5;

	st.u16 	[%rd1+4], %rs1;
	mov.u32 	%r19, 1;

BB0_5:
	mul.wide.u32 	%rd7, %r19, 2;
	add.s64 	%rd8, %rd1, %rd7;
	st.u16 	[%rd8+4], %rs1;
	add.s32 	%r20, %r19, 1;

BB0_6:
	mul.wide.u32 	%rd9, %r20, 2;
	add.s64 	%rd10, %rd1, %rd9;
	st.u16 	[%rd10+4], %rs1;
	add.s32 	%r22, %r20, 1;

BB0_7:
	setp.lt.u32	%p5, %r1, 4;
	@%p5 bra 	BB0_9;

BB0_8:
	mul.wide.u32 	%rd11, %r22, 2;
	add.s64 	%rd12, %rd1, 4;
	add.s64 	%rd13, %rd12, %rd11;
	st.u16 	[%rd13], %rs1;
	add.s32 	%r16, %r22, 1;
	mul.wide.u32 	%rd14, %r16, 2;
	add.s64 	%rd15, %rd12, %rd14;
	st.u16 	[%rd15], %rs1;
	add.s32 	%r17, %r22, 2;
	mul.wide.u32 	%rd16, %r17, 2;
	add.s64 	%rd17, %rd12, %rd16;
	st.u16 	[%rd17], %rs1;
	add.s32 	%r18, %r22, 3;
	mul.wide.u32 	%rd18, %r18, 2;
	add.s64 	%rd19, %rd12, %rd18;
	st.u16 	[%rd19], %rs1;
	add.s32 	%r22, %r22, 4;
	setp.lt.u32	%p6, %r22, %r1;
	@%p6 bra 	BB0_8;

BB0_9:
	st.u64 	[%rd3], %rd1;
	mov.u64 	%rd20, 0;
	st.param.b64	[func_retval0+0], %rd20;
	ret;
}

	// .globl	_Z30System_String_ctor_CharAIntIntPhS_S_
.visible .func  (.param .b64 func_retval0) _Z30System_String_ctor_CharAIntIntPhS_S_(
	.param .b64 _Z30System_String_ctor_CharAIntIntPhS_S__param_0,
	.param .b64 _Z30System_String_ctor_CharAIntIntPhS_S__param_1,
	.param .b64 _Z30System_String_ctor_CharAIntIntPhS_S__param_2
)
{
	.reg .pred 	%p<3>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<19>;


	ld.param.u64 	%rd9, [_Z30System_String_ctor_CharAIntIntPhS_S__param_1];
	ld.param.u64 	%rd7, [_Z30System_String_ctor_CharAIntIntPhS_S__param_2];
	ld.u64 	%rd10, [%rd9];
	ld.u32 	%r1, [%rd9+4];
	ld.u32 	%r2, [%rd9+8];
	shl.b32 	%r3, %r2, 1;
	add.s32 	%r4, %r3, 4;
	ld.global.u64 	%rd11, [types];
	ld.u64 	%rd12, [%rd11+72];
	// Callseq Start 1
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32	[param1+0], %r4;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd1, [retval0+0];
	
	//{
	}// Callseq End 1
	st.u32 	[%rd1], %r2;
	add.s64 	%rd2, %rd1, 4;
	shl.b32 	%r5, %r1, 1;
	cvt.u64.u32	%rd13, %r5;
	add.s64 	%rd14, %rd13, %rd10;
	add.s64 	%rd3, %rd14, 4;
	cvt.u64.u32	%rd4, %r3;
	mov.u64 	%rd18, 0;
	setp.eq.s32	%p1, %r3, 0;
	@%p1 bra 	BB1_2;

BB1_1:
	add.s64 	%rd15, %rd3, %rd18;
	ld.u8 	%rs1, [%rd15];
	add.s64 	%rd16, %rd2, %rd18;
	st.u8 	[%rd16], %rs1;
	add.s64 	%rd18, %rd18, 1;
	setp.lt.u64	%p2, %rd18, %rd4;
	@%p2 bra 	BB1_1;

BB1_2:
	st.u64 	[%rd7], %rd1;
	mov.u64 	%rd17, 0;
	st.param.b64	[func_retval0+0], %rd17;
	ret;
}

	// .globl	_Z31System_String_ctor_StringIntIntPhS_S_
.visible .func  (.param .b64 func_retval0) _Z31System_String_ctor_StringIntIntPhS_S_(
	.param .b64 _Z31System_String_ctor_StringIntIntPhS_S__param_0,
	.param .b64 _Z31System_String_ctor_StringIntIntPhS_S__param_1,
	.param .b64 _Z31System_String_ctor_StringIntIntPhS_S__param_2
)
{
	.reg .pred 	%p<3>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<19>;


	ld.param.u64 	%rd9, [_Z31System_String_ctor_StringIntIntPhS_S__param_1];
	ld.param.u64 	%rd7, [_Z31System_String_ctor_StringIntIntPhS_S__param_2];
	ld.u64 	%rd10, [%rd9];
	ld.u32 	%r1, [%rd9+4];
	ld.u32 	%r2, [%rd9+8];
	shl.b32 	%r3, %r2, 1;
	add.s32 	%r4, %r3, 4;
	ld.global.u64 	%rd11, [types];
	ld.u64 	%rd12, [%rd11+72];
	// Callseq Start 2
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32	[param1+0], %r4;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd1, [retval0+0];
	
	//{
	}// Callseq End 2
	st.u32 	[%rd1], %r2;
	add.s64 	%rd2, %rd1, 4;
	mul.wide.u32 	%rd13, %r1, 2;
	add.s64 	%rd14, %rd10, %rd13;
	add.s64 	%rd3, %rd14, 4;
	cvt.u64.u32	%rd4, %r3;
	mov.u64 	%rd18, 0;
	setp.eq.s32	%p1, %r3, 0;
	@%p1 bra 	BB2_2;

BB2_1:
	add.s64 	%rd15, %rd3, %rd18;
	ld.u8 	%rs1, [%rd15];
	add.s64 	%rd16, %rd2, %rd18;
	st.u8 	[%rd16], %rs1;
	add.s64 	%rd18, %rd18, 1;
	setp.lt.u64	%p2, %rd18, %rd4;
	@%p2 bra 	BB2_1;

BB2_2:
	st.u64 	[%rd7], %rd1;
	mov.u64 	%rd17, 0;
	st.param.b64	[func_retval0+0], %rd17;
	ret;
}

	// .globl	_Z23System_String_get_CharsPhS_S_
.visible .func  (.param .b64 func_retval0) System_String_get_Chars(
	.param .b64 _Z23System_String_get_CharsPhS_S__param_0,
	.param .b64 _Z23System_String_get_CharsPhS_S__param_1,
	.param .b64 _Z23System_String_get_CharsPhS_S__param_2
)
{
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<7>;


	ld.param.u64 	%rd1, [_Z23System_String_get_CharsPhS_S__param_0];
	ld.param.u64 	%rd2, [_Z23System_String_get_CharsPhS_S__param_1];
	ld.param.u64 	%rd3, [_Z23System_String_get_CharsPhS_S__param_2];
	ld.u32 	%r1, [%rd2];
	mul.wide.u32 	%rd4, %r1, 2;
	add.s64 	%rd5, %rd1, %rd4;
	ld.u16 	%r2, [%rd5+4];
	st.u32 	[%rd3], %r2;
	mov.u64 	%rd6, 0;
	st.param.b64	[func_retval0+0], %rd6;
	ret;
}

	// .globl	_Z28System_String_InternalConcatPhS_S_
.visible .func  (.param .b64 func_retval0) _Z28System_String_InternalConcatPhS_S_(
	.param .b64 _Z28System_String_InternalConcatPhS_S__param_0,
	.param .b64 _Z28System_String_InternalConcatPhS_S__param_1,
	.param .b64 _Z28System_String_InternalConcatPhS_S__param_2
)
{
	.reg .pred 	%p<5>;
	.reg .b16 	%rs<3>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd18, [_Z28System_String_InternalConcatPhS_S__param_1];
	ld.param.u64 	%rd16, [_Z28System_String_InternalConcatPhS_S__param_2];
	ld.u64 	%rd2, [%rd18];
	ld.u64 	%rd1, [%rd18+8];
	ld.u32 	%r1, [%rd2];
	ld.u32 	%r2, [%rd1];
	add.s32 	%r3, %r2, %r1;
	shl.b32 	%r4, %r3, 1;
	add.s32 	%r5, %r4, 4;
	ld.global.u64 	%rd19, [types];
	ld.u64 	%rd20, [%rd19+72];
	// Callseq Start 3
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd20;
	.param .b32 param1;
	st.param.b32	[param1+0], %r5;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd4, [retval0+0];
	
	//{
	}// Callseq End 3
	st.u32 	[%rd4], %r3;
	add.s64 	%rd6, %rd4, 4;
	add.s64 	%rd7, %rd2, 4;
	ld.u32 	%r6, [%rd2];
	shl.b32 	%r7, %r6, 1;
	cvt.u64.u32	%rd8, %r7;
	mov.u64 	%rd29, 0;
	setp.eq.s32	%p1, %r7, 0;
	@%p1 bra 	BB4_2;

BB4_1:
	add.s64 	%rd21, %rd7, %rd29;
	ld.u8 	%rs1, [%rd21];
	add.s64 	%rd22, %rd6, %rd29;
	st.u8 	[%rd22], %rs1;
	add.s64 	%rd29, %rd29, 1;
	setp.lt.u64	%p2, %rd29, %rd8;
	@%p2 bra 	BB4_1;

BB4_2:
	ld.u32 	%r8, [%rd2];
	mul.wide.u32 	%rd24, %r8, 2;
	add.s64 	%rd25, %rd4, %rd24;
	add.s64 	%rd11, %rd25, 4;
	add.s64 	%rd12, %rd1, 4;
	ld.u32 	%r9, [%rd1];
	shl.b32 	%r10, %r9, 1;
	cvt.u64.u32	%rd13, %r10;
	mov.u64 	%rd30, 0;
	setp.eq.s32	%p3, %r10, 0;
	@%p3 bra 	BB4_4;

BB4_3:
	add.s64 	%rd26, %rd12, %rd30;
	ld.u8 	%rs2, [%rd26];
	add.s64 	%rd27, %rd11, %rd30;
	st.u8 	[%rd27], %rs2;
	add.s64 	%rd30, %rd30, 1;
	setp.lt.u64	%p4, %rd30, %rd13;
	@%p4 bra 	BB4_3;

BB4_4:
	st.u64 	[%rd16], %rd4;
	mov.u64 	%rd28, 0;
	st.param.b64	[func_retval0+0], %rd28;
	ret;
}

	// .globl	_Z26System_String_InternalTrimPhS_S_
.visible .func  (.param .b64 func_retval0) _Z26System_String_InternalTrimPhS_S_(
	.param .b64 _Z26System_String_InternalTrimPhS_S__param_0,
	.param .b64 _Z26System_String_InternalTrimPhS_S__param_1,
	.param .b64 _Z26System_String_InternalTrimPhS_S__param_2
)
{
	.reg .pred 	%p<17>;
	.reg .b16 	%rs<6>;
	.reg .b32 	%r<33>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd10, [_Z26System_String_InternalTrimPhS_S__param_0];
	ld.param.u64 	%rd12, [_Z26System_String_InternalTrimPhS_S__param_1];
	ld.param.u64 	%rd11, [_Z26System_String_InternalTrimPhS_S__param_2];
	ld.u64 	%rd13, [%rd12];
	add.s64 	%rd1, %rd13, 4;
	ld.u32 	%r1, [%rd13];
	ld.u32 	%r2, [%rd12+4];
	and.b32  	%r17, %r2, 1;
	setp.eq.b32	%p1, %r17, 1;
	not.pred 	%p2, %p1;
	ld.u32 	%r32, [%rd10];
	setp.eq.s32	%p3, %r32, 0;
	or.pred  	%p4, %p2, %p3;
	mov.u32 	%r28, 0;
	@%p4 bra 	BB5_8;

	add.s64 	%rd2, %rd10, 4;
	mov.u32 	%r18, 0;
	mov.u32 	%r4, %r18;

BB5_2:
	mul.wide.u32 	%rd14, %r4, 2;
	add.s64 	%rd15, %rd2, %rd14;
	ld.u16 	%rs1, [%rd15];
	setp.eq.s32	%p5, %r1, 0;
	mov.u32 	%r27, %r18;
	@%p5 bra 	BB5_3;

BB5_4:
	mul.wide.u32 	%rd16, %r27, 2;
	add.s64 	%rd17, %rd1, %rd16;
	ld.u16 	%rs3, [%rd17];
	setp.eq.s16	%p6, %rs1, %rs3;
	add.s32 	%r27, %r27, 1;
	@%p6 bra 	BB5_7;

	setp.lt.u32	%p7, %r27, %r1;
	@%p7 bra 	BB5_4;
	bra.uni 	BB5_6;

BB5_7:
	add.s32 	%r4, %r4, 1;
	setp.lt.u32	%p8, %r4, %r32;
	mov.u32 	%r28, 0;
	@%p8 bra 	BB5_2;
	bra.uni 	BB5_8;

BB5_6:
	mov.u32 	%r28, %r4;

BB5_8:
	and.b32  	%r21, %r2, 2;
	setp.eq.s32	%p9, %r21, 0;
	@%p9 bra 	BB5_17;

	add.s32 	%r29, %r32, -1;
	setp.lt.u32	%p10, %r29, %r28;
	@%p10 bra 	BB5_17;

	add.s64 	%rd3, %rd10, 4;
	mov.u32 	%r30, %r32;

BB5_11:
	mov.u32 	%r10, %r29;
	mul.wide.u32 	%rd18, %r10, 2;
	add.s64 	%rd19, %rd3, %rd18;
	ld.u16 	%rs2, [%rd19];
	setp.eq.s32	%p11, %r1, 0;
	mov.u32 	%r31, 0;
	@%p11 bra 	BB5_12;

BB5_13:
	mul.wide.u32 	%rd20, %r31, 2;
	add.s64 	%rd21, %rd1, %rd20;
	ld.u16 	%rs4, [%rd21];
	setp.eq.s16	%p12, %rs2, %rs4;
	add.s32 	%r31, %r31, 1;
	@%p12 bra 	BB5_16;

	setp.lt.u32	%p13, %r31, %r1;
	@%p13 bra 	BB5_13;
	bra.uni 	BB5_15;

BB5_16:
	add.s32 	%r29, %r10, -1;
	setp.ge.u32	%p14, %r29, %r28;
	mov.u32 	%r30, %r10;
	@%p14 bra 	BB5_11;
	bra.uni 	BB5_17;

BB5_15:
	mov.u32 	%r32, %r30;

BB5_17:
	sub.s32 	%r23, %r32, %r28;
	shl.b32 	%r24, %r23, 1;
	add.s32 	%r25, %r24, 4;
	ld.global.u64 	%rd23, [types];
	ld.u64 	%rd24, [%rd23+72];
	// Callseq Start 4
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd24;
	.param .b32 param1;
	st.param.b32	[param1+0], %r25;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd4, [retval0+0];
	
	//{
	}// Callseq End 4
	st.u32 	[%rd4], %r23;
	add.s64 	%rd5, %rd4, 4;
	mul.wide.u32 	%rd25, %r28, 2;
	add.s64 	%rd26, %rd10, %rd25;
	add.s64 	%rd6, %rd26, 4;
	cvt.u64.u32	%rd7, %r24;
	mov.u64 	%rd30, 0;
	setp.eq.s32	%p15, %r24, 0;
	@%p15 bra 	BB5_19;

BB5_18:
	add.s64 	%rd27, %rd6, %rd30;
	ld.u8 	%rs5, [%rd27];
	add.s64 	%rd28, %rd5, %rd30;
	st.u8 	[%rd28], %rs5;
	add.s64 	%rd30, %rd30, 1;
	setp.lt.u64	%p16, %rd30, %rd7;
	@%p16 bra 	BB5_18;

BB5_19:
	st.u64 	[%rd11], %rd4;
	mov.u64 	%rd29, 0;
	st.param.b64	[func_retval0+0], %rd29;
	ret;

BB5_3:
	mov.u32 	%r28, %r4;
	bra.uni 	BB5_8;

BB5_12:
	mov.u32 	%r32, %r30;
	bra.uni 	BB5_17;
}

	// .globl	_Z20System_String_EqualsPhS_S_
.visible .func  (.param .b64 func_retval0) _Z20System_String_EqualsPhS_S_(
	.param .b64 _Z20System_String_EqualsPhS_S__param_0,
	.param .b64 _Z20System_String_EqualsPhS_S__param_1,
	.param .b64 _Z20System_String_EqualsPhS_S__param_2
)
{
	.reg .pred 	%p<7>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd4, [_Z20System_String_EqualsPhS_S__param_1];
	ld.param.u64 	%rd3, [_Z20System_String_EqualsPhS_S__param_2];
	ld.u64 	%rd1, [%rd4+8];
	ld.u64 	%rd2, [%rd4];
	setp.eq.s64	%p1, %rd2, %rd1;
	mov.u32 	%r10, 1;
	@%p1 bra 	BB6_4;

	setp.eq.s64	%p2, %rd2, 0;
	setp.eq.s64	%p3, %rd1, 0;
	or.pred  	%p4, %p2, %p3;
	mov.u32 	%r10, 0;
	@%p4 bra 	BB6_4;

	ld.u32 	%r7, [%rd1];
	ld.u32 	%r1, [%rd2];
	setp.ne.s32	%p5, %r1, %r7;
	@%p5 bra 	BB6_4;

	add.s64 	%rd5, %rd2, 4;
	add.s64 	%rd6, %rd1, 4;
	shl.b32 	%r8, %r1, 1;
	cvt.u64.u32	%rd7, %r8;
	// Callseq Start 5
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd5;
	.param .b64 param1;
	st.param.b64	[param1+0], %rd6;
	.param .b64 param2;
	st.param.b64	[param2+0], %rd7;
	.param .b32 retval0;
	call.uni (retval0), 
	_Z9gpumemcmpPKvS0_y, 
	(
	param0, 
	param1, 
	param2
	);
	ld.param.b32	%r9, [retval0+0];
	
	//{
	}// Callseq End 5
	setp.eq.s32	%p6, %r9, 0;
	selp.u32	%r10, 1, 0, %p6;

BB6_4:
	st.u32 	[%rd3], %r10;
	mov.u64 	%rd8, 0;
	st.param.b64	[func_retval0+0], %rd8;
	ret;
}

	// .globl	_Z25System_String_GetHashCodePhS_S_
.visible .func  (.param .b64 func_retval0) _Z25System_String_GetHashCodePhS_S_(
	.param .b64 _Z25System_String_GetHashCodePhS_S__param_0,
	.param .b64 _Z25System_String_GetHashCodePhS_S__param_1,
	.param .b64 _Z25System_String_GetHashCodePhS_S__param_2
)
{
	.reg .pred 	%p<8>;
	.reg .b32 	%r<49>;
	.reg .b64 	%rd<37>;


	ld.param.u64 	%rd22, [_Z25System_String_GetHashCodePhS_S__param_0];
	ld.param.u64 	%rd23, [_Z25System_String_GetHashCodePhS_S__param_2];
	add.s64 	%rd36, %rd22, 4;
	ld.u32 	%r13, [%rd22];
	cvt.u64.u32	%rd4, %r13;
	mul.wide.u32 	%rd24, %r13, 2;
	add.s64 	%rd25, %rd24, %rd36;
	add.s64 	%rd5, %rd25, -2;
	mov.u32 	%r47, 0;
	setp.ge.u64	%p1, %rd36, %rd5;
	@%p1 bra 	BB7_10;

	shl.b64 	%rd27, %rd4, 1;
	add.s64 	%rd28, %rd27, -3;
	shr.u64 	%rd29, %rd28, 2;
	add.s64 	%rd7, %rd29, 1;
	and.b64  	%rd8, %rd7, 3;
	setp.eq.s64	%p2, %rd8, 0;
	mov.u64 	%rd34, 0;
	mov.u32 	%r47, 0;
	@%p2 bra 	BB7_7;

	setp.eq.s64	%p3, %rd8, 1;
	mov.u32 	%r44, 0;
	@%p3 bra 	BB7_6;

	setp.eq.s64	%p4, %rd8, 2;
	mov.u32 	%r43, 0;
	@%p4 bra 	BB7_5;

	ld.u16 	%r17, [%rd22+4];
	ld.u16 	%r18, [%rd22+6];
	mad.lo.s32 	%r19, %r17, 31, %r18;
	add.s64 	%rd36, %rd22, 8;
	mul.lo.s32 	%r43, %r19, 31;

BB7_5:
	ld.u16 	%r20, [%rd36];
	add.s32 	%r21, %r20, %r43;
	ld.u16 	%r22, [%rd36+2];
	mad.lo.s32 	%r23, %r21, 31, %r22;
	add.s64 	%rd36, %rd36, 4;
	mul.lo.s32 	%r44, %r23, 31;

BB7_6:
	ld.u16 	%r24, [%rd36];
	add.s32 	%r25, %r24, %r44;
	ld.u16 	%r26, [%rd36+2];
	mad.lo.s32 	%r47, %r25, 31, %r26;
	add.s64 	%rd36, %rd36, 4;
	mov.u64 	%rd34, %rd36;

BB7_7:
	setp.lt.u64	%p5, %rd7, 4;
	@%p5 bra 	BB7_8;
	bra.uni 	BB7_9;

BB7_8:
	mov.u64 	%rd36, %rd34;
	bra.uni 	BB7_10;

BB7_9:
	ld.u16 	%r27, [%rd36];
	mad.lo.s32 	%r28, %r47, 31, %r27;
	ld.u16 	%r29, [%rd36+2];
	mad.lo.s32 	%r30, %r28, 31, %r29;
	ld.u16 	%r31, [%rd36+4];
	mad.lo.s32 	%r32, %r30, 31, %r31;
	ld.u16 	%r33, [%rd36+6];
	mad.lo.s32 	%r34, %r32, 31, %r33;
	ld.u16 	%r35, [%rd36+8];
	mad.lo.s32 	%r36, %r34, 31, %r35;
	ld.u16 	%r37, [%rd36+10];
	mad.lo.s32 	%r38, %r36, 31, %r37;
	ld.u16 	%r39, [%rd36+12];
	mad.lo.s32 	%r40, %r38, 31, %r39;
	ld.u16 	%r41, [%rd36+14];
	mad.lo.s32 	%r47, %r40, 31, %r41;
	add.s64 	%rd36, %rd36, 16;
	setp.lt.u64	%p6, %rd36, %rd5;
	@%p6 bra 	BB7_9;

BB7_10:
	setp.gt.u64	%p7, %rd36, %rd5;
	@%p7 bra 	BB7_12;

	ld.u16 	%r42, [%rd36];
	mad.lo.s32 	%r47, %r47, 31, %r42;

BB7_12:
	st.u32 	[%rd23], %r47;
	mov.u64 	%rd30, 0;
	st.param.b64	[func_retval0+0], %rd30;
	ret;
}

	// .globl	_Z29System_String_InternalReplacePhS_S_
.visible .func  (.param .b64 func_retval0) _Z29System_String_InternalReplacePhS_S_(
	.param .b64 _Z29System_String_InternalReplacePhS_S__param_0,
	.param .b64 _Z29System_String_InternalReplacePhS_S__param_1,
	.param .b64 _Z29System_String_InternalReplacePhS_S__param_2
)
{
	.reg .pred 	%p<16>;
	.reg .b16 	%rs<7>;
	.reg .b32 	%r<65>;
	.reg .b64 	%rd<35>;


	ld.param.u64 	%rd12, [_Z29System_String_InternalReplacePhS_S__param_0];
	ld.param.u64 	%rd13, [_Z29System_String_InternalReplacePhS_S__param_1];
	ld.param.u64 	%rd11, [_Z29System_String_InternalReplacePhS_S__param_2];
	ld.u64 	%rd1, [%rd13];
	ld.u64 	%rd2, [%rd13+8];
	ld.u32 	%r1, [%rd2];
	add.s64 	%rd3, %rd12, 4;
	ld.u32 	%r2, [%rd1];
	ld.u32 	%r3, [%rd12];
	sub.s32 	%r27, %r3, %r2;
	setp.eq.s32	%p1, %r27, -1;
	mov.u32 	%r52, 0;
	@%p1 bra 	BB8_7;

	mov.u32 	%r29, 0;
	mov.u32 	%r52, %r29;
	mov.u32 	%r53, %r29;

BB8_2:
	setp.eq.s32	%p2, %r2, 0;
	mov.u32 	%r54, %r29;
	@%p2 bra 	BB8_5;

BB8_3:
	add.s32 	%r31, %r54, %r53;
	mul.wide.u32 	%rd14, %r31, 2;
	add.s64 	%rd15, %rd3, %rd14;
	mul.wide.u32 	%rd16, %r54, 2;
	add.s64 	%rd17, %rd1, %rd16;
	ld.u16 	%rs1, [%rd17+4];
	ld.u16 	%rs2, [%rd15];
	add.s32 	%r54, %r54, 1;
	setp.ne.s16	%p3, %rs2, %rs1;
	@%p3 bra 	BB8_6;

	setp.lt.u32	%p4, %r54, %r2;
	@%p4 bra 	BB8_3;

BB8_5:
	add.s32 	%r32, %r2, %r53;
	add.s32 	%r53, %r32, -1;
	add.s32 	%r52, %r52, 1;

BB8_6:
	add.s32 	%r34, %r27, 1;
	add.s32 	%r53, %r53, 1;
	setp.lt.u32	%p5, %r53, %r34;
	@%p5 bra 	BB8_2;

BB8_7:
	sub.s32 	%r35, %r2, %r1;
	mul.lo.s32 	%r36, %r52, %r35;
	sub.s32 	%r37, %r3, %r36;
	shl.b32 	%r38, %r37, 1;
	add.s32 	%r39, %r38, 4;
	ld.global.u64 	%rd18, [types];
	ld.u64 	%rd19, [%rd18+72];
	// Callseq Start 6
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd19;
	.param .b32 param1;
	st.param.b32	[param1+0], %r39;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd4, [retval0+0];
	
	//{
	}// Callseq End 6
	st.u32 	[%rd4], %r37;
	setp.eq.s32	%p6, %r3, 0;
	@%p6 bra 	BB8_19;

	add.s64 	%rd5, %rd2, 4;
	mov.u32 	%r42, 0;
	mov.u32 	%r59, %r42;
	mov.u32 	%r60, %r42;

BB8_9:
	add.s32 	%r45, %r27, 1;
	setp.ge.u32	%p7, %r60, %r45;
	setp.eq.s32	%p8, %r2, 0;
	or.pred  	%p9, %p7, %p8;
	selp.b32	%r62, 0, %r62, %p7;
	mov.u32 	%r61, %r42;
	@%p9 bra 	BB8_13;

BB8_10:
	add.s32 	%r47, %r61, %r60;
	mul.wide.u32 	%rd20, %r47, 2;
	add.s64 	%rd21, %rd3, %rd20;
	mul.wide.u32 	%rd22, %r61, 2;
	add.s64 	%rd23, %rd1, %rd22;
	ld.u16 	%rs3, [%rd23+4];
	ld.u16 	%rs4, [%rd21];
	add.s32 	%r61, %r61, 1;
	setp.ne.s16	%p10, %rs4, %rs3;
	@%p10 bra 	BB8_11;

	setp.lt.u32	%p11, %r61, %r2;
	mov.u32 	%r62, 1;
	@%p11 bra 	BB8_10;
	bra.uni 	BB8_13;

BB8_11:
	mov.u32 	%r62, %r42;

BB8_13:
	setp.eq.s32	%p12, %r62, 0;
	@%p12 bra 	BB8_17;
	bra.uni 	BB8_14;

BB8_17:
	mul.wide.u32 	%rd29, %r60, 2;
	add.s64 	%rd30, %rd3, %rd29;
	ld.u16 	%rs6, [%rd30];
	mul.wide.u32 	%rd31, %r59, 2;
	add.s64 	%rd32, %rd4, %rd31;
	st.u16 	[%rd32+4], %rs6;
	mov.u32 	%r64, 1;
	bra.uni 	BB8_18;

BB8_14:
	mul.wide.u32 	%rd25, %r59, 2;
	add.s64 	%rd26, %rd4, %rd25;
	add.s64 	%rd7, %rd26, 4;
	shl.b32 	%r49, %r1, 1;
	cvt.u64.u32	%rd8, %r49;
	mov.u64 	%rd34, 0;
	setp.eq.s32	%p13, %r49, 0;
	@%p13 bra 	BB8_16;

BB8_15:
	add.s64 	%rd27, %rd5, %rd34;
	ld.u8 	%rs5, [%rd27];
	add.s64 	%rd28, %rd7, %rd34;
	st.u8 	[%rd28], %rs5;
	add.s64 	%rd34, %rd34, 1;
	setp.lt.u64	%p14, %rd34, %rd8;
	@%p14 bra 	BB8_15;

BB8_16:
	add.s32 	%r50, %r2, %r60;
	add.s32 	%r60, %r50, -1;
	mov.u32 	%r64, %r1;

BB8_18:
	add.s32 	%r59, %r64, %r59;
	add.s32 	%r60, %r60, 1;
	setp.lt.u32	%p15, %r60, %r3;
	@%p15 bra 	BB8_9;

BB8_19:
	st.u64 	[%rd11], %rd4;
	mov.u64 	%rd33, 0;
	st.param.b64	[func_retval0+0], %rd33;
	ret;
}

	// .globl	_Z29System_String_InternalIndexOfPhS_S_
.visible .func  (.param .b64 func_retval0) _Z29System_String_InternalIndexOfPhS_S_(
	.param .b64 _Z29System_String_InternalIndexOfPhS_S__param_0,
	.param .b64 _Z29System_String_InternalIndexOfPhS_S__param_1,
	.param .b64 _Z29System_String_InternalIndexOfPhS_S__param_2
)
{
	.reg .pred 	%p<5>;
	.reg .b16 	%rs<3>;
	.reg .b32 	%r<17>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd3, [_Z29System_String_InternalIndexOfPhS_S__param_0];
	ld.param.u64 	%rd4, [_Z29System_String_InternalIndexOfPhS_S__param_1];
	ld.param.u64 	%rd2, [_Z29System_String_InternalIndexOfPhS_S__param_2];
	ld.u16 	%rs1, [%rd4];
	ld.u32 	%r8, [%rd4+12];
	setp.eq.s32	%p1, %r8, 0;
	ld.u32 	%r9, [%rd4+4];
	ld.u32 	%r10, [%rd4+8];
	add.s32 	%r11, %r9, %r10;
	add.s32 	%r12, %r11, -1;
	selp.b32	%r13, -1, %r10, %p1;
	selp.b32	%r1, -1, 1, %p1;
	selp.b32	%r16, %r12, %r9, %p1;
	add.s32 	%r3, %r13, %r9;
	add.s64 	%rd1, %rd3, 4;
	setp.eq.s32	%p2, %r16, %r3;
	mov.u32 	%r7, -1;
	@%p2 bra 	BB9_3;

BB9_1:
	mul.wide.s32 	%rd5, %r16, 2;
	add.s64 	%rd6, %rd1, %rd5;
	ld.u16 	%rs2, [%rd6];
	setp.eq.s16	%p3, %rs2, %rs1;
	add.s32 	%r5, %r16, %r1;
	@%p3 bra 	BB9_4;

	setp.ne.s32	%p4, %r5, %r3;
	mov.u32 	%r16, %r5;
	@%p4 bra 	BB9_1;

BB9_3:
	mov.u32 	%r16, %r7;

BB9_4:
	st.u32 	[%rd2], %r16;
	mov.u64 	%rd7, 0;
	st.param.b64	[func_retval0+0], %rd7;
	ret;
}

	// .globl	_Z32System_String_InternalIndexOfAnyPhS_S_
.visible .func  (.param .b64 func_retval0) _Z32System_String_InternalIndexOfAnyPhS_S_(
	.param .b64 _Z32System_String_InternalIndexOfAnyPhS_S__param_0,
	.param .b64 _Z32System_String_InternalIndexOfAnyPhS_S__param_1,
	.param .b64 _Z32System_String_InternalIndexOfAnyPhS_S__param_2
)
{
	.reg .pred 	%p<6>;
	.reg .b16 	%rs<3>;
	.reg .b32 	%r<21>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd4, [_Z32System_String_InternalIndexOfAnyPhS_S__param_0];
	ld.param.u64 	%rd1, [_Z32System_String_InternalIndexOfAnyPhS_S__param_1];
	ld.param.u64 	%rd5, [_Z32System_String_InternalIndexOfAnyPhS_S__param_2];
	ld.u32 	%r11, [%rd1+12];
	setp.eq.s32	%p1, %r11, 0;
	ld.u32 	%r12, [%rd1+4];
	ld.u32 	%r13, [%rd1+8];
	add.s32 	%r14, %r12, %r13;
	add.s32 	%r15, %r14, -1;
	selp.b32	%r16, -1, %r13, %p1;
	selp.b32	%r18, %r15, %r12, %p1;
	add.s32 	%r2, %r16, %r12;
	setp.eq.s32	%p2, %r18, %r2;
	mov.u32 	%r20, -1;
	@%p2 bra 	BB10_7;

	ld.u64 	%rd6, [%rd1];
	ld.u32 	%r3, [%rd6];
	selp.b32	%r4, -1, 1, %p1;
	add.s64 	%rd2, %rd6, 4;
	add.s64 	%rd3, %rd4, 4;

BB10_2:
	mul.wide.s32 	%rd7, %r18, 2;
	add.s64 	%rd8, %rd3, %rd7;
	ld.u16 	%rs1, [%rd8];
	mov.u32 	%r19, %r3;

BB10_3:
	add.s32 	%r19, %r19, -1;
	setp.gt.s32	%p3, %r19, -1;
	@%p3 bra 	BB10_5;
	bra.uni 	BB10_4;

BB10_5:
	mul.wide.s32 	%rd9, %r19, 2;
	add.s64 	%rd10, %rd2, %rd9;
	ld.u16 	%rs2, [%rd10];
	setp.ne.s16	%p5, %rs1, %rs2;
	@%p5 bra 	BB10_3;
	bra.uni 	BB10_6;

BB10_4:
	add.s32 	%r18, %r18, %r4;
	setp.eq.s32	%p4, %r18, %r2;
	@%p4 bra 	BB10_7;
	bra.uni 	BB10_2;

BB10_6:
	mov.u32 	%r20, %r18;

BB10_7:
	st.u32 	[%rd5], %r20;
	mov.u64 	%rd11, 0;
	st.param.b64	[func_retval0+0], %rd11;
	ret;
}

	// .globl	_Z28SystemString_FromUserStringsP10tMetaData_j
.visible .func  (.param .b64 func_retval0) _Z28SystemString_FromUserStringsP10tMetaData_j(
	.param .b64 _Z28SystemString_FromUserStringsP10tMetaData_j_param_0,
	.param .b32 _Z28SystemString_FromUserStringsP10tMetaData_j_param_1
)
{
	.local .align 4 .b8 	__local_depot11[4];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<3>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<17>;


	mov.u64 	%rd16, __local_depot11;
	cvta.local.u64 	%SP, %rd16;
	ld.param.u64 	%rd8, [_Z28SystemString_FromUserStringsP10tMetaData_j_param_0];
	ld.param.u32 	%r1, [_Z28SystemString_FromUserStringsP10tMetaData_j_param_1];
	add.u64 	%rd9, %SP, 0;
	cvta.to.local.u64 	%rd10, %rd9;
	// Callseq Start 7
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd8;
	.param .b32 param1;
	st.param.b32	[param1+0], %r1;
	.param .b64 param2;
	st.param.b64	[param2+0], %rd9;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z22MetaData_GetUserStringP10tMetaData_jPj, 
	(
	param0, 
	param1, 
	param2
	);
	ld.param.b64	%rd3, [retval0+0];
	
	//{
	}// Callseq End 7
	ld.local.u32 	%r2, [%rd10];
	shr.u32 	%r3, %r2, 1;
	shl.b32 	%r4, %r3, 1;
	add.s32 	%r5, %r4, 4;
	ld.global.u64 	%rd11, [types];
	ld.u64 	%rd12, [%rd11+72];
	// Callseq Start 8
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32	[param1+0], %r5;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd1, [retval0+0];
	
	//{
	}// Callseq End 8
	st.u32 	[%rd1], %r3;
	add.s64 	%rd2, %rd1, 4;
	ld.local.u32 	%rd4, [%rd10];
	mov.u64 	%rd15, 0;
	setp.eq.s64	%p1, %rd4, 0;
	@%p1 bra 	BB11_2;

BB11_1:
	add.s64 	%rd13, %rd3, %rd15;
	ld.u8 	%rs1, [%rd13];
	add.s64 	%rd14, %rd2, %rd15;
	st.u8 	[%rd14], %rs1;
	add.s64 	%rd15, %rd15, 1;
	setp.lt.u64	%p2, %rd15, %rd4;
	@%p2 bra 	BB11_1;

BB11_2:
	st.param.b64	[func_retval0+0], %rd1;
	ret;
}

	// .globl	_Z29SystemString_FromCharPtrASCIIPc
.visible .func  (.param .b64 func_retval0) _Z29SystemString_FromCharPtrASCIIPc(
	.param .b64 _Z29SystemString_FromCharPtrASCIIPc_param_0
)
{
	.reg .pred 	%p<7>;
	.reg .b16 	%rs<8>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<25>;


	ld.param.u64 	%rd9, [_Z29SystemString_FromCharPtrASCIIPc_param_0];
	// Callseq Start 9
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd9;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z9gpustrlenPKc, 
	(
	param0
	);
	ld.param.b64	%rd10, [retval0+0];
	
	//{
	}// Callseq End 9
	cvt.u32.u64	%r1, %rd10;
	shl.b32 	%r10, %r1, 1;
	add.s32 	%r11, %r10, 4;
	ld.global.u64 	%rd11, [types];
	ld.u64 	%rd12, [%rd11+72];
	// Callseq Start 10
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32	[param1+0], %r11;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd1, [retval0+0];
	
	//{
	}// Callseq End 10
	st.u32 	[%rd1], %r1;
	setp.lt.s32	%p1, %r1, 1;
	@%p1 bra 	BB12_10;

	and.b32  	%r2, %r1, 3;
	setp.eq.s32	%p2, %r2, 0;
	mov.u32 	%r19, 0;
	@%p2 bra 	BB12_7;

	setp.eq.s32	%p3, %r2, 1;
	mov.u32 	%r17, 0;
	@%p3 bra 	BB12_6;

	setp.eq.s32	%p4, %r2, 2;
	mov.u32 	%r16, 0;
	@%p4 bra 	BB12_5;

	ld.s8 	%rs1, [%rd9];
	st.u16 	[%rd1+4], %rs1;
	mov.u32 	%r16, 1;

BB12_5:
	cvt.u64.u32	%rd13, %r16;
	add.s64 	%rd14, %rd9, %rd13;
	ld.s8 	%rs2, [%rd14];
	mul.wide.u32 	%rd15, %r16, 2;
	add.s64 	%rd16, %rd1, %rd15;
	st.u16 	[%rd16+4], %rs2;
	add.s32 	%r17, %r16, 1;

BB12_6:
	cvt.s64.s32	%rd17, %r17;
	add.s64 	%rd18, %rd9, %rd17;
	ld.s8 	%rs3, [%rd18];
	mul.wide.s32 	%rd19, %r17, 2;
	add.s64 	%rd20, %rd1, %rd19;
	st.u16 	[%rd20+4], %rs3;
	add.s32 	%r19, %r17, 1;

BB12_7:
	setp.lt.u32	%p5, %r1, 4;
	@%p5 bra 	BB12_10;

	cvt.s64.s32	%rd21, %r19;
	mul.wide.s32 	%rd22, %r19, 2;
	add.s64 	%rd24, %rd1, %rd22;
	add.s64 	%rd23, %rd9, %rd21;

BB12_9:
	ld.s8 	%rs4, [%rd23];
	st.u16 	[%rd24+4], %rs4;
	ld.s8 	%rs5, [%rd23+1];
	st.u16 	[%rd24+6], %rs5;
	ld.s8 	%rs6, [%rd23+2];
	add.s64 	%rd7, %rd24, 8;
	st.u16 	[%rd24+8], %rs6;
	ld.s8 	%rs7, [%rd23+3];
	st.u16 	[%rd24+10], %rs7;
	add.s64 	%rd23, %rd23, 4;
	add.s32 	%r19, %r19, 4;
	setp.lt.s32	%p6, %r19, %r1;
	mov.u64 	%rd24, %rd7;
	@%p6 bra 	BB12_9;

BB12_10:
	st.param.b64	[func_retval0+0], %rd1;
	ret;
}

	// .globl	_Z29SystemString_FromCharPtrUTF16Pt
.visible .func  (.param .b64 func_retval0) _Z29SystemString_FromCharPtrUTF16Pt(
	.param .b64 _Z29SystemString_FromCharPtrUTF16Pt_param_0
)
{
	.reg .pred 	%p<4>;
	.reg .b16 	%rs<3>;
	.reg .b32 	%r<7>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd7, [_Z29SystemString_FromCharPtrUTF16Pt_param_0];
	mov.u32 	%r6, 0;

BB13_1:
	mov.u32 	%r1, %r6;
	mul.wide.s32 	%rd8, %r1, 2;
	add.s64 	%rd9, %rd7, %rd8;
	ld.u16 	%rs1, [%rd9];
	add.s32 	%r6, %r1, 1;
	setp.ne.s16	%p1, %rs1, 0;
	@%p1 bra 	BB13_1;

	shl.b32 	%r4, %r1, 1;
	add.s32 	%r5, %r4, 4;
	ld.global.u64 	%rd11, [types];
	ld.u64 	%rd12, [%rd11+72];
	// Callseq Start 11
	{
	.reg .b32 temp_param_reg;
	// <end>}
	.param .b64 param0;
	st.param.b64	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32	[param1+0], %r5;
	.param .b64 retval0;
	call.uni (retval0), 
	_Z10Heap_AllocP12tMD_TypeDef_j, 
	(
	param0, 
	param1
	);
	ld.param.b64	%rd1, [retval0+0];
	
	//{
	}// Callseq End 11
	st.u32 	[%rd1], %r1;
	add.s64 	%rd2, %rd1, 4;
	cvt.s64.s32	%rd3, %r4;
	mov.u64 	%rd15, 0;
	setp.eq.s64	%p2, %rd3, 0;
	@%p2 bra 	BB13_4;

BB13_3:
	add.s64 	%rd13, %rd7, %rd15;
	ld.u8 	%rs2, [%rd13];
	add.s64 	%rd14, %rd2, %rd15;
	st.u8 	[%rd14], %rs2;
	add.s64 	%rd15, %rd15, 1;
	setp.lt.u64	%p3, %rd15, %rd3;
	@%p3 bra 	BB13_3;

BB13_4:
	st.param.b64	[func_retval0+0], %rd1;
	ret;
}

	// .globl	_Z22SystemString_GetStringPhPj
.visible .func  (.param .b64 func_retval0) _Z22SystemString_GetStringPhPj(
	.param .b64 _Z22SystemString_GetStringPhPj_param_0,
	.param .b64 _Z22SystemString_GetStringPhPj_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd1, [_Z22SystemString_GetStringPhPj_param_0];
	ld.param.u64 	%rd2, [_Z22SystemString_GetStringPhPj_param_1];
	setp.eq.s64	%p1, %rd2, 0;
	@%p1 bra 	BB14_2;

	ld.u32 	%r1, [%rd1];
	st.u32 	[%rd2], %r1;

BB14_2:
	add.s64 	%rd3, %rd1, 4;
	st.param.b64	[func_retval0+0], %rd3;
	ret;
}

	// .globl	_Z24SystemString_GetNumBytesPh
.visible .func  (.param .b32 func_retval0) _Z24SystemString_GetNumBytesPh(
	.param .b64 _Z24SystemString_GetNumBytesPh_param_0
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [_Z24SystemString_GetNumBytesPh_param_0];
	ld.u32 	%r1, [%rd1];
	shl.b32 	%r2, %r1, 1;
	add.s32 	%r3, %r2, 4;
	st.param.b32	[func_retval0+0], %r3;
	ret;
}


			";

    }
}
