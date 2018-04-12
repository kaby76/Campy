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
                            field_converted_type = field_converted_type;
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
                            field_converted_type = field_converted_type;
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

        public JITER()
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

            _importer = new IMPORTER();
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

            InitializeBCLRewrites();

            RUNTIME.Initialize();

        }

        private void InitializeBCLRewrites()
        {

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
                MethodReference method = basic_block._original_method_reference;
                var declaring_type = method.DeclaringType;

                {
                    // Let's first consider the parameter types to the function.
                    var parameters = method.Parameters;
                    for (int k = 0; k < parameters.Count; ++k)
                    {
                        ParameterDefinition par = parameters[k];
                        var type_to_consider = par.ParameterType;
                        type_to_consider = JITER.FromGenericParameterToTypeReference(type_to_consider, method.DeclaringType as GenericInstanceType);
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
                                    new_cfg_node._original_method_reference = basic_block._original_method_reference;
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
                                    new_cfg_node._original_method_reference = basic_block._original_method_reference;
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
                        //var type_to_consider_system_type = type_to_consider.ToSystemType();
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
                                    new_cfg_node._original_method_reference = basic_block._original_method_reference;
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

        // Method to denormalize information about the method this block is associated with,
        // placing that information into the block for easier access.
        private void ComputeBasicMethodProperties(IEnumerable<CFG.Vertex> basic_blocks_to_compile)
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

                MethodReference method = bb._original_method_reference;
                List<ParameterDefinition> parameters = method.Parameters.ToList();
                List<ParameterReference> instantiated_parameters = new List<ParameterReference>();

                ModuleRef mod = global_llvm_module; // LLVM.ModuleCreateWithName(mn);
                bb.Module = mod;

                uint count = (uint) bb.StackNumberOfArguments;
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
                            type_reference_of_parameter = FromGenericParameterToTypeReference(
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
                TYPE t_ret = new TYPE(FromGenericParameterToTypeReference(method.ReturnType, method.DeclaringType as GenericInstanceType));
                if (bb.HasStructReturnValue)
                {
                    t_ret = new TYPE(typeof(void).ToMonoTypeReference());
                }
                TypeRef ret_type = t_ret.IntermediateType;
                TypeRef met_type = LLVM.FunctionType(ret_type, param_types, false);
                ValueRef fun = LLVM.AddFunction(mod,
                    JITER.RenameToLegalLLVMName(JITER.MethodName(method)), met_type);
                BasicBlockRef entry = LLVM.AppendBasicBlock(fun, bb.Name.ToString());
                bb.BasicBlock = entry;
                bb.MethodValueRef = fun;
                var t_fun = LLVM.TypeOf(fun);
                var t_fun_con = LLVM.GetTypeContext(t_fun);
                var context = LLVM.GetModuleContext(JITER.global_llvm_module);
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
                    var context = LLVM.GetModuleContext(JITER.global_llvm_module);
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

                INST prev = null;
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
                List<CFG.Vertex> visited = new List<CFG.Vertex>();
                // Compute stack size for each basic block, processing nodes on work list
                // in DFT order.
                foreach (CFG.Vertex node in ordered_list)
                {
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
                            node.StackNumberOfLocals + node.StackNumberOfArguments;
                    }
                    else
                    {
                        int in_level = -1;
                        foreach (CFG.Vertex pred in _mcfg.PredecessorNodes(node))
                        {
                            // Do not consider interprocedural edges when computing stack size.
                            if (pred._original_method_reference != node._original_method_reference)
                                continue;
                            // If predecessor has not been visited, warn and do not consider.
                            if (pred.StackLevelOut == null)
                            {
                                continue;
                            }
                            // Warn if predecessor does not concur with another predecessor.
                            if (in_level != -1 && pred.StackLevelOut != node.StackLevelIn)
                                throw new Exception("Miscalculation in stack size.");
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
                    foreach (var inst in node.Instructions)
                    {
                        level_pre = level_after;
                        inst.ComputeStackLevel(this, ref level_after);
                        if (!(level_after >= node.StackNumberOfLocals + node.StackNumberOfArguments))
                            throw new Exception("Stack computation off. Internal error.");
                    }
                    node.StackLevelOut = level_after;
                    // Verify return node that it makes sense.
                    if (node.IsReturn && !unreachable.Contains(node))
                    {
                        if (node.StackLevelOut ==
                            node.StackNumberOfArguments
                            + node.StackNumberOfLocals
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
                        if (succ._original_method_reference != node._original_method_reference)
                            continue;
                        // If it's recursive, nothing more to do.
                        if (succ.IsEntry)
                            continue;
                        // If it's a return, nothing more to do also.
                        if (node.Instructions.Last() as i_ret != null)
                            continue;
                        // Nothing to update if no change.
                        if (node.StackLevelIn > level_after)
                        {
                            continue;
                        }
                        else if (node.StackLevelIn == level_after)
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
            string basic_block_id)
        {
            basic_blocks_to_compile = RemoveBasicBlocksAlreadyCompiled(basic_blocks_to_compile);

            ComputeBasicMethodProperties(basic_blocks_to_compile);

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

                    var state_in = new STATE(visited, bb, list_of_data_types_used);

                    if (Campy.Utils.Options.IsOn("state_computation_trace"))
                    {
                        System.Console.WriteLine("state in");
                        state_in.OutputTrace(new String(' ', 4));
                    }

                    bb.StateIn = state_in;
                    bb.StateOut = new STATE(state_in);

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
                        inst.DebuggerInfo(this);
                        inst = inst.Convert(this, bb.StateOut);
                        if (Campy.Utils.Options.IsOn("state_computation_trace"))
                            bb.StateOut.OutputTrace(new String(' ', 4));
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
                        var br = LLVM.BuildBr(bb.Builder, s.BasicBlock);
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
                    int size = llvm_node.StateIn._stack.Count;
                    for (int i = 0; i < size; ++i)
                    {
                        var count = llvm_node._graph.Predecessors(llvm_node).Count();
                        if (count < 2) continue;
                        if (Campy.Utils.Options.IsOn("state_computation_trace"))
                            System.Console.WriteLine("phi nodes need for "
                                + ob.Name + " for stack depth " + i);
                        ValueRef res;
                        res = llvm_node.StateIn._stack[i].V;
                        if (!llvm_node.StateIn._phi.Contains(res)) continue;
                        ValueRef[] phi_vals = new ValueRef[count];
                        for (int c = 0; c < count; ++c)
                        {
                            var p = llvm_node._graph.PredecessorEdges(llvm_node).ToList()[c].From;
                            if (Campy.Utils.Options.IsOn("state_computation_trace"))
                                System.Console.WriteLine("Adding in phi for pred state "
                                    + p.Name);
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
                        llvm_node.StateIn.OutputTrace(new String(' ', 4));
                        llvm_node.StateOut.OutputTrace(new String(' ', 4));
                    }
                }
            }

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
            //Singleton._converter.FindAllTargets(kernel));

            // Convert list into Mono data types.
            List<Mono.Cecil.TypeReference> list_of_mono_data_types_used = new List<TypeReference>();
            foreach (System.Type data_type_used in list_of_data_types_used)
            {
                list_of_mono_data_types_used.Add(
                    data_type_used.ToMonoTypeReference());
            }

            // In the same, in-order discovery of all methods, we're going to pass on
            // type information. As we spread the type info from basic block to successors,
            // copy the node with the type information associated with it if the type info
            // results in a different interpretation/compilation of the function.
            cs = InstantiateGenerics(
                cs, list_of_data_types_used, list_of_mono_data_types_used);

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

        private bool all_new = true;

        [global::System.Runtime.InteropServices.DllImport(@"C:\Users\kenne\Documents\Campy2\ConsoleApp4\bin\Debug\Campy.Runtime.Wrapper.dll", EntryPoint = "?InitTheBcl@@YAXPEAX_KH0@Z")]
        public static extern void InitTheBcl(System.IntPtr a1, long a2, int a3, System.IntPtr a4);

        [global::System.Runtime.InteropServices.DllImport(@"C:\Users\kenne\Documents\Campy2\ConsoleApp4\bin\Debug\Campy.Runtime.Wrapper.dll", EntryPoint = "?InitFileSystem@@YAXXZ")]
        public static extern void InitFileSystem();

        [global::System.Runtime.InteropServices.DllImport(@"C:\Users\kenne\Documents\Campy2\ConsoleApp4\bin\Debug\Campy.Runtime.Wrapper.dll", EntryPoint = "?GfsAddFile@@YAXPEAX0_K0@Z")]
        public static extern void GfsAddFile(System.IntPtr name, System.IntPtr file, long length, System.IntPtr result);

        [global::System.Runtime.InteropServices.DllImport(@"C:\Users\kenne\Documents\Campy2\ConsoleApp4\bin\Debug\Campy.Runtime.Wrapper.dll", EntryPoint = "?InitializeBCL1@@YAXXZ")]
        public static extern void InitializeBCL1();

        [global::System.Runtime.InteropServices.DllImport(@"C:\Users\kenne\Documents\Campy2\ConsoleApp4\bin\Debug\Campy.Runtime.Wrapper.dll", EntryPoint = "?InitializeBCL2@@YAXXZ")]
        public static extern void InitializeBCL2();

        public void InitBCL(CUmodule mod)
        {
            if (all_new)
            {
                if (!done_init)
                {
                    unsafe
                    {
                        BUFFERS buffers = new BUFFERS();
                        int the_size = 536870912;
                        IntPtr b = buffers.New(the_size);
                        RUNTIME.BclPtr = b;
                        int max_threads = 16;
                        IntPtr b2 = buffers.New(sizeof(int*));

                        InitTheBcl(b, the_size, max_threads, b2);
                    }

                    unsafe
                    {
                        InitFileSystem();

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
                        BUFFERS buffers = new BUFFERS();
                        IntPtr pointer1 = buffers.New(assem.Length + 1);
                        BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
                        var pointer4 = buffers.New(sizeof(int));

                        GfsAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);

                        InitializeBCL1();
                        InitializeBCL2();
                        done_init = true;
                    }
                }
            }
            else
            {
                if (!done_init)
                {
                    CUresult res;

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

                    CUjit_option[] op = new CUjit_option[0];
                    ulong[] op_values = new ulong[0];
                    var op_values_handle = GCHandle.Alloc(op_values, GCHandleType.Pinned);
                    var op_values_intptr = op_values_handle.AddrOfPinnedObject();

                    // Go to directory for Campy.
                    var dir = Path.GetDirectoryName(
                        Path.GetFullPath(System.Reflection.Assembly.GetEntryAssembly().Location));
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

                    CUmodule module = RUNTIME.InitializeModule(image);

                    // Initialize BCL.
                    Utils.CudaHelpers.CheckCudaError(Cuda.cuCtxGetLimit(out ulong pvalue, CUlimit.CU_LIMIT_STACK_SIZE));
                    Utils.CudaHelpers.CheckCudaError(
                        Cuda.cuCtxSetLimit(CUlimit.CU_LIMIT_STACK_SIZE, (uint) pvalue * 25));
                    if (Campy.Utils.Options.IsOn("jit_trace"))
                        System.Console.WriteLine("Stack size " + pvalue);

                    Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                        out Campy.Utils.CudaHelpers.dim3 tile_size,
                        out Campy.Utils.CudaHelpers.dim3 tiles);

                    // Set up malloc, required for everything else.
                    unsafe
                    {
                        BUFFERS buffers = new BUFFERS();
                        int the_size = 536870912;
                        IntPtr b = buffers.New(the_size);
                        RUNTIME.BclPtr = b;
                        int max_threads = 16;

                        // Set up parameters.
                        IntPtr parm1;
                        IntPtr parm2;
                        IntPtr parm3;
                        IntPtr parm4;
                        IntPtr[] x1 = new IntPtr[] {b};
                        GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                        parm1 = handle1.AddrOfPinnedObject();

                        IntPtr[] x2 = new IntPtr[] {new IntPtr(the_size)};
                        GCHandle handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                        parm2 = handle2.AddrOfPinnedObject();

                        IntPtr[] x3 = new IntPtr[] {new IntPtr(max_threads)};
                        GCHandle handle3 = GCHandle.Alloc(x3, GCHandleType.Pinned);
                        parm3 = handle3.AddrOfPinnedObject();

                        IntPtr b2 = buffers.New(sizeof(int*));
                        IntPtr[] x4 = new IntPtr[] {b2};
                        GCHandle handle4 = GCHandle.Alloc(x4, GCHandleType.Pinned);
                        parm4 = handle4.AddrOfPinnedObject();

                        IntPtr[] kp = new IntPtr[] {parm1, parm2, parm3, parm4};

                        CUfunction _Z22Initialize_BCL_GlobalsPvyiPP6_BCL_t =
                            RUNTIME._Z22Initialize_BCL_GlobalsPvyiPP6_BCL_t(module);
                        fixed (IntPtr* kernelParams = kp)
                        {
                            res = Cuda.cuLaunchKernel(
                                _Z22Initialize_BCL_GlobalsPvyiPP6_BCL_t,
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

                    // Set up a file system for GPU.
                    CUfunction _Z12Bcl_Gfs_initv = RUNTIME._Z12Bcl_Gfs_initv(module);
                    res = Cuda.cuLaunchKernel(
                        _Z12Bcl_Gfs_initv,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr) IntPtr.Zero,
                        (IntPtr) IntPtr.Zero
                    );
                    Utils.CudaHelpers.CheckCudaError(res);
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    Utils.CudaHelpers.CheckCudaError(res);

                    unsafe
                    {
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

                        // Set up parameters.
                        int count = 4;
                        IntPtr parm1; // Name of assembly.
                        IntPtr parm2; // Contents
                        IntPtr parm3; // Length
                        IntPtr parm4; // result

                        var ptrx = Marshal.StringToHGlobalAnsi(assem);
                        BUFFERS buffers = new BUFFERS();
                        IntPtr pointer1 = buffers.New(assem.Length + 1);
                        BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
                        IntPtr[] x1 = new IntPtr[] {pointer1};
                        GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                        parm1 = handle1.AddrOfPinnedObject();

                        parm2 = corlib_bytes_intptr;
                        IntPtr pointer2 = buffers.New((int) corlib_bytes_handle_len);
                        BUFFERS.Cp(pointer2, corlib_bytes_intptr, (int) corlib_bytes_handle_len);
                        IntPtr[] x2 = new IntPtr[] {pointer2};
                        GCHandle handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                        parm2 = handle2.AddrOfPinnedObject();

                        IntPtr[] x3 = new IntPtr[] {new IntPtr(corlib_bytes_handle_len)};
                        GCHandle handle3 = GCHandle.Alloc(x3, GCHandleType.Pinned);
                        parm3 = handle3.AddrOfPinnedObject();

                        var pointer4 = buffers.New(sizeof(long));
                        IntPtr[] x4 = new IntPtr[] {pointer4};
                        GCHandle handle4 = GCHandle.Alloc(x4, GCHandleType.Pinned);
                        parm4 = handle4.AddrOfPinnedObject();

                        IntPtr[] kp = new IntPtr[] {parm1, parm2, parm3, parm4};

                        CUfunction _Z16Bcl_Gfs_add_filePcS_yPi = RUNTIME._Z16Bcl_Gfs_add_filePcS_yPi(module);
                        fixed (IntPtr* kernelParams = kp)
                        {
                            res = Cuda.cuLaunchKernel(
                                _Z16Bcl_Gfs_add_filePcS_yPi,
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

                    CUfunction _Z15Initialize_BCL1v = RUNTIME._Z15Initialize_BCL1v(module);
                    res = Cuda.cuLaunchKernel(
                        _Z15Initialize_BCL1v,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr) IntPtr.Zero,
                        (IntPtr) IntPtr.Zero
                    );
                    Utils.CudaHelpers.CheckCudaError(res);
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    Utils.CudaHelpers.CheckCudaError(res);

                    CUfunction _Z15Initialize_BCL2v = RUNTIME._Z15Initialize_BCL2v(module);
                    res = Cuda.cuLaunchKernel(
                        _Z15Initialize_BCL2v,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr) IntPtr.Zero,
                        (IntPtr) IntPtr.Zero
                    );
                    Utils.CudaHelpers.CheckCudaError(res);
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    Utils.CudaHelpers.CheckCudaError(res);

                    // Now reset the stack size.
                    Utils.CudaHelpers.CheckCudaError(Cuda.cuCtxSetLimit(CUlimit.CU_LIMIT_STACK_SIZE, (uint) pvalue));

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
                    int count = 1;
                    IntPtr parm1;
                    IntPtr[] x1 = new IntPtr[] {RUNTIME.BclPtr};
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
