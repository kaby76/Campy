using System;
using System.Collections.Generic;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;
using Mono.Collections.Generic;
using System.Reflection;
using System.Diagnostics;
using Swigged.LLVM;


namespace Campy.Meta
{
    public class YOYO
    {

        public static List<GenericParameter> GetContainedGenericParameters(MemberReference type, List<GenericParameter> list = null)
        {
            if (list == null) list = new List<GenericParameter>();
            if (type.DeclaringType != null)
            {
                GetContainedGenericParameters(type.DeclaringType, list);
            }
            if (type is FieldReference field_reference)
            {
                GetContainedGenericParameters(field_reference.FieldType, list);
                return list;
            }
            if (type is FunctionPointerType function_pointer_type)
            {
                foreach (ParameterDefinition pd in function_pointer_type.Parameters)
                {
                    GetContainedGenericParameters(pd.ParameterType, list);
                }
                GetContainedGenericParameters(function_pointer_type.ReturnType, list);
                return list;
            }
            if (type is GenericInstanceMethod gim)
            {
                foreach (var a in gim.GenericArguments)
                    GetContainedGenericParameters(a, list);
                return list;
            }
            if (type is GenericInstanceType git)
            {
                foreach (var a in git.GenericArguments)
                    GetContainedGenericParameters(a, list);
                return list;
            }
            if (type is GenericParameter gp)
            {
                list.Add(gp);
                return list;
            }
            if (type is MethodReference method_reference)
            {
                foreach (ParameterDefinition pd in method_reference.Parameters)
                {
                    GetContainedGenericParameters(pd.ParameterType, list);
                }
                GetContainedGenericParameters(method_reference.ReturnType, list);
                return list;
            }
            if (type is MethodSpecification method_specification)
            {
                GetContainedGenericParameters(method_specification.ElementMethod, list);
                return list;
            }
            if (type is OptionalModifierType modifier_type)
            {
                GetContainedGenericParameters(modifier_type.ModifierType, list);
                return list;
            }
            if (type is RequiredModifierType required_modifier_type)
            {
                GetContainedGenericParameters(required_modifier_type.ModifierType, list);
                return list;
            }
            if (type is TypeSpecification type_specification)
            {
                GetContainedGenericParameters(type_specification.ElementType, list);
                return list;
            }
            return null;
        }
    }

    public static class METAHELPER
    {

        private static List<TypeDefinition> _cache = new List<TypeDefinition>();

        public static TypeReference Deresolve(this TypeReference type,
            TypeReference context,
            TypeReference argument_context)
        {
            if (!type.ContainsGenericParameter) return type;
            var gps = YOYO.GetContainedGenericParameters(type);
            bool method_defined = gps.Where(t => t.DeclaringMethod != null).Any();
            bool type_defined = gps.Where(t => t.DeclaringType != null).Any();
            if (method_defined && type_defined)
            {
            }
            if (type.IsArray)
            {
                var array_type = type as Mono.Cecil.ArrayType;
                var element_type = array_type.ElementType;
                var new_element_type = Deresolve(element_type, context, argument_context);
                if (element_type != new_element_type)
                {
                    var new_array_type = new ArrayType(new_element_type,
                        array_type.Rank);
                    type = new_array_type;
                }
            }
            else if (type as ByReferenceType != null)
            {
                var gp = type as ByReferenceType;
                var x = gp.ElementType;
                if (method_defined && argument_context as ByReferenceType != null)
                {
                    var ar = argument_context as ByReferenceType;
                    argument_context = ar.ElementType;
                }
                type = new ByReferenceType(Deresolve(x, context, argument_context));
            }
            else if (type as GenericInstanceType != null)
            {
                // For generic instance types, it could contain a generic parameter.
                // Substitute parameter if needed.
                var git = type as GenericInstanceType;
                var args = git.GenericArguments;
                var new_args = git.GenericArguments.ToArray();
                for (int i = 0; i < new_args.Length; ++i)
                {
                    var arg = args[i];
                    var new_arg = arg.Deresolve(context, argument_context);
                    git.GenericArguments[i] = new_arg;
                }
            }
            else if (type as GenericParameter != null)
            {
                var gp = type as GenericParameter;
                if (gp.DeclaringMethod != null)
                {
                    type = argument_context;
                }
                else if (gp.DeclaringType != null)
                {
                    var context_git = context as GenericInstanceType;
                    if (context_git == null)
                        throw new Exception("Can't do much with type that contains generics without context.");
                    var generic_arguments = context_git.GenericArguments;
                    var generic_parameters = type.GenericParameters;
                    if (generic_arguments.Count() > 0)
                    {
                        // Try to rewrite with parameter with argument.
                        var num = gp.Position;
                        var yo = generic_arguments.ToArray()[num];
                        type = yo;
                    }
                }
            }
            return type;
        }

        //public static MethodReference MakeMethodReference(this MethodDefinition method)
        //{
        //    var reference = new MethodReference(method.Name, method.ReturnType, method.DeclaringType);
        //    reference.MetadataToken = method.MetadataToken;
        //    reference.HasThis = method.HasThis;
        //    reference.ExplicitThis = method.ExplicitThis;
        //    reference.CallingConvention = method.CallingConvention;

        //    foreach (ParameterDefinition parameter in method.Parameters)
        //        reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));
        //    return reference;
        //}

        //public static MethodReference MakeMethodReference(this MethodReference method, TypeReference declaringType)
        //{
        //    var reference = new MethodReference(method.Name, method.ReturnType, declaringType);
        //    reference.MetadataToken = method.MetadataToken;
        //    reference.HasThis = method.HasThis;
        //    reference.ExplicitThis = method.ExplicitThis;
        //    reference.CallingConvention = method.CallingConvention;
        //    foreach (ParameterDefinition parameter in method.Parameters)
        //        reference.Parameters.Add(new ParameterDefinition(parameter.ParameterType));
        //    return reference;
        //}

        //public static TypeReference MakeGenericType(TypeReference type, params
        //    TypeReference[] arguments)
        //{
        //    if (type.GenericParameters.Count != arguments.Length)
        //        throw new ArgumentException();

        //    var instance = new GenericInstanceType(type);
        //    foreach (var argument in arguments)
        //        instance.GenericArguments.Add(argument);

        //    return instance;
        //}

        public static MethodReference Deresolve(
            this MethodReference self,
            TypeReference declaring_type,
            TypeReference[] arguments)
        {
            string a = declaring_type.FullName;
            if (a.Contains("List") || a.Contains("Dictionary"))
            {

            }

            if (declaring_type.ContainsGenericParameter)
                throw new Exception("Declaring type contains generic parameter.");

            var generic_type_of_declaring_type = declaring_type as GenericInstanceType;
            Collection<TypeReference> generic_arguments = generic_type_of_declaring_type?.GenericArguments;
            if (generic_arguments == null) generic_arguments = new Collection<TypeReference>();

            TypeReference parent = declaring_type;

            //if (self.DeclaringType.GenericParameters.Count != 0)
            //    parent = self.DeclaringType.MakeGenericInstanceType(args);

            var reference = new MethodReference(
                self.Name,
                self.ReturnType,
                parent);

            reference.HasThis = self.HasThis;
            reference.ExplicitThis = self.ExplicitThis;
            reference.CallingConvention = self.CallingConvention;
            reference.MetadataToken = self.MetadataToken;

            for (int i = 0; i < self.Parameters.Count(); ++i)
            {
                var parameter = self.Parameters[i];
                TypeReference argument = arguments == null ? null : arguments[i];
                var yo = Deresolve(parameter.ParameterType, declaring_type, argument);
                ParameterDefinition new_parameter_definition = new ParameterDefinition(yo);
                reference.Parameters.Add(new_parameter_definition);
            }

            foreach (var genericParam in self.GenericParameters)
                reference.GenericParameters.Add(new GenericParameter(genericParam.Name, reference));

            reference.ReturnType = reference.ReturnType.Deresolve(declaring_type, null);

            //var new_met = self.Module.ImportReference(reference);
            //if (new_met.ContainsGenericParameter)
            //{
            //    throw new Exception("Deresolve did not work.");
            //}
            var new_met = reference;
            return new_met;
        }

        public static MethodReference SubstituteMethod2(this MethodReference method_reference)
        {
            // Can't do anything if method isn't associated with a type.
            TypeReference declaring_type = method_reference.DeclaringType;
            if (declaring_type == null)
                return null;
            TypeDefinition declaring_type_resolved = declaring_type.Resolve();
            MethodDefinition method_definition_resolved = method_reference.Resolve();
            TypeReference substituted_declaring_type = declaring_type.RewriteMonoTypeReference();
            if (substituted_declaring_type != declaring_type)
                throw new Exception("Wrong place to rewrite function.");
            return method_reference;
        }

        public static MethodReference SubstituteMethod(
            this MethodReference method_reference,
            TypeReference parent,
            TypeReference[] method_arguments)
        {
            TypeReference declaring_type = method_reference.DeclaringType;
            if (declaring_type == null) return null;
            TypeReference substituted_declaring_type = declaring_type.RewriteMonoTypeReference();
            if (substituted_declaring_type.ContainsGenericParameter)
                substituted_declaring_type = substituted_declaring_type.Deresolve(parent, null);
            TypeDefinition substituted_declaring_type_definition = substituted_declaring_type.Resolve();
            if (substituted_declaring_type_definition == null) return null;
            var method_definition_resolved = method_reference.Resolve();
            if (method_definition_resolved == null) return null;
            var substituted_method_definition = substituted_declaring_type_definition.Methods
                .Where(m =>
                {
                    if (m.Name != method_definition_resolved.Name) return false;
                    if (m.Parameters.Count != method_definition_resolved.Parameters.Count) return false;
                    for (int i = 0; i < m.Parameters.Count; ++i)
                    {
                        // Should do a comparison of paramter types.
                        var p1 = m.Parameters[i].ParameterType;
                        var p2 = method_definition_resolved.Parameters[i].ParameterType;
                        if (p1.Name != p2.Name) return false;
                    }
                    return true;
                }).FirstOrDefault();
            if (substituted_method_definition == null)
                throw new Exception("Cannot find " + method_definition_resolved.FullName);
            var new_method_reference = substituted_method_definition.Deresolve(substituted_declaring_type, method_arguments);
            return new_method_reference;
        }


        ///////////////////////////////////////////////////////////////////////////////////////////
        // General routines associated with TypeReference accessors.
        ///////////////////////////////////////////////////////////////////////////////////////////
        // Resolving, correcting access functions

        public static TypeReference ResolveDeclaringType(this TypeReference tr)
        {
            var dt = tr.DeclaringType;
            return dt;
        }

        public static TypeReference ResolveDeclaringType(this FieldReference fr)
        {
            var dt = fr.DeclaringType;
            return dt;
        }

        public static TypeReference ResolveGetElementType(this TypeReference tr)
        {
            var et = tr.GetElementType();
            if (tr.IsGenericInstance)
            {
                var new_et = et.Deresolve(tr, null);
                et = new_et;
            }
            if (tr.IsArray)
            {
                var atr = tr as ArrayType;
                et = atr.ElementType;
            }
            return et;
        }

        public static List<FieldInfo> SafeFields(this Type type, BindingFlags binding_attr)
        {
            List<FieldInfo> safe_list = new List<FieldInfo>();
            var fields = type.GetFields(binding_attr);
            foreach (var field in fields)
            {
                if (field.FieldType.IsSubclassOf(typeof(MulticastDelegate)))
                    continue;
                safe_list.Add(field);
            }
            return safe_list;
        }

        public static Collection<FieldReference> MyGetFields(this TypeReference type)
        {
            // Get all fields, including parent chain, but not statics.
            Collection<FieldReference> result = new Collection<FieldReference>();
            Stack<TypeReference> chain = new Stack<TypeReference>();
            var p = type;
            while (p != null)
            {
                chain.Push(p);
                var bt = p.Resolve().BaseType;
                p = bt?.Deresolve(p, null);
            }
            while (chain.Any())
            {
                var q = chain.Pop();
                foreach (var f in q.ResolveFields())
                {
                    if (f.Resolve().IsStatic)
                        continue;
                    result.Add(f);
                }
            }
            return result;
        }

        public static Collection<FieldReference> ResolveFields(this TypeReference tr)
        {
            // Resolve throws away type information of fields, attributes, properties, methods.
            // This function performs a "resolve()" while retaining type information of generics.
            TypeDefinition resolved = tr.Resolve();
            Collection<FieldDefinition> resolved_fields = resolved.Fields;
            Collection<FieldReference> result = new Collection<FieldReference>();
            ModuleDefinition module = tr.Module;

            // Turn FieldDefinition back into FieldReference.
            if (tr.IsGenericInstance)
            {
                var gtr = tr as GenericInstanceType;
                var generic_arguments = gtr.GenericArguments;
                // For generics, convert any field defitions that use generic parameters into generic arguments.
                foreach (FieldDefinition field in resolved_fields)
                {
                    // Add in all fields, except Multicast delegate.
                    var field_type = field.FieldType;
                    if (field_type.FullName == typeof(MulticastDelegate).FullName)
                        continue;
                    var new_field_type = field_type.Deresolve(tr, null);
                    var new_field_reference = new FieldReference(field.Name,
                        new_field_type,
                        tr);
                    result.Add(new_field_reference);
                }
            }
            else
            {
                // For non-generics, just use the field definition--no need to create field reference from scratch.
                foreach (FieldDefinition f in resolved_fields)
                {
                    FieldReference fr = f;
                    if (fr.FullName == typeof(MulticastDelegate).FullName)
                        continue;
                    result.Add(fr);
                }
            }
            return result;
        }

        public static MethodReference FixGenericMethods(this MethodReference method_reference, MethodReference context)
        {
            if (method_reference == null) throw new Exception("Null method reference in FixGenericMethods");
            var result = method_reference;
            var declaring_type = method_reference.DeclaringType;
            result = method_reference.Deresolve(declaring_type, null);
            if (result.ContainsGenericParameter)
                throw new Exception("method reference contains generic " + result.FullName);
            return result;
        }

        public static Mono.Cecil.TypeReference ToMonoTypeReference(this System.Type type)
        {
            String kernel_assembly_file_name = type.Assembly.Location;
            Mono.Cecil.ModuleDefinition md = Campy.Meta.StickyReadMod.StickyReadModule(kernel_assembly_file_name);
            string ecma_name = type.FullName.Replace("+", "/");
            var reference = md.GetType(ecma_name);
            if (reference != null) return reference;
            var fallback = md.ImportReference(type);
            return fallback;
        }

        public static System.Type ToSystemType(this TypeReference type)
        {
            var to_type = Type.GetType(type.FullName);
            if (to_type == null) return null;
            string y = to_type.AssemblyQualifiedName;
            return Type.GetType(y, true);
        }

        public static TypeReference RewriteMonoTypeReference(this Mono.Cecil.TypeReference type)
        {
            var new_type = type.SubstituteMonoTypeReference();
            if (new_type == null) return type;
            return new_type;
        }

        public static TypeReference SubstituteMonoTypeReference(this Mono.Cecil.TypeReference type)
        {
            // No need to look if already done.
            string scope = type.Scope.Name;
            if (scope.Contains("corlib") && !scope.Contains("mscorlib")) return type;

            if (type as ArrayType != null)
            {
                var array_type = type as Mono.Cecil.ArrayType;
                var element_type = array_type.ElementType;
                var new_element_type = element_type.SubstituteMonoTypeReference();
                if (element_type != new_element_type)
                {
                    var new_array_type = new ArrayType(new_element_type, array_type.Rank);
                    type = new_array_type;
                }
                return type;
            }
            else if (type as ByReferenceType != null)
            {
                var gp = type as ByReferenceType;
                var x = gp.GetElementType();
                type = new ByReferenceType(x.SubstituteMonoTypeReference());
                return type;
            }
            else if (type as GenericInstanceType != null)
            {
                // For generic instance types, it could contain a generic parameter.
                // Substitute parameter if needed.
                var git = type as GenericInstanceType;
                var args = git.GenericArguments;
                var new_args = git.GenericArguments.ToArray();
                for (int i = 0; i < new_args.Length; ++i)
                {
                    var arg = args[i];
                    var new_arg = arg.SubstituteMonoTypeReference();
                    new_args[i] = new_arg;
                }
                var type_def = type.Resolve();
                var new_type = type_def.SubstituteMonoTypeReference();
                GenericInstanceType de = new_type.MakeGenericInstanceType(new_args);
                return de;
            }
            else if (type as GenericParameter != null)
            {
                return type;
            }
            else
            {
                RUNTIME.all_types.TryGetValue(type.Resolve().FullName, out TypeReference value);
                if (value == null) return type;
                return value;
            }
        }

        public static System.Reflection.MethodBase ToSystemMethodInfo(this Mono.Cecil.MethodDefinition md)
        {
            System.Reflection.MethodInfo result = null;
            String md_name = Campy.Utils.Utility.NormalizeMonoCecilName(md.FullName);
            // Get owning type.
            Mono.Cecil.TypeDefinition td = md.DeclaringType;
            Type t = td.ToSystemType();
            foreach (System.Reflection.MethodInfo mi in t.GetMethods(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.CreateInstance | System.Reflection.BindingFlags.Default))
            {
                String full_name = string.Format("{0} {1}.{2}({3})", mi.ReturnType.FullName, Campy.Utils.Utility.RemoveGenericParameters(mi.ReflectedType), mi.Name, string.Join(",", mi.GetParameters().Select(o => string.Format("{0}", o.ParameterType)).ToArray()));
                full_name = Campy.Utils.Utility.NormalizeSystemReflectionName(full_name);
                if (md_name.Contains(full_name))
                    return mi;
            }
            foreach (System.Reflection.ConstructorInfo mi in t.GetConstructors(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.CreateInstance | System.Reflection.BindingFlags.Default))
            {
                String full_name = string.Format("{0}.{1}({2})", Campy.Utils.Utility.RemoveGenericParameters(mi.ReflectedType), mi.Name, string.Join(",", mi.GetParameters().Select(o => string.Format("{0}", o.ParameterType)).ToArray()));
                full_name = Campy.Utils.Utility.NormalizeSystemReflectionName(full_name);
                if (md_name.Contains(full_name))
                    return mi;
            }
            Debug.Assert(result != null);
            return result;
        }

        public static bool IsStruct(this System.Type t)
        {
            return t.IsValueType && !t.IsPrimitive && !t.IsEnum;
        }

        public static bool IsStruct(this Mono.Cecil.TypeReference t)
        {
            return t.IsValueType && !t.IsPrimitive;
        }

        public static bool IsUnsigned(this Mono.Cecil.TypeReference t)
        {
            bool result =
                t.FullName == "System.Int16"
                || t.FullName == "System.Int32"
                || t.FullName == "System.Int64"
                || t.FullName == "System.SByte";
            return !result;
        }

        public static bool IsSubclassOf(this TypeReference t, TypeReference b)
        {
            var r = t.Resolve();
            if (r == null) return false;
            while (r != null)
            {
                if (r.FullName == b.FullName) return true;
                var r2 = r.BaseType;
                if (r2 != null) r = r2.Resolve();
                else r = null;
            }
            return false;
        }

        public static bool IsReferenceType(this Mono.Cecil.TypeReference t)
        {
            return !t.IsValueType;
        }

        public static TypeReference FromGenericParameterToTypeReference(TypeReference type_reference_of_parameter, GenericInstanceType git)
        {
            if (git == null)
                return type_reference_of_parameter;
            var genericArguments = git.GenericArguments;
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

        public static TypeRef ToLlvmTypeRef(this Mono.Cecil.ParameterReference p, MethodReference method)
        {
            TypeReference type_reference_of_parameter = p.ParameterType;

            if (method.DeclaringType.IsGenericInstance && method.ContainsGenericParameter)
            {
                var git = method.DeclaringType as GenericInstanceType;
                type_reference_of_parameter = FromGenericParameterToTypeReference(
                    type_reference_of_parameter, git);
            }

            var _cil_type = type_reference_of_parameter;
            var _verification_type = InitVerificationType(_cil_type);
            var _stack_verification_type = InitStackVerificationType(_verification_type, _cil_type);
            var _intermediate_type_ref = _stack_verification_type.ToTypeRef();

            return _intermediate_type_ref;
        }

        public static Mono.Cecil.TypeReference InitVerificationType(Mono.Cecil.TypeReference _cil_type)
        {
            // Roughly encoding table on page 311.
            if (_cil_type.FullName == typeof(sbyte).ToMonoTypeReference().FullName)
                return typeof(sbyte).ToMonoTypeReference(); // There is no "int8" in C#.
            else if (_cil_type.FullName == typeof(byte).ToMonoTypeReference().FullName)
                return typeof(sbyte).ToMonoTypeReference();
            else if (_cil_type.FullName == typeof(bool).ToMonoTypeReference().FullName)
                return typeof(sbyte).ToMonoTypeReference();

            else if (_cil_type.FullName == typeof(short).ToMonoTypeReference().FullName)
                return typeof(short).ToMonoTypeReference();
            else if (_cil_type.FullName == typeof(ushort).ToMonoTypeReference().FullName)
                return typeof(short).ToMonoTypeReference();
            else if (_cil_type.FullName == typeof(char).ToMonoTypeReference().FullName)
                return typeof(short).ToMonoTypeReference();

            else if (_cil_type.FullName == typeof(int).ToMonoTypeReference().FullName)
                return typeof(int).ToMonoTypeReference();
            else if (_cil_type.FullName == typeof(uint).ToMonoTypeReference().FullName)
                return typeof(int).ToMonoTypeReference();

            else if (_cil_type.FullName == typeof(long).ToMonoTypeReference().FullName)
                return typeof(long).ToMonoTypeReference();
            else if (_cil_type.FullName == typeof(ulong).ToMonoTypeReference().FullName)
                return typeof(long).ToMonoTypeReference();

            else if (_cil_type.FullName == typeof(float).ToMonoTypeReference().FullName)
                return typeof(float).ToMonoTypeReference();

            else if (_cil_type.FullName == typeof(double).ToMonoTypeReference().FullName)
                return typeof(double).ToMonoTypeReference();

            else
                return _cil_type;
        }

        public static Mono.Cecil.TypeReference InitStackVerificationType(Mono.Cecil.TypeReference _verification_type, Mono.Cecil.TypeReference _cil_type)
        {
            if (_verification_type.FullName == typeof(sbyte).ToMonoTypeReference().FullName)
                return typeof(int).ToMonoTypeReference();
            else if (_verification_type.FullName == typeof(short).ToMonoTypeReference().FullName)
                return typeof(int).ToMonoTypeReference();
            else if (_verification_type.FullName == typeof(int).ToMonoTypeReference().FullName)
                return typeof(int).ToMonoTypeReference();
            else if (_verification_type.FullName == typeof(long).ToMonoTypeReference().FullName)
                return typeof(long).ToMonoTypeReference();
            else if (_verification_type.FullName == typeof(float).ToMonoTypeReference().FullName)
                return typeof(float).ToMonoTypeReference();
            else if (_verification_type.FullName == typeof(double).ToMonoTypeReference().FullName)
                return typeof(double).ToMonoTypeReference();
            else if (_verification_type.FullName == typeof(bool).ToMonoTypeReference().FullName)
                return typeof(int).ToMonoTypeReference();
            else
                return _cil_type;
        }

        internal static Dictionary<TypeReference, TypeRef> previous_llvm_types_created_global = new Dictionary<TypeReference, TypeRef>();
        public static Dictionary<string, string> _rename_to_legal_llvm_name_cache = new Dictionary<string, string>();
        internal static Dictionary<TypeReference, TypeRef> basic_llvm_types_created = new Dictionary<TypeReference, TypeRef>();
        private static int _nn_id = 0;
        private static bool init = false;

        public static TypeRef ToTypeRef(
            this TypeReference tr)
        {
            if (!init)
            {
                // For basic value types, they can also appear as
                // reference types. We need to distinguish between the two.

                basic_llvm_types_created.Add(
                    typeof(sbyte).ToMonoTypeReference(),
                    LLVM.Int8Type());

                basic_llvm_types_created.Add(
                    typeof(byte).ToMonoTypeReference(),
                    LLVM.Int8Type());

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
                    LLVM.Int8Type()); // Asking for trouble if one tries to map directly to 1 bit.

                basic_llvm_types_created.Add(
                    typeof(char).ToMonoTypeReference(),
                    LLVM.Int16Type());

                basic_llvm_types_created.Add(
                    typeof(void).ToMonoTypeReference(),
                    LLVM.VoidType());

                basic_llvm_types_created.Add(
                    typeof(Mono.Cecil.TypeDefinition).ToMonoTypeReference(),
                    LLVM.PointerType(LLVM.VoidType(), 0));

                basic_llvm_types_created.Add(
                    typeof(System.Type).ToMonoTypeReference(),
                    LLVM.PointerType(LLVM.VoidType(), 0));

                init = true;
            }

            // Search for type if already converted. Note, there are several caches to search, each
            // containing types with different properties.
            // Also, NB: we use full name for the conversion, as types can be similarly named but within
            // different owning classes.
            foreach (var kv in basic_llvm_types_created)
            {
                if (kv.Key.FullName == tr.FullName)
                {
                    return kv.Value;
                }
            }
            foreach (var kv in previous_llvm_types_created_global)
            {
                if (kv.Key.FullName == tr.FullName)
                    return kv.Value;
            }
            foreach (var kv in previous_llvm_types_created_global)
            {
                if (kv.Key.FullName == tr.FullName)
                    return kv.Value;
            }

            tr = tr.RewriteMonoTypeReference();

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
                GenericInstanceType git = tr as GenericInstanceType;
                TypeDefinition gtd = tr as TypeDefinition;

                if (is_reference)
                {
                    // Convert the base type first.
                    var base_type = ToTypeRef(td);
                    // Add in pointer to type.
                    TypeRef p = LLVM.PointerType(base_type, 0);
                    return p;
                }

                // System.Array is not considered an "array", rather a "class". So, we need to handle
                // this type.
                if (tr.FullName == "System.Array")
                {
                    // Create a basic int[] and call it the day.
                    var original_tr = tr;

                    tr = typeof(int[]).ToMonoTypeReference();
                    var p = tr.ToTypeRef();
                    previous_llvm_types_created_global.Add(original_tr, p);
                    return p;
                }
                else if (tr.IsArray)
                {
                    // Note: mono_type_reference.GetElementType() is COMPLETELY WRONG! It does not function the same
                    // as system_type.GetElementType(). Use ArrayType.ElementType!
                    var array_type = tr as ArrayType;
                    var element_type = array_type.ElementType;
                    // ContextRef c = LLVM.ContextCreate();
                    ContextRef c = LLVM.GetGlobalContext();
                    string type_name = RenameToLegalLLVMName(tr.ToString());
                    TypeRef s = LLVM.StructCreateNamed(c, type_name);
                    TypeRef p = LLVM.PointerType(s, 0);
                    previous_llvm_types_created_global.Add(tr, p);
                    var e = ToTypeRef(element_type);
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
                    throw new Exception("Cannot convert " + tr.Name);
                }
                else if (td != null && td.IsEnum)
                {
                    // Enums are any underlying type, e.g., one of { bool, char, int8,
                    // unsigned int8, int16, unsigned int16, int32, unsigned int32, int64, unsigned int64, native int,
                    // unsigned native int }.
                    var bas = td.BaseType;
                    var fields = td.Fields;
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
                    var va = ToTypeRef(field_type);
                    return va;
                }
                else if (td != null && (td.IsClass || td.IsInterface || td.IsValueType))
                {
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
                        }
                    }

                    // Create a struct/class type.
                    ContextRef c = LLVM.GetGlobalContext();
                    string llvm_name = RenameToLegalLLVMName(tr.ToString());
                    TypeRef s = LLVM.StructCreateNamed(c, llvm_name);
                    var tr_bcltype = RUNTIME.MonoBclMap_GetBcl(tr);

                    TypeRef p;
		            if (td.IsValueType)
		            {
			            // Structs are implemented as value types, but if this type is a pointer,
			            // then return one.
			            if (is_pointer) p = LLVM.PointerType(s, 0);
			            else p = s;
		            }
		            else
		            {
			            // Classes are always implemented as pointers.
			            p = LLVM.PointerType(s, 0);
		            }
                    previous_llvm_types_created_global.Add(tr, p);

                    // Create array of typerefs as argument to StructSetBody below.
                    // Note, tr is correct type, but tr.Resolve of a generic type turns the type
                    // into an uninstantiated generic type. E.g., List<int> contains a generic T[] containing the
                    // data. T could be a struct/value type, or T could be a class.

                    List<TypeRef> list = new List<TypeRef>();
                    var fieldso = td.Fields;
                    var myfields = tr.MyGetFields();
//                    var fields = tr.ResolveFields();
                    int current_offset = 0;
                    foreach (var field in myfields)
                    {
                        Mono.Cecil.FieldAttributes attr = field.Resolve().Attributes;
                        if ((attr & Mono.Cecil.FieldAttributes.Static) != 0)
                            continue;

                        TypeReference field_type = field.FieldType;
                        TypeReference instantiated_field_type = field.FieldType;

                        if (git != null)
                        {
                            var generic_args = git.GenericArguments;
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

                        var bcl_field = RUNTIME.BclFindFieldInTypeAll(tr_bcltype, field.Name);
                        int field_size = RUNTIME.BclGetFieldSize(bcl_field);
                        int field_offset = RUNTIME.BclGetFieldOffset(bcl_field);
                        int padding = field_offset - current_offset;
                        if (padding < 0) throw new Exception("Fields out of order.");
                        if (padding != 0)
                        {
                            // Add in bytes to effect padding.
                            for (int j = 0; j < padding; ++j)
                                list.Add(LLVM.Int8Type());
                        }
                        var field_converted_type = ToTypeRef(instantiated_field_type);
                        list.Add(field_converted_type);
                        current_offset = field_offset + field_size;
                    }
                    LLVM.StructSetBody(s, list.ToArray(), true);
                    var xyz = LLVM.GetDataLayoutStr(RUNTIME.global_llvm_module);

                    if (Utils.Options.IsOn("name_trace"))
                    {
                        System.Console.WriteLine();
                        System.Console.WriteLine("TypeRef definition " + tr.FullName);
                        System.Console.WriteLine(" == " + s.ToString());
                    }

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

        public static TypeReference InstantiateGeneric(this TypeReference type, MethodReference mr)
        {
            if (!type.ContainsGenericParameter) return type;
            var declaring_type = mr.DeclaringType;
            if (declaring_type.IsGenericInstance)
            {
                var generic_type_of_declaring_type = declaring_type as GenericInstanceType;
                var generic_arguments = generic_type_of_declaring_type.GenericArguments;
                var new_arg = type.Deresolve(declaring_type, null);
                return new_arg;
            }
            return type;
        }

        public static VariableReference InstantiateGeneric(this VariableReference variable, MethodReference mr)
        {
            var type = variable.VariableType;
            if (type.IsGenericParameter)
            {
                // Go to basic block and get type.
                var declaring_type = mr.DeclaringType;
                if (declaring_type.IsGenericInstance)
                {
                    var generic_type_of_declaring_type = declaring_type as GenericInstanceType;
                    var generic_arguments = generic_type_of_declaring_type.GenericArguments;
                    var new_arg = type.Deresolve(declaring_type, null);
                    var new_var = new VariableDefinition(new_arg);
                    return new_var;
                }
            }
            return variable;
        }

        public static VariableDefinition InstantiateGeneric(this VariableDefinition variable, MethodReference mr)
        {
            var type = variable.VariableType;
            if (type.IsGenericParameter)
            {
                // Go to basic block and get type.
                var declaring_type = mr.DeclaringType;
                if (declaring_type.IsGenericInstance)
                {
                    var generic_type_of_declaring_type = declaring_type as GenericInstanceType;
                    var generic_arguments = generic_type_of_declaring_type.GenericArguments;
                    var new_arg = type.Deresolve(declaring_type, null);
                    var new_var = new VariableDefinition(new_arg);
                    //new_var.IsPinned = variable.IsPinned;
                    var a = variable.GetType().SafeFields(System.Reflection.BindingFlags.Instance
                                                         | System.Reflection.BindingFlags.NonPublic
                                                         | System.Reflection.BindingFlags.Public);
                    variable
                        .GetType()
                        .GetField("index", System.Reflection.BindingFlags.Instance
                                           | System.Reflection.BindingFlags.NonPublic
                                           | System.Reflection.BindingFlags.Public)
                        .SetValue(new_var, variable.Index);
                    return new_var;
                }
            }
            return variable;
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


        static int Alignment(System.Type type)
        {
            if (type.IsArray || type.IsClass)
                return 8;
            else if (type.IsEnum)
            {
                // Enums are any underlying type, e.g., one of { bool, char, int8,
                // unsigned int8, int16, unsigned int16, int32, unsigned int32, int64, unsigned int64, native int,
                // unsigned native int }.
                var bas = type.BaseType;
                var fields = type.GetFields();
                if (fields == null)
                    throw new Exception("Cannot convert " + type.Name);
                if (fields.Count() == 0)
                    throw new Exception("Cannot convert " + type.Name);
                var field = fields[0];
                if (field == null)
                    throw new Exception("Cannot convert " + type.Name);
                var field_type = field.FieldType;
                if (field_type == null)
                    throw new Exception("Cannot convert " + type.Name);
                return Alignment(field_type);
            }
            else if (type.IsStruct())
            {
                if (SizeOf(type) > 4)
                    return 8;
                else if (SizeOf(type) > 2)
                    return 4;
                else if (SizeOf(type) > 1)
                    return 2;
                else return 1;
            }
            else if (SizeOf(type) > 4)
                return 8;
            else if (SizeOf(type) > 2)
                return 4;
            else if (SizeOf(type) > 1)
                return 2;
            else return 1;
        }

        static int Padding(long offset, int alignment)
        {
            int padding = (alignment - (int)(offset % alignment)) % alignment;
            return padding;
        }

        static int SizeOf(System.Type type)
        {
            int result = 0;
            // For arrays, structs, and classes, elements and fields must
            // align 64-bit (or bigger) data on 64-bit boundaries.
            // We also assume the type has been converted to blittable.
            //if (type.FullName == "System.String")
            //{
            //    string str = (string)obj;
            //    var bytes = 0;
            //    bytes = 4 + str.Length * SizeOf(typeof(UInt16));
            //    return bytes;
            //}
            //else
            if (type.IsArray)
            {
                throw new Exception("Cannot determine size of array without data.");
            }
            else if (type.IsEnum)
            {
                // Enums are any underlying type, e.g., one of { bool, char, int8,
                // unsigned int8, int16, unsigned int16, int32, unsigned int32, int64, unsigned int64, native int,
                // unsigned native int }.
                var bas = type.BaseType;
                var fields = type.GetFields();
                if (fields == null)
                    throw new Exception("Cannot convert " + type.Name);
                if (fields.Count() == 0)
                    throw new Exception("Cannot convert " + type.Name);
                var field = fields[0];
                if (field == null)
                    throw new Exception("Cannot convert " + type.Name);
                var field_type = field.FieldType;
                if (field_type == null)
                    throw new Exception("Cannot convert " + type.Name);
                return SizeOf(field_type);
            }
            else if (type.IsStruct() || type.IsClass)
            {
                var fields = type.SafeFields(
                    System.Reflection.BindingFlags.Instance
                    | System.Reflection.BindingFlags.NonPublic
                    | System.Reflection.BindingFlags.Public
                //| System.Reflection.BindingFlags.Static
                );
                int size = 0;
                foreach (System.Reflection.FieldInfo field in fields)
                {
                    int field_size;
                    int alignment;
                    if (field.FieldType.IsArray || field.FieldType.IsClass)
                    {
                        field_size = System.Runtime.InteropServices.Marshal.SizeOf(typeof(IntPtr));
                        alignment = Alignment(field.FieldType);
                    }
                    else
                    {
                        field_size = SizeOf(field.FieldType);
                        alignment = Alignment(field.FieldType);
                    }
                    int padding = Padding(size, alignment);
                    size = size + padding + field_size;
                }
                result = size;
            }
            else
                result = System.Runtime.InteropServices.Marshal.SizeOf(type);
            return result;
        }
    }

    public static class TypeDefinitionEnumerator
    {
        public static IEnumerable<TypeDefinition> GetBoxes(this TypeDefinition t)
        {
            yield return t;

            if (t.HasNestedTypes)
            {
                foreach (TypeDefinition nested in t.NestedTypes)
                {
                    foreach (TypeDefinition x in nested.GetBoxes())
                        yield return x;
                }
            }
        }
    }

}
