
using Mono.Collections.Generic;

namespace Campy.Utils
{
    using Mono.Cecil;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    public static class MonoInterop
    {
        public static Mono.Cecil.TypeReference ToMonoTypeReference(this System.Type type)
        {
            String kernel_assembly_file_name = type.Assembly.Location;
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(kernel_assembly_file_name);
            var reference = md.Import(type);
            return reference;
        }

        public static Type ToSystemTypeAux(this TypeReference type)
        {
            return Type.GetType(type.GetReflectionName(), true);
        }

        private static string GetReflectionName(this TypeReference type)
        {
            var to_type = ToSystemType2(type);
            if (type.IsGenericInstance)
            {
                var genericInstance = (GenericInstanceType)type;
                var y = string.Format("{0}.{1}[{2}]", genericInstance.Namespace, type.Name, String.Join(",", genericInstance.GenericArguments.Select(p => p.GetReflectionName()).ToArray()));
                string ass = type.Module.Assembly.FullName;
                System.Console.WriteLine("1name is " + y);
                System.Console.WriteLine("ass = " + ass);
                return y;
            }
            if (type.DeclaringType != null)
            {
                string pre = type.DeclaringType.FullName;
                string suf = type.Name;
                string ass = type.Module.Assembly.FullName;
                string y = string.Format("{0}+{1},{2}", pre, suf, ass);
                System.Console.WriteLine("2name is " + y);
                return y;
            }
            {
                if (type.IsArray)
                {
                    var et = (type as ArrayType).ElementType;
                    string ss2 = et.GetReflectionName();
                    List<Int32>[] vvv = new List<int>[1];
                    string ss = vvv.GetType().FullName;
                    string y = string.Format("{0}[]", ss2);
                    System.Console.WriteLine("2name is " + y);
                    return y;
                }
                else
                {
                    string pre = type.FullName;
                    return pre;
                }
            }
        }

        public static System.Type ToSystemType(this Mono.Cecil.TypeReference tr)
        {
            var t1 = ToSystemTypeAux(tr);
            return t1;

            Type result = null;
            TypeReference element_type = null;

            // Find equivalent to type definition in Mono to System Reflection type.
            var td = tr.Resolve();

            // If the type isn't defined, can't do much about it. Just return null.
            if (td == null)
                return null;

            if (tr.IsArray)
            {
                // Get element type, and work on that first.
                var array_type = tr as ArrayType;
                element_type = array_type.ElementType;
                //element_type = tr.GetElementType();
                result = element_type.ToSystemType();
                // Create array type.
                if (result == null) return null;
                return result.MakeArrayType();
            }
            else
            {
                String ss = tr.Module.FullyQualifiedName;
                String assembly_location = td.Module.FullyQualifiedName;
                System.Reflection.Assembly assembly = System.Reflection.Assembly.LoadFile(assembly_location);
                List<Type> types = new List<Type>();
                StackQueue<Type> type_definitions = new StackQueue<Type>();
                StackQueue<Type> type_definitions_closure = new StackQueue<Type>();
                foreach (Type t in assembly.GetTypes())
                    type_definitions.Push(t);
                while (type_definitions.Count > 0)
                {
                    Type t = type_definitions.Pop();
                    if (Campy.Utils.Utility.IsSimilarType(t, td))
                        return t;
                    type_definitions_closure.Push(t);
                    foreach (Type ntd in t.GetNestedTypes())
                        type_definitions.Push(ntd);
                }
                foreach (Type t in type_definitions_closure)
                {
                    if (Campy.Utils.Utility.IsSimilarType(t, td))
                        return t;
                }
            }
            return null;
        }

        public static System.Type ToSystemType2(this Mono.Cecil.TypeReference tr)
        {
            Type result = null;
            TypeReference element_type = null;

            // Find equivalent to type definition in Mono to System Reflection type.
            var td = tr.Resolve();

            // If the type isn't defined, can't do much about it. Just return null.
            if (td == null)
                return null;

            if (tr.IsArray)
            {
                // Get element type, and work on that first.
                var array_type = tr as ArrayType;
                element_type = array_type.ElementType;
                //element_type = tr.GetElementType();
                result = element_type.ToSystemType();
                // Create array type.
                if (result == null) return null;
                return result.MakeArrayType();
            }
            else
            {
                String ss = tr.Module.FullyQualifiedName;
                String assembly_location = td.Module.FullyQualifiedName;
                System.Reflection.Assembly assembly = System.Reflection.Assembly.LoadFile(assembly_location);
                List<Type> types = new List<Type>();
                StackQueue<Type> type_definitions = new StackQueue<Type>();
                StackQueue<Type> type_definitions_closure = new StackQueue<Type>();
                foreach (Type t in assembly.GetTypes())
                    type_definitions.Push(t);
                while (type_definitions.Count > 0)
                {
                    Type t = type_definitions.Pop();
                    if (Campy.Utils.Utility.IsSimilarType(t, td))
                        return t;
                    type_definitions_closure.Push(t);
                    foreach (Type ntd in t.GetNestedTypes())
                        type_definitions.Push(ntd);
                }
                foreach (Type t in type_definitions_closure)
                {
                    if (Campy.Utils.Utility.IsSimilarType(t, td))
                        return t;
                }
            }
            return null;
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


        public static void Rewrite(this MethodReference method_reference)
        {
            Dictionary<TypeReference, System.Type> additional = new Dictionary<TypeReference, System.Type>();
            var mr_gp = method_reference.GenericParameters;
            var mr_dt = method_reference.DeclaringType;
            var mr_hgp = method_reference.HasGenericParameters;
            var mr_dt_hgp = method_reference.DeclaringType.HasGenericParameters;
            var mr_igi = method_reference.IsGenericInstance;
            var mr_dt_igi = method_reference.DeclaringType.IsGenericInstance;
            if (mr_igi)
            {
                GenericInstanceMethod i = method_reference as GenericInstanceMethod;
                var mr_hga = i.HasGenericArguments;
            }
            if (mr_dt_igi)
            {
                GenericInstanceType git = mr_dt as GenericInstanceType;
                var mr_dt_hga = git.HasGenericArguments;
                Collection<TypeReference> ga = git.GenericArguments;
                var e1 = git.ElementType;
                var e2 = git.GetElementType();
                Collection<GenericParameter> gg = e1.GenericParameters;
                // Map parameter to instantiated type.
                for (int i = 0; i < gg.Count; ++i)
                {
                    GenericParameter pp = gg[i];
                    TypeReference qq = ga[i];
                    TypeReference trrr = pp as TypeReference;
                    var system_type = qq
                        .ToSystemType();
                    if (system_type == null) throw new Exception("Failed to convert " + qq);
                    additional[pp] = system_type;
                }
            }
            return;
            //Mono.Collections.Generic.Collection<TypeReference> ga = null;
            //if (git != null)
            //{
            //    ga = git.GenericArguments;
            //    Mono.Collections.Generic.Collection<GenericParameter> gg = td.GenericParameters;
            //    // Map parameter to instantiated type.
            //    for (int i = 0; i < gg.Count; ++i)
            //    {
            //        var pp = gg[i];
            //        var qq = ga[i];
            //        TypeReference trrr = pp as TypeReference;
            //        var system_type = qq
            //            .ToSystemType();
            //        if (system_type == null) throw new Exception("Failed to convert " + qq);
            //        additional[pp] = system_type;
            //    }
            //}

        }
    }
}
