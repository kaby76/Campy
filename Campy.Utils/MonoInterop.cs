
namespace Campy.Utils
{
    using Campy.Utils;
    using Mono.Cecil;
    using Mono.Cecil.Rocks;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
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

        private static Mono.Cecil.TypeReference ToMonoTypeReferenceAux(this System.Type ty)
        {
            throw new Exception("Use ModuleDefinition.Import() instead.");

            var mono_type = FindSimilarMonoType(ty);
            if (mono_type == null)
            {
                if (ty.IsArray)
                {
                    var element_type = ty.GetElementType();
                    var result = FindSimilarMonoType(element_type);
                    // Create array type.
                    if (result == null) return null;
                    ArrayType art = result.MakeArrayType();
                    TypeReference type_reference = (TypeReference)art;
                    return type_reference;
                }
            }
            else if (mono_type.HasGenericParameters && !mono_type.IsGenericInstance)
            {
                if (ty.IsConstructedGenericType)
                {
                    var j2 = ty.GetGenericArguments();
                    List<TypeReference> list = new List<TypeReference>();
                    foreach (var j3 in j2)
                    {
                        TypeReference m = j3.ToMonoTypeReference();
                        list.Add(m);
                    }
                    var instantiated_type = mono_type.MakeGenericInstanceType(list.ToArray());
                    return instantiated_type;
                }
                else throw new Exception("Cannot convert.");
            }
            return mono_type;
        }

        public static System.Type ToSystemType(this Mono.Cecil.TypeReference tr)
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

        private static Mono.Cecil.TypeDefinition FindSimilarMonoType(System.Type ty)
        {
            // Get assembly name which encloses code for kernel.
            String kernel_assembly_file_name = ty.Assembly.Location;

            // Get directory containing the assembly.
            String full_path = Path.GetFullPath(kernel_assembly_file_name);
            full_path = Path.GetDirectoryName(full_path);

            // Decompile entire module using Mono.
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(kernel_assembly_file_name);

            // Examine all types, and all methods of types in order to find the lambda in Mono.Cecil.
            List<Type> types = new List<Type>();
            StackQueue<Mono.Cecil.TypeDefinition> type_definitions = new StackQueue<Mono.Cecil.TypeDefinition>();
            StackQueue<Mono.Cecil.TypeDefinition> type_definitions_closure = new StackQueue<Mono.Cecil.TypeDefinition>();
            foreach (Mono.Cecil.TypeDefinition td in md.Types)
            {
                type_definitions.Push(td);
            }
            while (type_definitions.Count > 0)
            {
                Mono.Cecil.TypeDefinition td = type_definitions.Pop();
                //System.Console.WriteLine("M Type = " + td);
                if (Campy.Utils.Utility.IsSimilarType(ty, td))
                    return td;
                type_definitions_closure.Push(td);
                foreach (Mono.Cecil.TypeDefinition ntd in td.NestedTypes)
                    type_definitions.Push(ntd);
            }
            foreach (Mono.Cecil.TypeDefinition td in type_definitions_closure)
            {
                // System.Console.WriteLine("M Type = " + td);
                if (Campy.Utils.Utility.IsSimilarType(ty, td))
                    return td;
            }
            return null;
        }

        private static Mono.Cecil.MethodDefinition ToMonoMethodDefinition(this System.Reflection.MethodBase mi)
        {
            throw new Exception("Use ModuleDefinition.Import() instead.");

            // Get assembly name which encloses code for kernel.
            String kernel_assembly_file_name = mi.DeclaringType.Assembly.Location;

            // Get directory containing the assembly.
            String full_path = Path.GetFullPath(kernel_assembly_file_name);
            full_path = Path.GetDirectoryName(full_path);

            String kernel_full_name = null;
            // Get full name of kernel, including normalization because they cannot be compared directly with Mono.Cecil names.
            if (mi as System.Reflection.MethodInfo != null)
            {
                System.Reflection.MethodInfo mik = mi as System.Reflection.MethodInfo;
                kernel_full_name = string.Format("{0} {1}.{2}({3})", mik.ReturnType.FullName, Campy.Utils.Utility.RemoveGenericParameters(mi.ReflectedType), mi.Name, string.Join(",", mi.GetParameters().Select(o => string.Format("{0}", o.ParameterType)).ToArray()));
            }
            else
                kernel_full_name = string.Format("{0}.{1}({2})", Campy.Utils.Utility.RemoveGenericParameters(mi.ReflectedType), mi.Name, string.Join(",", mi.GetParameters().Select(o => string.Format("{0}", o.ParameterType)).ToArray()));

            kernel_full_name = Campy.Utils.Utility.NormalizeSystemReflectionName(kernel_full_name);

            // Decompile entire module.
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(kernel_assembly_file_name);

            // Examine all types, and all methods of types in order to find the lambda in Mono.Cecil.
            List<Type> types = new List<Type>();
            StackQueue<Mono.Cecil.TypeDefinition> type_definitions = new StackQueue<Mono.Cecil.TypeDefinition>();
            StackQueue<Mono.Cecil.TypeDefinition> type_definitions_closure = new StackQueue<Mono.Cecil.TypeDefinition>();
            foreach (Mono.Cecil.TypeDefinition td in md.Types)
            {
                type_definitions.Push(td);
            }
            while (type_definitions.Count > 0)
            {
                Mono.Cecil.TypeDefinition ty = type_definitions.Pop();
                type_definitions_closure.Push(ty);
                foreach (Mono.Cecil.TypeDefinition ntd in ty.NestedTypes)
                    type_definitions.Push(ntd);
            }
            foreach (Mono.Cecil.TypeDefinition td in type_definitions_closure)
            {
                foreach (Mono.Cecil.MethodDefinition md2 in td.Methods)
                {
                    String md2_name = Campy.Utils.Utility.NormalizeMonoCecilName(md2.FullName);
                    if (md2_name.Contains(kernel_full_name))
                        return md2;
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
    }
}
