
using System.IO;

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

        public static System.Type ToSystemType(this TypeReference type)
        {
            var to_type = ToSystemType2(type);
            string y = to_type.AssemblyQualifiedName;
            return Type.GetType(y, true);
        }

        public static System.Type ToSystemType2(this Mono.Cecil.TypeReference tr)
        {
            return Type.GetType(tr.FullName);

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
            else if (tr.Resolve().Module.FullyQualifiedName.Contains("\\corlib.dll"))
            {
                throw new Exception();
            }
            else
            {
                String assembly_location = Path.GetFullPath(tr.Resolve().Module.FullyQualifiedName);
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


        public static Mono.Cecil.TypeReference SubstituteMonoTypeReference(this System.Type type, Mono.Cecil.ModuleDefinition md)
        {
            var reference = md.ImportReference(type);
            return reference;
        }

        public static Mono.Cecil.TypeDefinition SubstituteMonoTypeReference(this Mono.Cecil.TypeReference type, Mono.Cecil.ModuleDefinition md)
        {
            // ImportReference does not work as expected because the scope of the type found isn't in the module.
            foreach (var tt in md.Types)
            {
                if (type.Name == tt.Name && type.Namespace == tt.Namespace)
                {
                    return tt;
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
