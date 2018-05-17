
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
            var reference = md.ImportReference(type);
            return reference;
        }

        public static System.Type ToSystemType(this TypeReference type)
        {
            var to_type = Type.GetType(type.FullName);
            if (to_type == null) return null;
            string y = to_type.AssemblyQualifiedName;
            return Type.GetType(y, true);
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
