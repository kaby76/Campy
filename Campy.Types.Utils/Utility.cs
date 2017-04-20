using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SR=System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using Mono.Cecil;
using Campy.Utils;

namespace Campy.Types.Utils
{
    public class Utility
    {
        /// <summary>
        /// Data class used by CreateBlittableType in order to create a blittable type
        /// corresponding to a host type.
        /// </summary>
        class Data
        {
            public SR.AssemblyName assemblyName;
            public AssemblyBuilder ab;
            public ModuleBuilder mb;
            static int v = 1;

            public Data()
            {
                assemblyName = new SR.AssemblyName("DynamicAssembly" + v++);
                ab = AppDomain.CurrentDomain.DefineDynamicAssembly(
                    assemblyName,
                    AssemblyBuilderAccess.RunAndSave);
                mb = ab.DefineDynamicModule(assemblyName.Name, assemblyName.Name + ".dll");
            }
        }

        public static Type CreateBlittableType(Type hostType, bool declare_parent_chain, bool declare_flatten_structure)
        {
            try
            {
                String name;
                SR.TypeFilter tf;
                Type bbt = null;

                // Declare inheritance types.
                if (declare_parent_chain)
                {
                    // First, declare base type
                    Type bt = hostType.BaseType;
                    if (bt != null && !bt.FullName.Equals("System.Object"))
                    {
                        bbt = CreateBlittableType(bt, declare_parent_chain, declare_flatten_structure);
                    }
                }

                name = hostType.FullName;
                tf = new SR.TypeFilter((Type t, object o) =>
                {
                    return t.FullName == name;
                });

                // Find if blittable type for hostType was already performed.
                Data data = new Data();
                Type[] types = data.mb.FindTypes(tf, null);

                // If blittable type was not created, create one with all fields corresponding
                // to that in host, with special attention to arrays.
                if (types.Length == 0)
                {
                    if (hostType.IsArray)
                    {
                        // Recurse
                        Type elementType = CreateBlittableType(hostType.GetElementType(), declare_parent_chain, declare_flatten_structure);
                        object array_obj = Array.CreateInstance(elementType, 0);
                        Type array_type = array_obj.GetType();
                        TypeBuilder tb = null;
                        if (bbt != null)
                        {
                            tb = data.mb.DefineType(
                                array_type.Name,
                                SR.TypeAttributes.Public | SR.TypeAttributes.SequentialLayout
                                    | SR.TypeAttributes.Serializable, bbt);
                        }
                        else
                        {
                            tb = data.mb.DefineType(
                                array_type.Name,
                                SR.TypeAttributes.Public | SR.TypeAttributes.SequentialLayout
                                    | SR.TypeAttributes.Serializable);
                        }
                        return tb.CreateType();
                    }
                    else if (Campy.Types.Utils.ReflectionCecilInterop.IsStruct(hostType) || hostType.IsClass)
                    {
                        TypeBuilder tb = null;
                        if (bbt != null)
                        {
                            tb = data.mb.DefineType(
                                name,
                                SR.TypeAttributes.Public | SR.TypeAttributes.SequentialLayout
                                    | SR.TypeAttributes.Serializable, bbt);
                        }
                        else
                        {
                            tb = data.mb.DefineType(
                                name,
                                SR.TypeAttributes.Public | SR.TypeAttributes.SequentialLayout
                                    | SR.TypeAttributes.Serializable);
                        }
                        Type ht = hostType;
                        while (ht != null)
                        {
                            var fields = ht.GetFields(
                                SR.BindingFlags.Instance
                                | SR.BindingFlags.NonPublic
                                | SR.BindingFlags.Public
                                | SR.BindingFlags.Static);
                            var fields2 = ht.GetFields();
                            foreach (var field in fields)
                            {
                                if (field.FieldType.IsArray)
                                {
                                    // Convert byte, int, etc., in host type to pointer in blittable type.
                                    // With array, we need to also encode the length.
                                    tb.DefineField(field.Name, typeof(IntPtr), SR.FieldAttributes.Public);
                                    tb.DefineField(field.Name + "Len0", typeof(Int32), SR.FieldAttributes.Public);
                                }
                                else
                                {
                                    // For non-array type fields, just define the field as is.
                                    tb.DefineField(field.Name, field.FieldType, SR.FieldAttributes.Public);
                                }
                            }
                            if (declare_flatten_structure)
                                ht = ht.BaseType;
                            else
                                ht = null;
                        }
                        // Base type will be used.
                        return tb.CreateType();
                    }
                    else return null;
                }
                else
                    return types[0];
            }
            catch
            {
                return null;
            }
        }

        public static Type CreateBlittableTypeMono(Mono.Cecil.TypeReference hostType, bool declare_parent_chain)
        {
            try
            {
                Mono.Cecil.TypeDefinition td = hostType.Resolve();
                String name;
                SR.TypeFilter tf;

                // Declare parent chain since TypeBuilder works top down not bottom up.
                if (declare_parent_chain)
                {
                    name = hostType.FullName;
                    name = name.Replace('+', '.');
                    tf = new SR.TypeFilter((Type t, object o) =>
                    {
                        return t.FullName == name;
                    });
                }
                else
                {
                    name = hostType.Name;
                    tf = new SR.TypeFilter((Type t, object o) =>
                    {
                        return t.Name == name;
                    });
                }

                // Find if blittable type for hostType was already performed.
                Data data = new Data();
                Type[] types = data.mb.FindTypes(tf, null);

                // If blittable type was not created, create one with all fields corresponding
                // to that in host, with special attention to arrays.
                if (types.Length == 0)
                {
                    if (hostType.IsArray)
                    {
                        // Recurse
                        Type elementType = CreateBlittableTypeMono(hostType.GetElementType(), true);
                        object array_obj = Array.CreateInstance(elementType, 0);
                        Type array_type = array_obj.GetType();
                        TypeBuilder tb = null;
                        tb = data.mb.DefineType(
                            array_type.Name,
                            SR.TypeAttributes.Public | SR.TypeAttributes.Sealed | SR.TypeAttributes.SequentialLayout
                                | SR.TypeAttributes.Serializable, typeof(ValueType));
                        return tb.CreateType();
                    }
                    else if (Campy.Types.Utils.ReflectionCecilInterop.IsStruct(hostType) || !hostType.IsValueType)
                    {
                        TypeBuilder tb = null;
                        tb = data.mb.DefineType(
                            name,
                            SR.TypeAttributes.Public | SR.TypeAttributes.Sealed | SR.TypeAttributes.SequentialLayout
                                | SR.TypeAttributes.Serializable, typeof(ValueType));

                        var fields = td.Fields;
                        foreach (var field in fields)
                        {
                            if (field.FieldType.IsArray)
                            {
                                // Convert byte, int, etc., in host type to pointer in blittable type.
                                // With array, we need to also encode the length.
                                tb.DefineField(field.Name, typeof(IntPtr), SR.FieldAttributes.Public);
                                tb.DefineField(field.Name + "Len0", typeof(Int32), SR.FieldAttributes.Public);
                            }
                            else
                            {
                                // For non-array type fields, just define the field as is.
                                tb.DefineField(field.Name,
                                    Campy.Types.Utils.ReflectionCecilInterop.ConvertToSystemReflectionType(field.FieldType),
                                    SR.FieldAttributes.Public);
                            }
                        }
                        return tb.CreateType();
                    }
                    else return null;
                }
                else
                    return types[0];
            }
            catch
            {
                return null;
            }
        }

        public static void CopyToBlittableType(object from, ref object to)
        {
            Type f = from.GetType();
            SR.FieldInfo[] ffi = f.GetFields();
            Type t = to.GetType();
            SR.FieldInfo[] tfi = t.GetFields();
            foreach (SR.FieldInfo fi in ffi)
            {
                object field_value = fi.GetValue(from);
                String na = fi.Name;
                
                // Copy.
                var tfield = tfi.Where(k => k.Name == fi.Name).FirstOrDefault();
                if (tfield == null)
                    throw new ArgumentException("Field not found.");
                tfield.SetValue(to, field_value);
            }
        }

        public static void CopyFromBlittableType(object from, ref object to)
        {
            Type f = from.GetType();
            SR.FieldInfo[] ffi = f.GetFields();
            Type t = to.GetType();
            SR.FieldInfo[] tfi = t.GetFields();
            foreach (SR.FieldInfo fi in ffi)
            {
                object field_value = fi.GetValue(from);
                String na = fi.Name;

                // Copy.
                var tfield = tfi.Where(k => k.Name == fi.Name).FirstOrDefault();
                if (tfield == null)
                    throw new ArgumentException("Field not found.");
                tfield.SetValue(to, field_value);
            }
        }

        public static void CopyFromPtrToBlittable(IntPtr ptr, object blittable_object)
        {
            Marshal.PtrToStructure(ptr, blittable_object);
        }

        public static IntPtr CreateNativeArray(int length, int blittable_element_size)
        {
            IntPtr cpp_array = Marshal.AllocHGlobal(blittable_element_size * length);
            return cpp_array;
        }

        public static IntPtr CreateNativeArray(Array from, Type blittable_element_type)
        {
            int size_element = Marshal.SizeOf(blittable_element_type);
            IntPtr cpp_array = Marshal.AllocHGlobal(size_element * from.Length);
            return cpp_array;
        }

        public static IntPtr CopyToNativeArray(Array from, IntPtr cpp_array, Type blittable_element_type)
        {
            IntPtr byte_ptr = cpp_array;
            int size_element = Marshal.SizeOf(blittable_element_type);
            for (int i = 0; i < from.Length; ++i)
            {
                // copy.
                object obj = Activator.CreateInstance(blittable_element_type);
                Campy.Types.Utils.Utility.CopyToBlittableType(from.GetValue(i), ref obj);
                Marshal.StructureToPtr(obj, byte_ptr, false);
                byte_ptr = new IntPtr((long)byte_ptr + size_element);
            }
            return cpp_array;
        }

        public static IntPtr CopyFromNativeArray(IntPtr a, Array to, Type blittable_element_type)
        {
            int size_element = Marshal.SizeOf(blittable_element_type);
            IntPtr mem = a;
            for (int i = 0; i < to.Length; ++i)
            {
                // copy.
                object obj = Marshal.PtrToStructure(mem, blittable_element_type);
                object to_obj = to.GetValue(i);
                Campy.Types.Utils.Utility.CopyFromBlittableType(obj, ref to_obj);
                to.SetValue(to_obj, i);
                mem = new IntPtr((long)mem + size_element);
            }
            return a;
        }
    }
}
