using System.Runtime.CompilerServices;

namespace Campy.Compiler
{
    using Utils;
    using Swigged.Cuda;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Reflection;
    using System.Runtime.InteropServices;
    using System.Runtime.Serialization;
    using System.Text;

    /// <summary>
    /// This code marshals C#/Net data structures that have an unknown implementation to/from
    /// the implementation for NVIDIA GPUs.
    /// </summary>
    public class BUFFERS
    {
        private Dictionary<string, string> _type_name_map = new Dictionary<string, string>();

        // A dictionary of allocated blocks of memory corresponding to an object in C#,
        // calculated when a C# object is copied to the GPU space.
        private Dictionary<object, IntPtr> _allocated_objects = new Dictionary<object, IntPtr>();

        // The above mapping in reverse.
        private Dictionary<IntPtr, object> _allocated_buffers = new Dictionary<IntPtr, object>();

        // A list of objects with deep copy level.
        private Dictionary<object, int> _allocated_object_level = new Dictionary<object, int>();

        // A list of object that have been copied from GPU space back to C#.
        private List<object> _copied_from_gpu = new List<object>();

        // A list of object that should not be copied back to the CPU after a For loop call.
        private List<object> _delayed_from_gpu = new List<object>();

        // A list of object that should not be copied back to the CPU after a For loop call.
        private List<object> _never_copy_from_gpu = new List<object>();

        // A list of object that have been copied to GPU space.
        private List<object> _copied_to_gpu
        {
            get;
            set;
        } = new List<object>();

        private int _level = 0;
        public bool _delay = false;

        public void Synchronize()
        {
        }

        public void Delay(object obj)
        {
            if (_delayed_from_gpu.Contains(obj))
                return;
            _delayed_from_gpu.Add(obj);
        }

        public void ClearAllocatedObjects()
        {
        }

        public BUFFERS()
        {
        }

        /// <summary>
        /// This code to check if a type is blittable.
        /// See http://aakinshin.net/blog/post/blittable/
        /// Original from https://stackoverflow.com/questions/10574645/the-fastest-way-to-check-if-a-type-is-blittable/31485271#31485271
        /// Purportedly, System.Decimal is supposed to not be blittable, but appears on Windows 10, VS 2017, NF 4.6.
        /// </summary>

        public static bool IsBlittable<T>()
        {
            return IsBlittableCache<T>.Value;
        }

        public static bool IsBlittable(System.Type type)
        {
            if (type.IsArray)
            {
                var elem = type.GetElementType();
                return elem.IsValueType && IsBlittable(elem);
            }
            try
            {
                object instance = FormatterServices.GetUninitializedObject(type);
                GCHandle.Alloc(instance, GCHandleType.Pinned).Free();
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static class IsBlittableCache<T>
        {
            public static readonly bool Value = IsBlittable(typeof(T));
        }

        public static int Alignment(System.Type type)
        {
            if (type.IsArray || type.IsClass)
                return 8;
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

        public static int Padding(long offset, int alignment)
        {
            int padding = (alignment - (int)(offset % alignment)) % alignment;
            return padding;
        }

        public static int SizeOf(System.Type type)
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
            else if (type.IsStruct() || type.IsClass)
            {
                var fields = type.GetFields(
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
                        field_size = Marshal.SizeOf(typeof(IntPtr));
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
                result = Marshal.SizeOf(type);
            return result;
        }

        public void ResetDataStructures()
        {
            _copied_from_gpu = new List<object>();
            _copied_to_gpu = new List<object>();
        }

        // Create a buffer corresponding to the given object. If there is a buffer
        // for the object already noted, return the existing buffer. Copy the C#
        // object to the buffer.
        public IntPtr AddDataStructure(object to_gpu)
        {
            IntPtr result = IntPtr.Zero;
            var type = to_gpu.GetType();
            var btype = type;

            result = _allocated_objects.Where(p => p.Key == to_gpu).FirstOrDefault().Value;

            // Allocate new buffer for object on GPU.
            if (result == IntPtr.Zero)
            {
                if (Campy.Utils.Options.IsOn("copy_trace"))
                    System.Console.WriteLine("Allocating GPU buf " + to_gpu);
                result = New(BUFFERS.SizeOf(btype));
                _allocated_objects[to_gpu] = result;
                _allocated_object_level[to_gpu] = _level + 1;
            }

            // Copy to GPU if it hasn't been done before.
            if (!_copied_to_gpu.Contains(result))
                DeepCopyToImplementation(to_gpu, result);

            return result;
        }

        public void SynchDataStructures()
        {
            // Copy all pointers from shared global or device memory back to C# space.
            // As this is a deep copy, and because it is performed bottom-up, do only
            // "top level" data structures.
            Stack<object> nuke = new Stack<object>();

            foreach (object k in _copied_to_gpu)
            {
                if (_never_copy_from_gpu.Contains(k))
                    continue;

                if (_delayed_from_gpu.Contains(k))
                    continue;

                if (!_allocated_objects.ContainsKey(k))
                    continue; // Honestly, this is actually a problem as the object was somehow lost.

                if (!_allocated_object_level.ContainsKey(k))
                    throw new Exception();

                IntPtr v = _allocated_objects[k];

                DeepCopyFromImplementation(v, out object to, k.GetType());

                nuke.Push(k);
            }

            // After copying object back to CPU, we need to delete the GPU copy
            // and remove it from the list.
            while (nuke.Count > 0)
            {
                object k = nuke.Pop();
                IntPtr v = _allocated_objects[k];

                _copied_to_gpu.Remove(k);
                _allocated_objects.Remove(k);
                _copied_from_gpu.Remove(k);
                Free(v);
            }

        }

        public void FullSynch()
        {
            // Reset any delayed object copies. In other words, copy the objects
            // on this list back to the CPU.
            _delayed_from_gpu = new List<object>();

            // Copy objects from GPU to CPU.
            SynchDataStructures();
        }

        public int SizeOf(object obj)
        {
            System.Type type = obj.GetType();
            if (type.IsArray)
            {
                Array array = (Array)obj;
                var bytes = 0;
                int rank = array.Rank;
                var blittable_element_type = array.GetType().GetElementType();
                if (array.GetType().GetElementType().IsClass)
                {
                    // We create a buffer for the class, and stuff a pointer in the array.
                    bytes = BUFFERS.SizeOf(typeof(IntPtr)); // Pointer
                    bytes += BUFFERS.SizeOf((typeof(Int64))); // Rank
                    bytes += BUFFERS.SizeOf(typeof(Int64)) * rank; // Length for each dimension
                    bytes += BUFFERS.SizeOf(typeof(IntPtr)) * array.Length; // Elements
                }
                else
                {
                    // We create a buffer for the class, and stuff a pointer in the array.
                    bytes = BUFFERS.SizeOf(typeof(IntPtr)); // Pointer
                    bytes += BUFFERS.SizeOf((typeof(Int64))); // Rank
                    bytes += BUFFERS.SizeOf(typeof(Int64)) * rank; // Length for each dimension
                    bytes += BUFFERS.SizeOf(blittable_element_type) * array.Length; // Elements
                }
                return bytes;
            }
            else if (type.FullName == "System.String")
            {
                string str = (string)obj;
                var bytes = 0;
                bytes = 4 + str.Length * SizeOf(typeof(UInt16));
                return bytes;
            }
            else
                return SizeOf(type);
        }

        /// <summary>
        /// This method copies from a managed type into a blittable managed type.
        /// The type is converted from managed into a blittable managed type.
        /// </summary>
        /// <param name="from_cpu"></param>
        /// <param name="to"></param>
        public unsafe void DeepCopyToImplementation(object from_cpu, void* to_gpu)
        {
            // Copy object to a buffer.
            try
            {
                _level++;

                {
                    bool is_null = false;
                    try
                    {
                        if (from_cpu == null) is_null = true;
                        else if (from_cpu.Equals(null)) is_null = true;
                    }
                    catch (Exception e)
                    {
                    }
                    if (is_null)
                    {
                        throw new Exception("Unknown type of object.");
                    }
                }

                System.Type from_cpu_type = from_cpu.GetType();

                // Let's start with basic types.
                if (from_cpu_type.FullName.Equals("System.Object"))
                {
                    throw new Exception("Type is System.Object, but I don't know what to represent it as.");
                }
                if (from_cpu_type.FullName.Equals("System.Int16"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.Int32"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.Int64"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.UInt16"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.UInt32"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.UInt64"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.IntPtr"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }

                // Map boolean into byte.
                if (from_cpu_type.FullName.Equals("System.Boolean"))
                {
                    bool v = (bool)from_cpu;
                    System.Byte v2 = (System.Byte)(v ? 1 : 0);
                    Cp(to_gpu, v2);
                    return;
                }

                // Map char into uint16.
                if (from_cpu_type.FullName.Equals("System.Char"))
                {
                    Char v = (Char)from_cpu;
                    System.UInt16 v2 = (System.UInt16)v;
                    Cp(to_gpu, v2);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.Single"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }
                if (from_cpu_type.FullName.Equals("System.Double"))
                {
                    Cp(to_gpu, from_cpu);
                    return;
                }

                String name = from_cpu_type.FullName;
                _type_name_map.TryGetValue(name, out string alt);
                System.Reflection.TypeFilter tf = new System.Reflection.TypeFilter((System.Type t, object o) =>
                {
                    return t.FullName == name || t.FullName == alt;
                });

                var blittable_type = from_cpu_type;

                if (from_cpu_type.FullName.Equals("System.String"))
                {
                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Adding object to 'copied_to_gpu' " + from_cpu);
                    _copied_to_gpu.Add(from_cpu);

                    System.Type f = from_cpu.GetType();
                    System.Type tr = blittable_type;
                    int* ip = (int*) to_gpu;
                    string s = (string)from_cpu;
                    int v = s.Length;
                    *ip = v;
                    ++ip;
                    short* sp = (short*)ip;
                    for (int i = 0; i < v; ++i)
                        *sp++ = (short)s[i];
                    return;
                }

                if (from_cpu_type.IsArray)
                {
                    // First, make sure allocated object 
                    if (_copied_to_gpu.Contains(from_cpu))
                    {
                        if (Campy.Utils.Options.IsOn("copy_trace"))
                            System.Console.WriteLine("Not copying object to GPU -- already done.' " + from_cpu);
                        // Full object already stuffed into implementation buffer.
                    }
                    else
                    {
                        if (Campy.Utils.Options.IsOn("copy_trace"))
                            System.Console.WriteLine("Adding object to 'copied_to_gpu' " + from_cpu);
                        _copied_to_gpu.Add(from_cpu);

                        // An array is represented as a struct, Runtime::A.
                        // The data in the array is contained in the buffer following the length.
                        // The buffer allocated must be big enough to contain all data. Use
                        // Buffer.SizeOf(array) to get the representation buffer size.
                        // If the element is an array or a class, a buffer is allocated for each
                        // element, and an intptr used in the array.
                        Array a = (Array)from_cpu;
                        int rank = a.Rank;
                        int len = a.Length;
                        int bytes = SizeOf(a);
                        var destIntPtr = (byte*)to_gpu;
                        byte* df_ptr = destIntPtr;
                        byte* df_rank = df_ptr + BUFFERS.SizeOf(typeof(IntPtr));
                        byte* df_length = df_rank + BUFFERS.SizeOf(typeof(Int64));
                        byte* df_elements = df_length + BUFFERS.SizeOf(typeof(Int64)) * rank;
                        Cp(df_ptr, (IntPtr)df_elements); // Copy df_elements to *df_ptr
                        Cp(df_rank, rank);
                        for (int i = 0; i < rank; ++i)
                            Cp(df_length + i * BUFFERS.SizeOf(typeof(Int64)), a.GetLength(i));
                        CpArrayToGpu(df_elements, a);
                    }
                    return;
                }

                if (from_cpu_type.IsStruct() || from_cpu_type.IsClass)
                {
                    // Classes are not copied if already copied before, AND
                    // if it isn't a closure object. Normally, we wouldn't copy anything,
                    // but it turns out some algorithms modify the closure, which has
                    // nested closure. So, we copy these objects.
                    if (from_cpu_type.Name.StartsWith("<>c__DisplayClass") ||
                        from_cpu_type
                            .GetCustomAttributes(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute),
                                false).Length > 0)
                        ;
                    else if (_copied_to_gpu.Contains(from_cpu) && !from_cpu_type.IsStruct())
                    {
                        if (Campy.Utils.Options.IsOn("copy_trace"))
                            System.Console.WriteLine("Not copying object to GPU -- already done.' " + from_cpu);
                        // Full object already stuffed into implementation buffer.
                        return;
                    }
                    {
                        if (!from_cpu_type.IsStruct())
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Adding object to 'copied_to_gpu' " + from_cpu);
                            _copied_to_gpu.Add(from_cpu);
                        }

                        System.Type f = from_cpu.GetType();
                        System.Type tr = blittable_type;
                        int size = SizeOf(tr);
                        void* ip = to_gpu;
                        var rffi = f.GetRuntimeFields();
                        var ffi = f.GetFields(
                            System.Reflection.BindingFlags.Instance
                            | System.Reflection.BindingFlags.NonPublic
                            | System.Reflection.BindingFlags.Public
                            //| System.Reflection.BindingFlags.Static
                        );
                        var tfi = tr.GetFields(
                            System.Reflection.BindingFlags.Instance
                            | System.Reflection.BindingFlags.NonPublic
                            | System.Reflection.BindingFlags.Public
                            //| System.Reflection.BindingFlags.Static
                        );

                        foreach (System.Reflection.FieldInfo fi in ffi)
                        {
                            object field_value = fi.GetValue(from_cpu);
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Copying field "
                                                         + field_value
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(field_value));
                            if (field_value != null && Campy.Utils.Options.IsOn("copy_trace"))
                                    System.Console.WriteLine("Copying field type "
                                                             + field_value.GetType().FullName);

                            String na = fi.Name;
                            var tfield = tfi.Where(k => k.Name == fi.Name).FirstOrDefault();
                            if (tfield == null) throw new ArgumentException("Field not found.");
                            if (fi.FieldType.IsArray)
                            {
                                // Allocate a whole new buffer, copy to that, place buffer pointer into field at ip.
                                ip = (void*)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))));
                                if (field_value != null)
                                {
                                    Array ff = (Array)field_value;
                                    var field_size = SizeOf(ff);
                                    IntPtr gp;
                                    if (_allocated_objects.ContainsKey(field_value))
                                    {
                                        gp = _allocated_objects[field_value];
                                    }
                                    else
                                    {
                                        if (Campy.Utils.Options.IsOn("copy_trace"))
                                            System.Console.WriteLine("Allocating GPU buf " + field_value);
                                        gp = New(field_size);
                                        _allocated_objects[field_value] = (IntPtr)gp;
                                        _allocated_object_level[field_value] = _level + 1;
                                    }
                                    DeepCopyToImplementation(gp, ip);
                                    DeepCopyToImplementation(field_value, gp);
                                }
                                else
                                {
                                    field_value = IntPtr.Zero;
                                    DeepCopyToImplementation(field_value, ip);
                                }
                                ip = (void*)((long)ip
                                             + BUFFERS.SizeOf(typeof(IntPtr)));
                            }
                            else if (field_value as Delegate != null)
                            {
                                ip = (void*)((long)ip
                                             + BUFFERS.SizeOf(typeof(IntPtr)));
                            }
                            else if (fi.FieldType.IsClass)
                            {
                                System.Console.WriteLine("field type is " + fi.FieldType.FullName);
                                // Allocate a whole new buffer, copy to that, place buffer pointer into field at ip.
                                if (field_value != null)
                                {
                                    ip = (void*)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))));
                                    IntPtr gp;
                                    if (_allocated_objects.ContainsKey(field_value))
                                    {
                                        gp = _allocated_objects[field_value];
                                    }
                                    else
                                    {
                                        var field_size = SizeOf(field_value);
                                        if (Campy.Utils.Options.IsOn("copy_trace"))
                                            System.Console.WriteLine("Allocating GPU buf " + field_value);
                                        gp = New(field_size);
                                        _allocated_objects[field_value] = (IntPtr)gp;
                                        _allocated_object_level[field_value] = _level + 1;
                                    }
                                    // Copy pointer to field.
                                    DeepCopyToImplementation(gp, ip);
                                    // Copy object to GPU.
                                    DeepCopyToImplementation(field_value, gp);
                                }
                                else
                                {
                                    field_value = IntPtr.Zero;
                                    DeepCopyToImplementation(field_value, ip);
                                }
                                ip = (void*)((long)ip
                                             + BUFFERS.SizeOf(typeof(IntPtr)));
                            }
                            else if (fi.FieldType.IsStruct())
                            {
                                throw new Exception("Whoops.");
                            }
                            else
                            {
                                DeepCopyToImplementation(field_value, ip);
                                var field_size = SizeOf(tfield.FieldType);
                                ip = (void*)((long)ip + field_size);
                            }
                        }
                    }
                    return;
                }

                throw new Exception("Unknown type.");
            }
            catch (Exception e)
            {
                System.Console.WriteLine("Exception");
                System.Console.WriteLine(e);
                throw e;
            }
            finally
            {
                _level--;
            }
        }

        public void DeepCopyToImplementation(object from_cpu, IntPtr to_buffer)
        {
            unsafe
            {
                if (Campy.Utils.Options.IsOn("copy_trace"))
                    System.Console.WriteLine("Copy from CPU " + from_cpu + " TYPE = "
                                             + from_cpu.GetType().FullName);
                DeepCopyToImplementation(from_cpu, (void*)to_buffer);
            }
        }

        public unsafe void DeepCopyFromImplementation(IntPtr from_gpu, out object to_cpu, System.Type target_type)
        {
            try
            {
                _level++;

                System.Type t_type = target_type;
                System.Type f_type = t_type;

                if (t_type.FullName.Equals("System.Object"))
                {
                    to_cpu = null;
                    return;
                }
                if (t_type.FullName.Equals("System.Int16"))
                {
                    object o = Marshal.PtrToStructure<System.Int16>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.Int32"))
                {
                    object o = Marshal.PtrToStructure<System.Int32>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.Int64"))
                {
                    object o = Marshal.PtrToStructure<System.Int64>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.UInt16"))
                {
                    object o = Marshal.PtrToStructure<System.UInt16>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.UInt32"))
                {
                    object o = Marshal.PtrToStructure<System.UInt32>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.UInt64"))
                {
                    object o = Marshal.PtrToStructure<System.UInt64>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.IntPtr"))
                {
                    object o = Marshal.PtrToStructure<System.IntPtr>(from_gpu);
                    to_cpu = o;
                    return;
                }

                // Map boolean into byte.
                if (t_type.FullName.Equals("System.Boolean"))
                {
                    byte v = *(byte*)from_gpu;
                    to_cpu = (System.Boolean)(v == 1 ? true : false);
                    return;
                }

                // Map char into uint16.
                if (t_type.FullName.Equals("System.Char"))
                {
                    to_cpu = (System.Char)from_gpu;
                    return;
                }
                if (t_type.FullName.Equals("System.Single"))
                {
                    object o = Marshal.PtrToStructure<System.Single>(from_gpu);
                    to_cpu = o;
                    return;
                }
                if (t_type.FullName.Equals("System.Double"))
                {
                    object o = Marshal.PtrToStructure<System.Double>(from_gpu);
                    to_cpu = o;
                    return;
                }

                if (t_type.FullName.Equals("System.String"))
                {
                    // For now, assume data exists on GPU. Perform memcpy using CUDA.
                    int* block = stackalloc int[1];
                    IntPtr intptr = new IntPtr(block);
                    var res = Cuda.cuMemcpyDtoH_v2(intptr, from_gpu, sizeof(int));
                    int len = *block;
                    short* block2 = stackalloc short[len + 1];
                    var intptr2 = new IntPtr(block2);
                    Cuda.cuMemcpyDtoH_v2(intptr2, from_gpu + sizeof(int), (uint)len * sizeof(short));
                    block2[len] = 0;
                    to_cpu = new string((char*)intptr2);
                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copy from GPU " + to_cpu);
                    return;
                }

                if (t_type.IsArray)
                {
                    if (from_gpu == IntPtr.Zero)
                    {
                        to_cpu = null;
                        return;
                    }

                    Array to_array = (Array)_allocated_objects.Where(p => p.Value == from_gpu).Select(p => p.Key)
                        .FirstOrDefault();

                    to_cpu = to_array;

                    if (to_cpu != default(Array))
                    {
                        if (_delayed_from_gpu.Contains(to_cpu))
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Not copying to CPU "
                                                         + to_cpu
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(to_cpu)
                                                         + " because it is delayed.");
                            return;
                        }
                        if (_never_copy_from_gpu.Contains(to_cpu))
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Not copying to CPU "
                                                         + to_cpu
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(to_cpu)
                                                         + " because it never copied to GPU.");
                            return;
                        }
                        if (_copied_from_gpu.Contains(to_cpu))
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Not copying to CPU "
                                                         + to_cpu
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(to_cpu)
                                                         + " because it was copied back before.");
                            return;
                        }
                    }

                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copying to CPU "
                                                 + to_cpu
                                                 + " "
                                                 + RuntimeHelpers.GetHashCode(to_cpu)
                                                 + " because it was copied back before.");

                    // "from" is assumed to be a unmanaged buffer
                    // with record Runtime.A used.
                    long* long_ptr = (long*)((long)(byte*)from_gpu);
                    long_ptr++;
                    int rank = (int)*long_ptr++;

                    System.Type to_element_type = t_type.GetElementType();
                    System.Type from_element_type = to_element_type;
                    if (to_element_type.IsArray || to_element_type.IsClass)
                        from_element_type = typeof(IntPtr);
                    int[] dims = new int[rank];
                    for (int kk = 0; kk < rank; ++kk)
                        dims[kk] = (int)*long_ptr++;

                    _allocated_buffers[from_gpu] = to_cpu;

                    _copied_from_gpu.Add(to_array);

                    CpArraytoCpu((void*)long_ptr, to_array, from_element_type);
                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copy from GPU " + to_cpu);
                    return;
                }

                if (t_type.IsClass)
                {
                    IntPtr ip = from_gpu;
                    if (from_gpu == IntPtr.Zero)
                    {
                        to_cpu = null;
                        return;
                    }

                    to_cpu = _allocated_objects.Where(p => p.Value == ip).Select(p => p.Key)
                        .FirstOrDefault();

                    _allocated_buffers[from_gpu] = to_cpu;

                    if (to_cpu != default(Array))
                    {
                        if (_delayed_from_gpu.Contains(to_cpu))
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Not copying to CPU "
                                                         + to_cpu
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(to_cpu)
                                                         + " because it is delayed.");
                            return;
                        }
                        if (_never_copy_from_gpu.Contains(to_cpu))
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Not copying to CPU "
                                                         + to_cpu
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(to_cpu)
                                                         + " because it never copied to GPU.");
                            return;
                        }
                        if (_copied_from_gpu.Contains(to_cpu))
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Not copying to CPU "
                                                         + to_cpu
                                                         + " "
                                                         + RuntimeHelpers.GetHashCode(to_cpu)
                                                         + " because it was copied back before.");
                            return;
                        }
                    }

                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copying to CPU "
                                                 + to_cpu
                                                 + " "
                                                 + RuntimeHelpers.GetHashCode(to_cpu)
                                                 + " because it was copied back before.");

                    FieldInfo[] all_from_fieldinfo = f_type.GetFields(
                        System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                        | System.Reflection.BindingFlags.Public
                    //| System.Reflection.BindingFlags.Static
                    );
                    FieldInfo[] all_to_fieldinfo = t_type.GetFields(
                        System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                        | System.Reflection.BindingFlags.Public
                    //| System.Reflection.BindingFlags.Static
                    );

                    for (int i = 0; i < all_from_fieldinfo.Length; ++i)
                    {
                        var from_fieldinfo = all_from_fieldinfo[i];
                        var to_fieldinfo = all_to_fieldinfo.Where(k => k.Name == from_fieldinfo.Name).FirstOrDefault();
                        if (to_fieldinfo == null) throw new ArgumentException("Field not found.");
                        // Note, special case all field types.
                        if (to_fieldinfo.FieldType.IsArray)
                        {
                            ip = (IntPtr)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))));
                            int field_size = BUFFERS.SizeOf(typeof(IntPtr));
                            IntPtr ipv = (IntPtr)Marshal.PtrToStructure<IntPtr>(ip);
                            if (ipv == IntPtr.Zero)
                            {
                            }
                            else if (_allocated_buffers.ContainsKey(ipv))
                            {
                                object tooo = _allocated_buffers[ipv];
                                to_fieldinfo.SetValue(to_cpu, tooo);
                            }
                            else
                            {
                                DeepCopyFromImplementation(ipv, out object tooo, to_fieldinfo.FieldType);
                                to_fieldinfo.SetValue(to_cpu, tooo);
                            }
                            ip = (IntPtr)((long)ip + field_size);
                        }
                        else if (to_fieldinfo.FieldType.IsClass)
                        {
                            ip = (IntPtr)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))));
                            int field_size = BUFFERS.SizeOf(typeof(IntPtr));
                            IntPtr ipv = (IntPtr)Marshal.PtrToStructure<IntPtr>(ip);
                            DeepCopyFromImplementation(ipv, out object tooo, to_fieldinfo.FieldType);
                            to_fieldinfo.SetValue(to_cpu, tooo);
                            ip = (IntPtr)((long)ip + field_size);
                        }
                        else
                        {
                            int field_size = BUFFERS.SizeOf(from_fieldinfo.FieldType);
                            ip = (IntPtr)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(from_fieldinfo.FieldType)));
                            DeepCopyFromImplementation(ip, out object tooo, to_fieldinfo.FieldType);
                            to_fieldinfo.SetValue(to_cpu, tooo);
                            ip = (IntPtr)((long)ip + field_size);
                        }
                    }
                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copy from GPU " + to_cpu);

                    return;
                }

                if (t_type.IsStruct())
                {
                    IntPtr ip = from_gpu;
                    if (ip == IntPtr.Zero)
                    {
                        to_cpu = null;
                        return;
                    }

                    to_cpu = Activator.CreateInstance(t_type);
                    _allocated_buffers[ip] = to_cpu;

                    FieldInfo[] ffi = f_type.GetFields(
                        System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                        | System.Reflection.BindingFlags.Public
                    //| System.Reflection.BindingFlags.Static
                    );
                    FieldInfo[] tfi = t_type.GetFields(
                        System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                        | System.Reflection.BindingFlags.Public
                    //| System.Reflection.BindingFlags.Static
                    );

                    for (int i = 0; i < ffi.Length; ++i)
                    {
                        var ffield = ffi[i];
                        var tfield = tfi.Where(k => k.Name == ffield.Name).FirstOrDefault();
                        if (tfield == null) throw new ArgumentException("Field not found.");
                        // Note, special case all field types.
                        if (tfield.FieldType.IsArray)
                        {
                            ip = (IntPtr)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))));
                            int field_size = BUFFERS.SizeOf(typeof(IntPtr));
                            IntPtr ipv = (IntPtr)Marshal.PtrToStructure<IntPtr>(ip);
                            if (ipv == IntPtr.Zero)
                            {
                            }
                            else if (_allocated_buffers.ContainsKey(ipv))
                            {
                                object tooo = _allocated_buffers[ipv];
                                tfield.SetValue(to_cpu, tooo);
                            }
                            else
                            {
                                DeepCopyFromImplementation(ipv, out object tooo, tfield.FieldType);
                                tfield.SetValue(to_cpu, tooo);
                            }
                            ip = (IntPtr)((long)ip + field_size);
                        }
                        else if (tfield.FieldType.IsClass)
                        {
                            ip = (IntPtr)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(typeof(IntPtr))));
                            int field_size = BUFFERS.SizeOf(typeof(IntPtr));
                            IntPtr ipv = (IntPtr)Marshal.PtrToStructure<IntPtr>(ip);
                            if (_allocated_buffers.ContainsKey(ipv))
                            {
                                object tooo = _allocated_buffers[ipv];
                                tfield.SetValue(to_cpu, tooo);
                            }
                            else
                            {
                                DeepCopyFromImplementation(ipv, out object tooo, tfield.FieldType);
                                tfield.SetValue(to_cpu, tooo);
                            }
                            ip = (IntPtr)((long)ip + field_size);
                        }
                        else
                        {
                            int field_size = BUFFERS.SizeOf(ffield.FieldType);
                            ip = (IntPtr)((long)ip + BUFFERS.Padding((long)ip, BUFFERS.Alignment(ffield.FieldType)));
                            DeepCopyFromImplementation(ip, out object tooo, tfield.FieldType);
                            tfield.SetValue(to_cpu, tooo);
                            ip = (IntPtr)((long)ip + field_size);
                        }
                    }
                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copy from GPU " + to_cpu);

                    return;
                }

                throw new Exception("Unknown type.");
            }
            catch (Exception e)
            {
                System.Console.WriteLine("Exception");
                System.Console.WriteLine(e);
                throw e;
            }
            finally
            {
                _level--;
            }
        }

        private unsafe void CpArrayToGpu(byte* to_gpu, Array from_cpu)
        {
            System.Type orig_element_type = from_cpu.GetType().GetElementType();

            // As the array could be multi-dimensional, we need to do a copy in row major order.
            // This is essentially the same as doing a number conversion to a string and vice versa
            // over the total number of elements in the entire multi-dimensional array.
            // See https://stackoverflow.com/questions/7123490/how-compiler-is-converting-integer-to-string-and-vice-versa
            // https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays/
            long total_size = 1;
            for (int i = 0; i < from_cpu.Rank; ++i)
                total_size *= from_cpu.GetLength(i);
            for (int i = 0; i < total_size; ++i)
            {
                int[] index = new int[from_cpu.Rank];
                string s = "";
                int c = i;
                for (int j = from_cpu.Rank - 1; j >= 0; --j)
                {
                    int ind_size = from_cpu.GetLength(j);
                    var remainder = c % ind_size;
                    c = c / from_cpu.GetLength(j);
                    index[j] = remainder;
                    s = (char)((short)('0') + remainder) + s;
                }
                //sdfg
                var from_element_value = from_cpu.GetValue(index);
                if (orig_element_type.FullName == "System.String")
                {
                    if (from_element_value != null)
                    {
                        int size_element = 4 + 2 * ((string)from_element_value).Length;
                        if (_allocated_objects.ContainsKey(from_element_value))
                        {
                            IntPtr gp = _allocated_objects[from_element_value];
                            DeepCopyToImplementation(gp, to_gpu);
                        }
                        else
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Allocating GPU buf " + from_element_value);
                            IntPtr gp = New(size_element);
                            DeepCopyToImplementation(from_element_value, gp);
                            DeepCopyToImplementation(gp, to_gpu);
                        }
                        to_gpu = (byte*)((long)to_gpu
                                     + BUFFERS.Padding((long)to_gpu, BUFFERS.Alignment(typeof(IntPtr)))
                                     + BUFFERS.SizeOf(typeof(IntPtr)));
                    }
                    else
                    {
                        from_element_value = IntPtr.Zero;
                        DeepCopyToImplementation(from_element_value, to_gpu);
                        to_gpu = (byte*)((long)to_gpu
                                     + BUFFERS.Padding((long)to_gpu, BUFFERS.Alignment(typeof(IntPtr)))
                                     + BUFFERS.SizeOf(typeof(IntPtr)));
                    }
                }
                else if (orig_element_type.IsArray || orig_element_type.IsClass)
                {
                    if (from_element_value != null)
                    {
                        // Each element is a pointer.
                        var size_element = SizeOf(from_element_value);
                        IntPtr gp;
                        if (_allocated_objects.ContainsKey(from_element_value))
                        {
                            gp = _allocated_objects[from_element_value];
                        }
                        else
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Allocating GPU buf " + size_element);
                            gp = New(size_element);
                            _allocated_objects[from_element_value] = (IntPtr)gp;
                            _allocated_object_level[from_element_value] = _level + 1;
                        }
                        DeepCopyToImplementation(gp, to_gpu);
                        DeepCopyToImplementation(from_element_value, gp);

                        to_gpu = (byte*)((long)to_gpu
                                     + BUFFERS.Padding((long)to_gpu, BUFFERS.Alignment(typeof(IntPtr)))
                                     + BUFFERS.SizeOf(typeof(IntPtr)));
                    }
                    else
                    {
                        from_element_value = IntPtr.Zero;
                        DeepCopyToImplementation(from_element_value, to_gpu);
                        to_gpu = (byte*)((long)to_gpu
                                     + BUFFERS.Padding((long)to_gpu, BUFFERS.Alignment(typeof(IntPtr)))
                                     + BUFFERS.SizeOf(typeof(IntPtr)));
                    }
                }
                else
                {
                    int size_element = BUFFERS.SizeOf(from_element_value.GetType());
                    DeepCopyToImplementation(from_element_value, to_gpu);
                    to_gpu = (byte*)((long)to_gpu
                        + BUFFERS.Padding((long)to_gpu, BUFFERS.Alignment(from_element_value.GetType()))
                        + size_element);
                }
            }
        }

        private unsafe void CpArraytoCpu(void* from_gpu, Array to_cpu, System.Type from_element_type)
        {
            var to_type = to_cpu.GetType();
            if (!to_type.IsArray)
                throw new Exception("Expecting array.");
            var to_element_type = to_cpu.GetType().GetElementType();
            IntPtr mem = (IntPtr)from_gpu;
            for (int i = 0; i < to_cpu.Length; ++i)
            {
                int[] index = new int[to_cpu.Rank];
                string s = "";
                int c = i;
                for (int j = to_cpu.Rank - 1; j >= 0; --j)
                {
                    int ind_size = to_cpu.GetLength(j);
                    var remainder = c % ind_size;
                    c = c / to_cpu.GetLength(j);
                    index[j] = remainder;
                    s = (char)((short)('0') + remainder) + s;
                }
                //sdfg
                if (to_element_type.IsArray || to_element_type.IsClass)
                {
                    object obj = Marshal.PtrToStructure(mem, typeof(IntPtr));
                    IntPtr obj_intptr = (IntPtr) obj;
                    DeepCopyFromImplementation(obj_intptr, out object to_obj, to_cpu.GetType().GetElementType());
                    to_cpu.SetValue(to_obj, index);
                    mem = new IntPtr((long)mem + SizeOf(typeof(IntPtr)));
                }
                else
                {
                    DeepCopyFromImplementation(mem, out object to_obj, to_cpu.GetType().GetElementType());
                    to_cpu.SetValue(to_obj, index);
                    int from_size_element = BUFFERS.SizeOf(from_element_type);
                    mem = new IntPtr((long)mem + from_size_element);
                }
            }
        }

        public static string OutputType(System.Type type)
        {
            if (type.IsArray)
            {
                return type.GetElementType().FullName + "[]";
            }
            if (type.IsValueType && !type.IsStruct())
            {
                return type.FullName;
            }
            StringBuilder sb = new StringBuilder();
            var fields = type.GetFields(
                System.Reflection.BindingFlags.Instance
                | System.Reflection.BindingFlags.NonPublic
                | System.Reflection.BindingFlags.Public
                //| System.Reflection.BindingFlags.Static
            );
            if (type.IsValueType && type.IsStruct())
                sb.Append("struct {").AppendLine();
            else if (type.IsClass)
                sb.Append("class {").AppendLine();
            foreach (var field in fields)
                sb.AppendFormat("{0} = {1}", field.Name, field.FieldType.Name).AppendLine();
            sb.Append("}").AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// Allocated a GPU buffer.
        /// Code based on https://www.codeproject.com/Articles/32125/Unmanaged-Arrays-in-C-No-Problem
        /// </summary>
        public IntPtr New(int bytes)
        {
            if (true)
            {
                // Let's try allocating a block of memory on the host. cuMemHostAlloc allocates bytesize
                // bytes of host memory that is page-locked and accessible to the device.
                // Note: cuMemHostAlloc and cuMemAllocHost seem to be almost identical except for the
                // third parameter to cuMemHostAlloc that is used for the type of memory allocation.
                var size = bytes;
                var res = Cuda.cuMemHostAlloc(out IntPtr pointer, (uint)size, (uint)Cuda.CU_MEMHOSTALLOC_DEVICEMAP);
                CudaHelpers.CheckCudaError(res);
                if (Campy.Utils.Options.IsOn("memory_trace"))
                    System.Console.WriteLine("Cu Alloc (" + bytes + " bytes) {0:X}", pointer.ToInt64());
                return pointer;
            }

            //if (false)
            //{
            //    // Allocate CPU memory, pin it, then register it with GPU.
            //    int f = new int();
            //    GCHandle handle = GCHandle.Alloc(f, GCHandleType.Pinned);
            //    IntPtr pointer = (IntPtr)handle;
            //    var size = Marshal.SizeOf(f);
            //    var res = Cuda.cuMemHostRegister_v2(pointer, (uint)size, (uint)Cuda.CU_MEMHOSTALLOC_DEVICEMAP);
            //    if (res == CUresult.CUDA_SUCCESS) System.Console.WriteLine("Worked.");
            //    else System.Console.WriteLine("Did not work.");
            //}

            {
                // Allocate Unified Memory.
                var size = bytes;
                var res = Cuda.cuMemAllocManaged(out IntPtr pointer, (uint)size, (uint)Swigged.Cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL);
                if (Campy.Utils.Options.IsOn("memory_trace"))
                    System.Console.WriteLine("Cu Alloc (" + bytes + " bytes) " + pointer);
                Utils.CudaHelpers.CheckCudaError(res);
                return pointer;
            }

            //if (false)
            //{
            //    return Marshal.AllocHGlobal(bytes);
            //}
        }

        public void Free(IntPtr pointer)
        {
            if (Campy.Utils.Options.IsOn("memory_trace"))
                System.Console.WriteLine("Cu Free {0:X}", pointer.ToInt64());
            var res = Cuda.cuMemFreeHost(pointer);
            CudaHelpers.CheckCudaError(res);
            //Marshal.FreeHGlobal(pointerToUnmanagedMemory);
        }

        public unsafe void* Resize<T>(void* oldPointer, int newElementCount)
            where T : struct
        {
            return (Marshal.ReAllocHGlobal(new IntPtr(oldPointer),
                new IntPtr(BUFFERS.SizeOf(typeof(T)) * newElementCount))).ToPointer();
        }

        public static void Cp(IntPtr destPtr, IntPtr srcPtr, int size)
        {
            unsafe
            {
                // srcPtr and destPtr are IntPtr's pointing to valid memory locations
                // size is the number of bytes to copy
                byte* src = (byte*)srcPtr;
                byte* dest = (byte*)destPtr;
                for (int i = 0; i < size; i++)
                {
                    dest[i] = src[i];
                }
            }
        }

        public static unsafe void Cp(void* destPtr, object src)
        {
            Marshal.StructureToPtr(src, (IntPtr)destPtr, false);
        }
    }
}

