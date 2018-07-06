namespace Campy.Compiler
{
    using Mono.Cecil;
    using Swigged.Cuda;
    using System.Collections.Generic;
    using System.Linq;
    using System.Reflection;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using System.Text;
    using System;
    using Utils;
    using Campy.Meta;

    /// <summary>
    /// This code marshals C#/Net data structures to/from the GPU implementation.
    /// It also performs memory allocation and disposal for the GPU.
    /// </summary>
    public class BUFFERS
    {
        private Dictionary<string, string> _type_name_map { get; } = new Dictionary<string, string>();

        // A dictionary of allocated blocks of memory corresponding to an object in C#,
        // calculated when a C# object is copied to the GPU space.
        private Dictionary<object, IntPtr> _allocated_objects { get; } = new Dictionary<object, IntPtr>();

        // The above mapping in reverse.
        private Dictionary<IntPtr, object> _allocated_buffers { get; } = new Dictionary<IntPtr, object>();

        // A list of object that have been copied from GPU space back to C#.
        private List<object> _copied_from_gpu { get; } = new List<object>();

        // A list of object that should not be copied back to the CPU after a For loop call.
        private List<object> _delayed_from_gpu { get; } = new List<object>();

        // A list of object that should not be copied back to the CPU after a For loop call.
        private List<object> _never_copy_from_gpu { get; } = new List<object>();

        // A list of object that have been copied to GPU space.
        private List<object> _copied_to_gpu { get; } = new List<object>();

        public void Delay(object obj)
        {
            if (_delayed_from_gpu.Contains(obj))
                return;
            _delayed_from_gpu.Add(obj);
        }

        public void ReadOnly(object obj)
        {
            if (_never_copy_from_gpu.Contains(obj))
                return;
            _never_copy_from_gpu.Add(obj);
        }

        public static int Alignment(System.Type type)
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

        public static int Padding(long offset, int alignment)
        {
            int padding = (alignment - (int)(offset % alignment)) % alignment;
            return padding;
        }

        // Create a buffer corresponding to the given object. If there is a buffer
        // for the object already noted, return the existing buffer. Copy the C#
        // object to the buffer.
        public IntPtr AddDataStructure(object to_gpu)
        {
            IntPtr result = IntPtr.Zero;

            var type = to_gpu.GetType();
            var expected_bcl_type = type.ToMonoTypeReference();

            expected_bcl_type = expected_bcl_type.RewriteMonoTypeReference();

            var find_object = _allocated_objects.Where(p => p.Key == to_gpu);

            if (Campy.Utils.Options.IsOn("copy_trace"))
                System.Console.WriteLine("On CPU closure value before copying to GPU:"
                                         + Environment.NewLine
                                         + PrintCpuObject(0, to_gpu));

            //// Allocate new buffer for object on GPU.
            //if (!find_object.Any())
            //{
            //    result = New(to_gpu);
            //    _allocated_objects[to_gpu] = result;
            //}
            //else result = find_object.First().Value;

            //// Copy to GPU if it hasn't been done before.
            //if (!_copied_to_gpu.Contains(result))
            //    unsafe { DeepCopyToImplementation(to_gpu, (void*)result); }

            unsafe {result = (IntPtr)DCToBcl(to_gpu);}

            if (Campy.Utils.Options.IsOn("copy_trace"))
                System.Console.WriteLine("Closure value on GPU:"
                                         + Environment.NewLine
                                         + PrintBclObject(0, result, expected_bcl_type));

            return result;
        }

        public void SynchDataStructures()
        {
            // Copy all pointers from shared global or device memory back to C# space.
            // As this is a deep copy, and because it is performed bottom-up, do only
            // "top level" data structures.
            Stack<object> nuke = new Stack<object>();

            foreach (object back_to_cpu in _copied_to_gpu)
            {
                if (_never_copy_from_gpu.Contains(back_to_cpu))
                    continue;

                if (_delayed_from_gpu.Contains(back_to_cpu))
                    continue;

                if (!_allocated_objects.ContainsKey(back_to_cpu))
                    continue; // Honestly, this is actually a problem as the object was somehow lost.

                IntPtr gpu_buffer_pointer = _allocated_objects[back_to_cpu];

                if (Campy.Utils.Options.IsOn("copy_trace"))
                {
                    System.Console.Write("Copying GPU buffer {0:X}", gpu_buffer_pointer.ToInt64());
                    System.Console.Write(" back to CPU object " + back_to_cpu);
                    System.Console.WriteLine();
                }

                unsafe { DCToCpu((void*)gpu_buffer_pointer); }

                nuke.Push(back_to_cpu);
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
                _allocated_buffers.Remove(v);
                Free(v);
            }
            // GC.
            //BclGcCollect();

        }

        public void FullSynch()
        {
            // Reset any delayed object copies. In other words, copy the objects
            // on this list back to the CPU.
            _delayed_from_gpu.Clear();

            // Copy objects from GPU to CPU.
            SynchDataStructures();
        }

        // SizeOf operations
        //
        // When we are trying to work with sizes of objects, we need to be very
        // careful here on what we mean. In the BCL, an "object" is the memory of the
        // structured (aggregate) type allocated in the GC heap. For example, 
        // class A { int a; } has a size of 4 bytes. A "reference" is a pointer to
        // the "object". The BCL keeps a dictionary of all pointers in order to know
        // what the type of the pointer refers to. Consequently, a reference has a size
        // of 8 bytes for a 64-bit target.
        //
        // When allocating via "New()", we need to keep in mind we are trying to get
        // the size for allocating an object. When computing the size of an object,
        // we need to be careful to note what we are talking about: is it the object
        // or the reference? For value types, the size of the value type the size of the
        // base type. Boxed values is a reference, which is a pointer, which has a size
        // of 8 bytes.
        //
        // One added issue here is that Campy has two object spaces to deal with: C#
        // data structures in GPU space and data structures in CPU space. So, space computation
        // requires BCL calls to the meta data system. SizeOf operations should only really
        // deal with the BCL data structures, not what's back on the CPU.
        // Note, in DotNetAnywhere, the size of an object vs reference is distinguished
        // by the terms "heap size" and "stack size".
        //
        // SizeOf always refers to the size of the object, given a reference to an object or type.
        // SizeOfRefOrValType always reference to the size of the referenced type.

        public static int SizeOf(System.Type type)
        {
            int result = 0;
            if (type.IsArray)
            {
                throw new Exception("Improper use of SizeOf with Array.");
            }
            else if (type.FullName == "System.String")
            {
                throw new Exception("Improper use of SizeOf with String.");
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

        public static int SizeOf(Array obj)
        {
            var type = obj.GetType();
            var mono_type = type.ToMonoTypeReference();
            var bcl_type = mono_type.RewriteMonoTypeReference();
            if (bcl_type.IsArray)
            {
                Array array = (Array)obj;
                var bytes = 0;
                int rank = array.Rank;
                var bcl_element_type = bcl_type.ResolveGetElementType();
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
                    bytes += BUFFERS.SizeOfRefOrValType(bcl_element_type) * array.Length; // Elements
                }
                return bytes;
            }
            else
                throw new Exception("TypeOf error.");
        }

        public static int SizeOf(string obj)
        {
            var type = obj.GetType();
            var mono_type = type.ToMonoTypeReference();
            var bcl_type = mono_type.RewriteMonoTypeReference();
            if (type.FullName == "System.String")
            {
                string str = (string)obj;
                var bytes = 0;
                bytes = 4 + str.Length * SizeOf(typeof(UInt16));
                return bytes;
            }
            else
                throw new Exception("TypeOf error.");
        }

        public static int SizeOfRefOrValType(TypeReference type)
        {
            if (type.IsArray)
            {
                return 8;
            }
            else if (!type.IsValueType)
            {
                return 8;
            }
            // Let's start with basic types.
            else if (type.FullName.Equals("System.Object"))
            {
                return 8;
            }
            else if (type.FullName.Equals("System.Int16"))
            {
                return 2;
            }
            else if (type.FullName.Equals("System.Int32"))
            {
                return 4;
            }
            else if (type.FullName.Equals("System.Int64"))
            {
                return 8;
            }
            else if (type.FullName.Equals("System.UInt16"))
            {
                return 2;
            }
            if (type.FullName.Equals("System.UInt32"))
            {
                return 4;
            }
            else if (type.FullName.Equals("System.UInt64"))
            {
                return 8;
            }
            else if (type.FullName.Equals("System.IntPtr"))
            {
                return 8;
            }

            // Map boolean into byte.
            else if (type.FullName.Equals("System.Boolean"))
            {
                return 1;
            }

            // Map char into uint16.
            else if (type.FullName.Equals("System.Char"))
            {
                return 2;
            }
            else if (type.FullName.Equals("System.Single"))
            {
                return 4;
            }
            else if (type.FullName.Equals("System.Double"))
            {
                return 8;
            }

            return 0;
        }

        // For a reference, create a new object of the type in BCL, and copy. Then, return BCL reference.
        // For any value type, crash because the value had to have been boxed, which slows it down considerably.
        // Undoubtably, this routine has to understand the format of both the BCL and the object data
        // structure from the CPU.
        private unsafe IntPtr DCToBcl(object from_cpu)
        {
            // For null reference, return null.
            bool is_null = false;
            try
            {
                if (from_cpu == null) is_null = true;
                else if (from_cpu.Equals(null)) is_null = true;
            }
            catch (Exception)
            {
            }

            if (is_null) return IntPtr.Zero;

            // For value types, crash because we cannot "allocate" an object per se.
            if (from_cpu.GetType().IsValueType)
                throw new Exception("Trying to copy value type in DCToBcl");

            // Return previous cached object if copied before.
            if (_copied_to_gpu.Contains(from_cpu))
            {
                if (Campy.Utils.Options.IsOn("copy_trace"))
                    System.Console.WriteLine("Not copying object to GPU -- already done.' " + from_cpu);
                // Full object already stuffed into implementation buffer.
                return _allocated_objects[from_cpu];
            }

            if (Campy.Utils.Options.IsOn("copy_trace"))
                System.Console.WriteLine("Adding object to 'copied_to_gpu' " + from_cpu);
            _copied_to_gpu.Add(from_cpu);

            Type system_type = from_cpu.GetType();
            var mono_type = system_type.ToMonoTypeReference().RewriteMonoTypeReference();

            var result = New(from_cpu);
            _allocated_objects[from_cpu] = result;

            DCToBclValue(from_cpu, (void*)result);

            return result;
        }

        // Copy from a C# object to a specific location given by a pointer.
        // Note, the location where this object is copied to is the size of the object,
        // not a reference.
        private unsafe void DCToBclValue(object from_cpu, void* address)
        {
            // For null reference, set to_gpu to null--assume that it's a reference or boxed type.
            bool is_null = false;
            try
            {
                if (from_cpu == null) is_null = true;
                else if (from_cpu.Equals(null)) is_null = true;
            }
            catch (Exception)
            {
            }
            if (is_null)
            {
                if (address == null) throw new Exception("Bad copy to.");
                *(void**)address = null;
                return;
            }


            Type system_type = from_cpu.GetType();
            TypeReference mono_type = system_type.ToMonoTypeReference().RewriteMonoTypeReference();

            DCToBclValueAux(from_cpu, system_type, mono_type, address);
        }

        private unsafe void DCToBclValueAux(object from_cpu, Type system_type, TypeReference mono_type, void* address)
        {
            if (mono_type.FullName.Equals("System.Object"))
            {
                throw new Exception("Type is System.Object, but I don't know what to represent it as.");
            }
            if (mono_type.FullName.Equals("System.Int16"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.Int32"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.Int64"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.UInt16"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.UInt32"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.UInt64"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.IntPtr"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.Boolean"))
            {
                bool v = (bool)from_cpu;
                System.Byte v2 = (System.Byte)(v ? 1 : 0);
                Marshal.StructureToPtr(v2, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.Char"))
            {
                Char v = (Char)from_cpu;
                System.UInt16 v2 = (System.UInt16)v;
                Marshal.StructureToPtr(v2, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.Single"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (mono_type.FullName.Equals("System.Double"))
            {
                Marshal.StructureToPtr(from_cpu, (IntPtr)address, false);
                return;
            }
            if (system_type.IsEnum)
            {
                var bas = system_type.BaseType;
                var fields = system_type.GetFields();
                if (fields == null)
                    throw new Exception("Cannot convert " + system_type.Name);
                if (fields.Count() == 0)
                    throw new Exception("Cannot convert " + system_type.Name);
                var field = fields[0];
                if (field == null)
                    throw new Exception("Cannot convert " + system_type.Name);
                var field_type = field.FieldType;
                if (field_type == null)
                    throw new Exception("Cannot convert " + system_type.Name);
                // Cast to base type and call recursively to solve.
                var v = Convert.ChangeType(from_cpu, field_type);
                DCToBclValue(v, address);
                return;
            }
            if (mono_type.FullName.Equals("System.String"))
            {
                // When allocated, BCL sets up the string. So, nothing to do here,
                // no further copying of from_cpu.
                return;
            }
            if (mono_type.IsArray)
            {
                var array = from_cpu as Array;
                var etype = array.GetType().GetElementType().ToMonoTypeReference().RewriteMonoTypeReference();
                var bcl_etype = RUNTIME.GetBclType(etype);
                uint[] lengths = new uint[array.Rank];
                for (int i = 0; i < array.Rank; ++i) lengths[i] = (uint)array.GetLength(i);
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

                var destIntPtr = (byte*)address;
                byte* df_ptr = destIntPtr;
                byte* df_rank = df_ptr + BUFFERS.SizeOf(typeof(IntPtr));
                byte* df_length = df_rank + BUFFERS.SizeOf(typeof(Int64));
                byte* df_elements = df_length + BUFFERS.SizeOf(typeof(Int64)) * rank;
                Cp(df_ptr, (IntPtr)df_elements); // Copy df_elements to *df_ptr
                Cp(df_rank, rank);
                for (int i = 0; i < rank; ++i)
                    Cp(df_length + i * BUFFERS.SizeOf(typeof(Int64)), a.GetLength(i));
                System.Type orig_element_type = from_cpu.GetType().GetElementType();
                var to_element_mono_type = orig_element_type.ToMonoTypeReference().RewriteMonoTypeReference();
                bool is_ref = to_element_mono_type.IsReferenceType();
                if (is_ref)
                {
                    orig_element_type = typeof(IntPtr);
                    to_element_mono_type = orig_element_type.ToMonoTypeReference().RewriteMonoTypeReference();
                }

                byte* ip = df_elements;

                // As the array could be multi-dimensional, we need to do a copy in row major order.
                // This is essentially the same as doing a number conversion to a string and vice versa
                // over the total number of elements in the entire multi-dimensional array.
                // See https://stackoverflow.com/questions/7123490/how-compiler-is-converting-integer-to-string-and-vice-versa
                // https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays/
                long total_size = 1;
                for (int i = 0; i < a.Rank; ++i) total_size *= a.GetLength(i);
                for (int i = 0; i < total_size; ++i)
                {
                    Int64[] index = new Int64[a.Rank];
                    int c = i;
                    for (int j = a.Rank - 1; j >= 0; --j)
                    {
                        int ind_size = a.GetLength(j);
                        var remainder = c % ind_size;
                        c = c / a.GetLength(j);
                        index[j] = remainder;
                    }
                    var from_element_value = a.GetValue(index);
                    // Note individual elements are copied here, but for reference types,
                    // the reference value is placed in the array.
                    IntPtr mem;
                    fixed (Int64* inds = index)
                    {
                        RUNTIME.BclSystemArrayLoadElementIndicesAddress((IntPtr)address, (IntPtr)inds, (IntPtr)(&mem));
                    }
                    if (is_ref)
                        from_element_value = DCToBcl(from_element_value);
                    DCToBclValueAux(from_element_value, orig_element_type, to_element_mono_type, (void*)mem);
                }

                RUNTIME.BclCheckHeap();
                return;
            }
            if (mono_type.IsStruct() || mono_type.IsReferenceType())
            {
                var bcl_type = RUNTIME.BclHeapGetType((IntPtr)address);
                if (bcl_type == IntPtr.Zero) throw new Exception();
                IntPtr[] fields = null;
                IntPtr* buf;
                int len;
                RUNTIME.BclGetFields(bcl_type, &buf, &len);
                fields = new IntPtr[len];
                for (int i = 0; i < len; ++i) fields[i] = buf[i];
                var mono_fields = mono_type.ResolveFields().ToArray();

                // Copy fields.
                for (int i = 0; i < fields.Length; ++i)
                {
                    var f = fields[i];
                    var ptrName = RUNTIME.BclGetFieldName(f);
                    string name = Marshal.PtrToStringAnsi(ptrName);
                    var fBclType = RUNTIME.BclGetFieldType(f);
                    var find = mono_fields.Where(t => t.Name == name);
                    var mono_field_reference = find.FirstOrDefault();
                    if (mono_field_reference == null) continue;
                    // Simple name matching does not work because the meta is sufficiently different
                    // between the framework used and the BCL framework. For example
                    // in List<>, BCL names a field "_items", but Net Framework the field
                    // "items". There is no guarentee that the field is in the same order
                    // either.
                    var fit = system_type.GetField(mono_field_reference.Name,
                        System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                        | System.Reflection.BindingFlags.Public
                        | System.Reflection.BindingFlags.Static
                    );
                    if (fit == null)
                    {
                        fit = system_type.GetField("_" + mono_field_reference.Name,
                            System.Reflection.BindingFlags.Instance
                            | System.Reflection.BindingFlags.NonPublic
                            | System.Reflection.BindingFlags.Public
                            | System.Reflection.BindingFlags.Static
                        );
                    };
                    if (fit == null)
                    {
                        if (Campy.Utils.Options.IsOn("copy_trace"))
                            System.Console.WriteLine("Unknown field in source " + mono_field_reference.Name + ". Ignoring.");
                        continue;
                    }
                    object field_value = fit.GetValue(from_cpu);
                    if (field_value != null && Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copying field " + field_value);
                    var mono_field_type = mono_field_reference.FieldType;
                    mono_field_type = mono_field_type.RewriteMonoTypeReference();
                    var fPtr = RUNTIME.BclGetField((IntPtr)address, f);
                    // In the case of a pointer, you have to deref the field.
                    var oPtr = fPtr;
                    void* ip = (void*)oPtr;
                    if (mono_field_type.IsReferenceType())
                        field_value = DCToBcl(field_value);
                    DCToBclValue(field_value, ip);
                    var field_size = SizeOfRefOrValType(mono_field_reference.FieldType);
                }

                return;
            }
            throw new Exception("Rotten apples");
        }


        private unsafe object DCToCpu(void* address)
        {
            // Find managed object in BCL and copy to CPU.
            // I need to know what type I'm copying from to allocate a managed
            // object on CPU side.
            if (address == (void*)0) return null;
            var bcl_type_of_object = RUNTIME.BclHeapGetType((IntPtr)address);
            if (bcl_type_of_object == IntPtr.Zero) return null;
            var mono_type_of_bcl_type = RUNTIME.GetMonoTypeFromBclType(bcl_type_of_object);
            var list = _allocated_objects.Where(t =>
            {
                if (t.Value == (IntPtr) address) return true;
                return false;
            });
            object cpu = null;
            if (!list.Any())
            {
                var from_type = RUNTIME.GetMonoTypeFromBclType(bcl_type_of_object);
                var type = from_type.ToSystemType();
                cpu = Activator.CreateInstance(type);
                _allocated_objects[cpu] = (IntPtr)address;
            }
            else
            {
                cpu = list.First().Key;
            }

            if (cpu == null) throw new Exception("Copy to CPU object is null.");
            if (_delayed_from_gpu.Contains(cpu))
            {
                if (Campy.Utils.Options.IsOn("copy_trace"))
                    System.Console.WriteLine("Not copying to CPU "
                                             + cpu
                                             + " "
                                             + RuntimeHelpers.GetHashCode(cpu)
                                             + " because it is delayed.");
                return cpu;
            }
            if (_never_copy_from_gpu.Contains(cpu))
            {
                if (Campy.Utils.Options.IsOn("copy_trace"))
                    System.Console.WriteLine("Not copying to CPU "
                                             + cpu
                                             + " "
                                             + RuntimeHelpers.GetHashCode(cpu)
                                             + " because it never copied to GPU.");
                return cpu;
            }
            if (_copied_from_gpu.Contains(cpu))
            {
                if (Campy.Utils.Options.IsOn("copy_trace"))
                    System.Console.WriteLine("Not copying to CPU "
                                             + cpu
                                             + " "
                                             + RuntimeHelpers.GetHashCode(cpu)
                                             + " because it was copied back before.");
                return cpu;
            }

            var type_of_cpu = cpu.GetType();
            if (Campy.Utils.Options.IsOn("copy_trace"))
                System.Console.WriteLine("Copying to CPU "
                                         + cpu
                                         + " "
                                         + "of type "
                                         + type_of_cpu.FullName
                                         + " because it was copied to GPU.");


            DCtoCpuRefValue(address, cpu, null);

            return cpu;
        }

        private unsafe void DCtoCpuRefValue(void* address, object target, FieldInfo target_field)
        {
            if (target_field == null)
            {
                System.Type system_type = target.GetType();
                var mono_type = system_type.ToMonoTypeReference().RewriteMonoTypeReference();
                var bcl_type = RUNTIME.BclHeapGetType((IntPtr)address);
                if (bcl_type == IntPtr.Zero) throw new Exception();
                if (system_type.IsArray)
                {
                    if ((IntPtr)address == IntPtr.Zero)
                        throw new Exception("from address null.");

                    if (target == null)
                        throw new Exception("target null.");

                    long* long_ptr = (long*)((long)(byte*)address);
                    long_ptr++;
                    int rank = (int)*long_ptr++;
                    System.Type to_element_type = system_type.GetElementType();
                    System.Type from_element_type = to_element_type;
                    if (to_element_type.IsArray || to_element_type.IsClass)
                        from_element_type = typeof(IntPtr);
                    int[] dims = new int[rank];
                    for (int kk = 0; kk < rank; ++kk)
                        dims[kk] = (int)*long_ptr++;
                    Array to_array = (Array) target;
                    CpArraytoCpu((void*)address, to_array, from_element_type);
                    return;
                }
                if (system_type.FullName.Equals("System.String"))
                {
                    return;

                    // For now, assume data exists on GPU. Perform memcpy using CUDA.
                    int* block = stackalloc int[1];
                    IntPtr intptr = new IntPtr(block);
                    var res = Cuda.cuMemcpyDtoH_v2(intptr, (IntPtr)address, sizeof(int));
                    int len = *block;
                    short* block2 = stackalloc short[len + 1];
                    var intptr2 = new IntPtr(block2);
                    Cuda.cuMemcpyDtoH_v2(intptr2, (IntPtr)address + sizeof(int), (uint)len * sizeof(short));
                    block2[len] = 0;
                    var o = new string((char*)intptr2);
                    if (Campy.Utils.Options.IsOn("copy_trace"))
                        System.Console.WriteLine("Copy from GPU " + o);
                    target_field.SetValue(target, o);
                    return;
                }
                if (system_type.IsClass)
                {
                    IntPtr[] fields = null;
                    IntPtr* buf;
                    int len;
                    RUNTIME.BclGetFields(bcl_type, &buf, &len);
                    fields = new IntPtr[len];
                    for (int i = 0; i < len; ++i) fields[i] = buf[i];
                    var mono_fields = mono_type.ResolveFields().ToArray();

                    // Copy fields.
                    for (int i = 0; i < fields.Length; ++i)
                    {
                        var f = fields[i];
                        var ptrName = RUNTIME.BclGetFieldName(f);
                        string name = Marshal.PtrToStringAnsi(ptrName);
                        var fBclType = RUNTIME.BclGetFieldType(f);
                        var find = mono_fields.Where(t => t.Name == name);
                        var mono_field_reference = find.FirstOrDefault();
                        if (mono_field_reference == null) continue;

                        string system_type_field_name = mono_field_reference.Name;
                        var fit = system_type.GetField(system_type_field_name,
                            System.Reflection.BindingFlags.Instance
                            | System.Reflection.BindingFlags.NonPublic
                            | System.Reflection.BindingFlags.Public
                            | System.Reflection.BindingFlags.Static
                        );
                        if (fit == null)
                        {
                            system_type_field_name = "_" + system_type_field_name;
                            fit = system_type.GetField(system_type_field_name,
                                System.Reflection.BindingFlags.Instance
                                | System.Reflection.BindingFlags.NonPublic
                                | System.Reflection.BindingFlags.Public
                                | System.Reflection.BindingFlags.Static
                            );
                        }
                        if (fit == null)
                        {
                            if (Campy.Utils.Options.IsOn("copy_trace"))
                                System.Console.WriteLine("Unknown field in source " + mono_field_reference.Name + ". Ignoring.");
                            continue;
                        }
                        // Get from BCL the address of the field.
                        var mono_field_type = mono_field_reference.FieldType;
                        mono_field_type = mono_field_type.RewriteMonoTypeReference();
                        var fPtr = (void*) RUNTIME.BclGetField((IntPtr) address, f);
                        object field_value = null;
                        DCtoCpuRefValue(fPtr, target, fit);
                        if (field_value != null && Campy.Utils.Options.IsOn("copy_trace"))
                            System.Console.WriteLine("Copying field " + field_value);
                    }

                    return;
                }
                throw new Exception("Unknown type.");
            }

            {
                System.Type system_type = target_field.FieldType;
                var mono_type = system_type.ToMonoTypeReference().RewriteMonoTypeReference();
                if (system_type.FullName.Equals("System.Object"))
                {
                    if (address == null)
                    {
                        target_field.SetValue(target, null);
                        return;
                    }

                    var v = DCToCpu(address);
                    target_field.SetValue(target, v);
                    return;
                }
                if (system_type.FullName.Equals("System.Int16"))
                {
                    object o = Marshal.PtrToStructure<System.Int16>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.Int32"))
                {
                    object o = Marshal.PtrToStructure<System.Int32>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.Int64"))
                {
                    object o = Marshal.PtrToStructure<System.Int64>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.UInt16"))
                {
                    object o = Marshal.PtrToStructure<System.UInt16>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.UInt32"))
                {
                    object o = Marshal.PtrToStructure<System.UInt32>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.UInt64"))
                {
                    object o = Marshal.PtrToStructure<System.UInt64>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.IntPtr"))
                {
                    object o = Marshal.PtrToStructure<System.IntPtr>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.Boolean"))
                {
                    byte v = *(byte*) address;
                    bool w = v == 0 ? false : true;
                    try
                    {
                        target_field.SetValue(target, w);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.Char"))
                {

                    var c = *(short*) address;
                    var to_cpu = (Char) c;
                    try
                    {
                        target_field.SetValue(target, to_cpu);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.Single"))
                {
                    object o = Marshal.PtrToStructure<System.Single>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.Double"))
                {
                    object o = Marshal.PtrToStructure<System.Double>((IntPtr) address);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.IsEnum)
                {
                    object o = null;
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.FullName.Equals("System.String"))
                {
                    var ip = *(void**)address;
                    var o = DCToCpu(ip);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.IsArray)
                {
                    var ip = * (void**)address;
                    var o = DCToCpu(ip);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
                if (system_type.IsClass)
                {
                    var ip = *(void**) address;
                    var o = DCToCpu(ip);
                    try
                    {
                        target_field.SetValue(target, o);
                    }
                    catch
                    {
                    }
                    return;
                }
            }
            throw new Exception("Unknown type.");
        }


        private unsafe void CpArraytoCpu(void* from_gpu, Array to_cpu, System.Type from_element_type)
        {
            var to_type = to_cpu.GetType();
            if (!to_type.IsArray)
                throw new Exception("Expecting array.");
            Type to_element_type = to_cpu.GetType().GetElementType();
            TypeReference to_element_mono_type = to_element_type.ToMonoTypeReference().RewriteMonoTypeReference();
            for (int i = 0; i < to_cpu.Length; ++i)
            {
                Int64[] index = new Int64[to_cpu.Rank];
                int c = i;
                for (int j = to_cpu.Rank - 1; j >= 0; --j)
                {
                    int ind_size = to_cpu.GetLength(j);
                    var remainder = c % ind_size;
                    c = c / to_cpu.GetLength(j);
                    index[j] = remainder;
                }
                IntPtr mem;
                fixed (Int64* inds = index)
                {
                    RUNTIME.BclSystemArrayLoadElementIndicesAddress((IntPtr) from_gpu, (IntPtr) inds, (IntPtr) (&mem));
                }
                if (to_element_mono_type.IsReferenceType())
                {
                    object obj = Marshal.PtrToStructure((IntPtr)mem, typeof(IntPtr));
                    IntPtr obj_intptr = (IntPtr)obj;
                    var ob2 = DCToCpu((void*)obj_intptr);
                    to_cpu.SetValue(ob2, index);
                    mem = new IntPtr((long)mem + SizeOf(typeof(IntPtr)));
                }
                else
                {
                    if (to_element_type.FullName.Equals("System.Int16"))
                    {
                        object o = Marshal.PtrToStructure<System.Int16>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                        return;
                    }
                    if (to_element_type.FullName.Equals("System.Int32"))
                    {
                        object o = Marshal.PtrToStructure<System.Int32>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.Int64"))
                    {
                        object o = Marshal.PtrToStructure<System.Int64>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.UInt16"))
                    {
                        object o = Marshal.PtrToStructure<System.UInt16>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.UInt32"))
                    {
                        object o = Marshal.PtrToStructure<System.UInt32>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.UInt64"))
                    {
                        object o = Marshal.PtrToStructure<System.UInt64>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.IntPtr"))
                    {
                        object o = Marshal.PtrToStructure<System.IntPtr>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.Boolean"))
                    {
                        object o = Marshal.PtrToStructure<byte>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.Char"))
                    {

                        object o = Marshal.PtrToStructure<short>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.Single"))
                    {
                        object o = Marshal.PtrToStructure<System.Single>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                    if (to_element_type.FullName.Equals("System.Double"))
                    {
                        object o = Marshal.PtrToStructure<System.Double>(mem);
                        try
                        {
                            to_cpu.SetValue(o, index);
                        }
                        catch
                        {
                        }
                    }
                }
            }
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

//            {
//                // Allocate Unified Memory.
//                var size = bytes;
//                var res = Cuda.cuMemAllocManaged(out IntPtr pointer, (uint)size, (uint)Swigged.Cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL);
//                if (Campy.Utils.Options.IsOn("memory_trace"))
//                    System.Console.WriteLine("Cu Alloc (" + bytes + " bytes) " + pointer);
//                Utils.CudaHelpers.CheckCudaError(res);
//                return pointer;
//            }

            //if (false)
            //{
            //    return Marshal.AllocHGlobal(bytes);
            //}
        }

        public IntPtr New(object obj)
        {
            if (obj == null)
                return IntPtr.Zero;

            Type type = obj.GetType();
            Mono.Cecil.ModuleDefinition campy_bcl_runtime = Mono.Cecil.ModuleDefinition.ReadModule(RUNTIME.FindCoreLib());
            TypeReference substituted_type = type.SubstituteMonoTypeReference(campy_bcl_runtime);

            if (type.FullName == "System.String")
            {
                IntPtr result;
                unsafe
                {
                    var str = obj as string;
                    var len = str.Length;
                    RUNTIME.BclCheckHeap();
                    fixed (char* chars = str)
                    {
                        result = RUNTIME.BclAllocString(len, (IntPtr)chars);
                    }
                }
                RUNTIME.BclCheckHeap();
                return result;
            }

            if (type.IsArray)
            {
                var array = obj as Array;
                Type etype = array.GetType().GetElementType();
                RUNTIME.BclCheckHeap();
                var bcl_type = RUNTIME.GetBclType(etype.ToMonoTypeReference());
                RUNTIME.BclCheckHeap();

                uint[] lengths = new uint[array.Rank];
                for (int i = 0; i < array.Rank; ++i) lengths[i] = (uint)array.GetLength(i);
                RUNTIME.BclCheckHeap();
                IntPtr result = RUNTIME.BclArrayAlloc(bcl_type, array.Rank, lengths);
                RUNTIME.BclCheckHeap();
                var bcl_type_of_object = RUNTIME.BclHeapGetType((IntPtr)result);
                return result;
            }

            {
                var bcl_type = RUNTIME.GetBclType(type.ToMonoTypeReference());
                RUNTIME.BclCheckHeap();
                IntPtr result = RUNTIME.BclHeapAlloc(bcl_type);
                RUNTIME.BclCheckHeap();
                return result;
            }
        }

        public void Free(IntPtr pointer)
        {
            // There are two buffer types: one for straight unmanaged buffers shared between CPU and GPU,
            // and the other for BCL managed object. We need to nuke each appropriately.
            ulong p = (ulong) pointer;
            ulong l = (ulong) RUNTIME.BclPtr;
            ulong h = l + RUNTIME.BclPtrSize;
            if (p >= l && p < h)
                return;

            if (Campy.Utils.Options.IsOn("memory_trace"))
                System.Console.WriteLine("Cu Free {0:X}", pointer.ToInt64());
            var res = Cuda.cuMemFreeHost(pointer);
            CudaHelpers.CheckCudaError(res);
            //Marshal.FreeHGlobal(pointerToUnmanagedMemory);
        }

        public static void Cp(IntPtr destPtr, IntPtr srcPtr, int size)
        {
            unsafe
            {
                // srcPtr and destPtr are IntPtr's pointing to valid memory locations
                // size is the number of bytes to copy
                RUNTIME.BclCheckHeap();
                byte* src = (byte*)srcPtr;
                byte* dest = (byte*)destPtr;
                for (int i = 0; i < size; i++)
                {
                    dest[i] = src[i];
                }
                RUNTIME.BclCheckHeap();
            }
        }

        private static unsafe void Cp(void* destPtr, object src)
        {
            Marshal.StructureToPtr(src, (IntPtr)destPtr, false);
        }

        public static string Indent(int size, string value)
        {
            var strArray = value.Split('\r');
            var sb = new StringBuilder();
            foreach (var s in strArray)
                sb.Append(new string(' ', size)).Append(s.Replace("\n", "\r\n"));
            return sb.ToString();
        }

        public static string PrintCpuObject(int level, object obj)
        {
            if (obj == null)
                return Indent(level, "null" + Environment.NewLine);

            var type = obj.GetType();
            var sb = new StringBuilder();
            var s = Indent(level, type.Name + ":");
            sb.Append(s + Environment.NewLine);

            if (type.IsValueType && !type.IsStruct())
            {
                sb.Append(Indent(level + 2, obj.ToString()) + Environment.NewLine);
                return sb.ToString();
            }

            if (type.IsArray)
            {
                var a = obj as Array;
                var rank = a.Rank;
                long total_size = 1;
                for (int i = 0; i < rank; ++i)
                    total_size *= a.GetLength(i);
                for (int i = 0; i < total_size; ++i)
                {
                    Int64[] index = new Int64[rank];
                    int c = i;
                    for (int j = rank - 1; j >= 0; --j)
                    {
                        int ind_size = a.GetLength(j);
                        var remainder = c % ind_size;
                        c = c / a.GetLength(j);
                        index[j] = remainder;
                    }
                    var v = a.GetValue(index);
                    sb.Append(Indent(level + 2, i.ToString() + ":" + PrintCpuObject(level + 2, v)));
                }
                return sb.ToString();
            }
            else
            {
                // Print out value as string.
                var object_type = obj.GetType();
                var methods = object_type.GetMethods(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance);
                var fields = object_type.SafeFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance);
                foreach (var f in fields)
                {
                    var ft = f.FieldType;
                    if (ft.BaseType == typeof(MulticastDelegate))
                        continue;
                    var v = f.GetValue(obj);
                    sb.Append(Indent(level + 2, f.Name.ToString() + ":" + PrintCpuObject(level + 2, v)));
                }
                return sb.ToString();
            }
        }

        public static string PrintBclObject(int level, IntPtr obj, TypeReference expected_bcl_type)
        {
            var sb = new StringBuilder();
            var s = Indent(level, expected_bcl_type.Name + " at 0x" + obj.ToInt64().ToString("X8") + ":");
            sb.Append(s + Environment.NewLine);

            if (expected_bcl_type.IsValueType)
            {
                if (!expected_bcl_type.IsStruct())
                {
                    var sys_type = expected_bcl_type.ToSystemType();
                    var o = Marshal.PtrToStructure(obj, sys_type);
                    sb.Append(Indent(level + 2, o.ToString()) + Environment.NewLine);
                    return sb.ToString();
                }
                else
                {
                    throw new Exception("unhandled");
                }
            }
            else
            {
                // Reference type.
                // Look up type in BCL of object pointer.
                var bcl_type = RUNTIME.BclHeapGetType(obj);
                if (bcl_type == IntPtr.Zero) return sb.ToString();
                if (expected_bcl_type.IsArray)
                {
                    var et = expected_bcl_type.ResolveGetElementType();
                    uint rank = (uint)RUNTIME.BclSystemArrayGetRank(obj);
                    IntPtr len_ptr = RUNTIME.BclSystemArrayGetDims(obj);
                    unsafe
                    {
                        long total_size = 1;
                        long * lens = (long*)len_ptr;
                        for (int i = 0; i < rank; ++i) total_size *= lens[i];
                        for (int i = 0; i < total_size; ++i)
                        {
                            Int64[] index = new Int64[rank];
                            long c = i;
                            for (int j = (int)rank - 1; j >= 0; --j)
                            {
                                long ind_size = lens[j];
                                long remainder = c % ind_size;
                                c = c / lens[j];
                                index[j] = remainder;
                            }
                            fixed (long* inds = index)
                            {
                                void* address;
                                RUNTIME.BclSystemArrayLoadElementIndicesAddress(obj, (IntPtr)inds, (IntPtr)(& address));
                                // In the case of a pointer, you have to deref the field.
                                IntPtr fPtr = (IntPtr)address;
                                var oPtr = fPtr;
                                if (!et.IsValueType)
                                    oPtr = (IntPtr)Marshal.PtrToStructure(fPtr, typeof(IntPtr));
                                sb.Append(Indent(level + 2,
                                    i.ToString()
                                    + " at 0x" + oPtr.ToInt64().ToString("X8")
                                    + ":"
                                    + PrintBclObject(level + 2, oPtr, et)));
                            }
                        }
                    }

                    return sb.ToString();
                }
                else
                {
                    if (bcl_type == IntPtr.Zero) return sb.ToString();

                    // Get all bcl fields of class object.
                    IntPtr[] fields = null;
                    unsafe
                    {
                        IntPtr* buf;
                        int len;
                        RUNTIME.BclGetFields(bcl_type, &buf, &len);
                        fields = new IntPtr[len];
                        for (int i = 0; i < len; ++i) fields[i] = buf[i];
                    }
                    // Get all Mono fields of class object.
                    var mono_fields = expected_bcl_type.ResolveFields().ToArray();
                    // Match up and print out.
                    for (int i = 0; i < fields.Length; ++i)
                    {
                        var f = fields[i];
                        var field_definition = mono_fields[i].Resolve();
                        if (field_definition != null && field_definition.IsStatic) continue;
                        var mono_field_type = mono_fields[i].FieldType;
                        mono_field_type = mono_field_type.RewriteMonoTypeReference();
                        if (mono_field_type.FullName == "Campy.SimpleKernel")
                            continue;
                        if (mono_field_type.IsSubclassOf(typeof(MulticastDelegate).ToMonoTypeReference()))
                            continue;
                        var ptrName = RUNTIME.BclGetFieldName(f);
                        string name = Marshal.PtrToStringAnsi(ptrName);
                        var fBclType = RUNTIME.BclGetFieldType(f);
                        var find = mono_fields.Where(t => t.Name == name);
                        var fPtr = RUNTIME.BclGetField(obj, f);
                        // In the case of a pointer, you have to deref the field.
                        var oPtr = fPtr;
                        if (!mono_field_type.IsValueType)
                            oPtr = (IntPtr)Marshal.PtrToStructure(fPtr, typeof(IntPtr));
                        sb.Append(Indent(level + 2,
                            name
                            + " at 0x" + oPtr.ToInt64().ToString("X8")
                            + ":"
                            + PrintBclObject(level + 2, oPtr, mono_field_type)));
                    }

                    return sb.ToString();
                }
            }
            throw new Exception("unhandled");
        }
    }
}




