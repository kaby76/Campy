using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Swigged.Cuda;
using Swigged.LLVM;
using Campy.Utils;
using Mono.Cecil;
using Mono.Cecil.Cil;
using MethodImplAttributes = Mono.Cecil.MethodImplAttributes;

namespace Campy.Compiler
{
    public class RUNTIME
    {
        // This table encodes runtime type information for rewriting BCL types. Use this to determine
        // what a type (represented in Mono.Cecil.TypeReference) in the user's program maps to
        // in the GPU base class layer (also represented in Mono.Cecil.TypeReference).
        private static Dictionary<TypeReference, TypeReference> _substituted_bcl = new Dictionary<TypeReference, TypeReference>();

        // We also need to convert system types that are in the GPU BCL types (represented in Mono.Cecil). Unfortunately,
        // NET Core seems to have problems with System.Reflection.
        private static Dictionary<System.Type, TypeReference> _system_type_to_mono_type_for_bcl = new Dictionary<System.Type, TypeReference>();
 
        // Some methods references resolve to null. And, some methods we might want to substitute a
        // different implementation that the one normally found through reference Resolve(). Retain a
        // mapping of methods to be rewritten.
        private static Dictionary<string, string> _rewritten_runtime = new Dictionary<string, string>();

        private static Dictionary<string, TypeRef> _ptx_type_to_llvm_typeref = new Dictionary<string, TypeRef>()
        {
            {".b8", LLVM.Int8Type()},
            {".b16", LLVM.Int16Type()},
            {".b32", LLVM.Int32Type()},
            {".b64", LLVM.Int64Type()},
        };

        public RUNTIME()
        {
        }

        // Arrays are implemented as a struct, with the data following the struct
        // in row major format. Note, each dimension has a length that is recorded
        // following the pointer p. The one shown here is for only one-dimensional
        // arrays.
        // Calls have to be casted to this type.
        public unsafe struct A
        {
            public void* p;
            public long d;

            public long l; // Width of dimension 0.
            // Additional widths for dimension 1, dimension 2, ...
            // Value data after all dimensional sizes.
        }

        public static unsafe int get_length_multi_array(A* arr, int i0)
        {
            byte* bp = (byte*)arr;
            bp = bp + 16 + 8 * i0;
            long* lp = (long*)bp;
            return (int)*lp;
        }

        public static unsafe int get_multi_array(A* arr, int i0)
        {
            int* a = *(int**)arr;
            return *(a + i0);
        }

        public static unsafe int get_multi_array(A* arr, int i0, int i1)
        {
            // (y * xMax) + x
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            byte* d0_ptr = (byte*)arr;
            d0_ptr = d0_ptr + 24;
            long o = 0;
            long d0 = *(long*)d0_ptr;
            o = i0 * d0 + i1;
            return *(a + o);
        }

        public static unsafe int get_multi_array(A* arr, int i0, int i1, int i2)
        {
            // (z * xMax * yMax) + (y * xMax) + x;
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            byte* bp_d0 = (byte*)arr;
            byte* bp_d1 = (byte*)arr;
            bp_d1 = bp_d1 + 24;
            long o = 0;
            long* lp_d1 = (long*)bp_d1;
            byte* bp_d2 = bp_d1 + 8;
            long* lp_d2 = (long*)bp_d2;
            o = (*lp_d1) * i0 + i1;
            return *(a + o);
        }

        public static unsafe void set_multi_array(A* arr, int i0, int value)
        {
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            long o = i0;
            *(a + o) = value;
        }

        public static unsafe void set_multi_array(A* arr, int i0, int i1, int value)
        {
            //  b[i, j] = j  + i * ex[1];

            int* a = (int*)(*arr).p;
            long ex1 = *(long*)(24 + (byte*)arr);
            long o = i1 + ex1 * i0;
            *(a + o) = value;
        }

        public static unsafe void set_multi_array(A* arr, int i0, int i1, int i2, int value)
        {
            //  b[i, j, k] = k + j * ex[2] + i * ex[2] * ex[1];

            int* a = (int*)(*arr).p;
            long ex1 = *(long*)(24 + (byte*)arr);
            long ex2 = *(long*)(32 + (byte*)arr);
            long o = i2 + i1 * ex2 + i0 * ex2 * ex1;
            *(a + o) = value;
        }

        public static void ThrowArgumentOutOfRangeException()
        {
        }

        public class BclNativeMethod
        {
            public TypeReference _bcl_type;
            public MethodDefinition _md;
            public string _nameSpace;
            public string _type;
            public string _full_name;
            public string _short_name;
            public string _native_name;
            public TypeReference _returnType;
            public List<Mono.Cecil.ParameterDefinition> _parameterTypes;

            public BclNativeMethod(TypeReference bcl_type, MethodDefinition md)
            {
                _bcl_type = bcl_type;
                _md = md;
                _nameSpace = bcl_type.Namespace;
                _type = bcl_type.FullName;
                _full_name = md.FullName;
                _short_name = md.Name;
                _returnType = md.ReturnType;
                _parameterTypes = md.Parameters.ToList();
                // Unfortunately, I don't know the C++ name decoration rules in the NVCC compiler. Further,
                // DotNetAnywhere originally didn't implement all the "internal call"-labeled attributed methods in Corlib.
                // Further, the only table that does make note of the internal call-labeled methods was in C. So,
                // for Campy, the BCL was extended with another attribute, GPUBCLAttribute, to indicate the
                // name of the native call, making it visible to C#. The following code grabs and caches this information.
                var cust_attrs = md.CustomAttributes;
                if (cust_attrs.Count > 0)
                {
                    var a = cust_attrs.First();
                    if (a.AttributeType.FullName == "System.GPUBCLAttribute")
                    {
                        var arg = a.ConstructorArguments.First();
                        var v = arg.Value;
                        var s = (string)v;
                        _native_name = s;
                        //string mangled_name = "_Z" + _native_name.Length + _native_name + "PhS_S_";
                        //CampyConverter.built_in_functions.Add(mangled_name,
                        //    LLVM.AddFunction(
                        //        CampyConverter.global_llvm_module,
                        //        mangled_name,
                        //        LLVM.FunctionType(LLVM.Int64Type(),
                        //            new TypeRef[]
                        //            {
                        //                    LLVM.PointerType(LLVM.VoidType(), 0), // "this"
                        //                    LLVM.PointerType(LLVM.VoidType(), 0), // params in a block.
                        //                    LLVM.PointerType(LLVM.VoidType(), 0) // return value block.
                        //            }, false)));
                    }
                }
            }
        }

        public class PtxFunction
        {
            public string _mangled_name;
            public string _short_name;
            public ValueRef _valueref;

            public PtxFunction(string mangled_name, ValueRef value_ref)
            {
                _mangled_name = mangled_name;
                _valueref = value_ref;
            }
        }

        // This table encodes runtime type information for rewriting internal calls in the native portion of
        // the BCL for the GPU. It was originally encoded in dna/internal.c. However, it's easier and
        // safer to derive the information from the C# portion of the BCL using System.Reflection.
        //
        // Why is this information needed? In Inst.c, I need to make a call of a function to the runtime.
        // I only have PTX files, which removes the type information from the signature of
        // the original call (it is all three parameters of void*).
        private static List<BclNativeMethod> _internalCalls = new List<BclNativeMethod>();

        // This table is a record of all '.visible' functions in a generated PTX file. Use this name when calling
        // functions in PTX/LLVM.
        private static List<PtxFunction> _ptx_functions = new List<PtxFunction>();

        private class InternalCallEnumerable : IEnumerable<BclNativeMethod>
        {
            public InternalCallEnumerable()
            {
            }

            public IEnumerator<BclNativeMethod> GetEnumerator()
            {
                foreach (var key in _internalCalls)
                {
                    yield return key;
                }
            }

            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        public static IEnumerable<BclNativeMethod> BclNativeMethods
        {
            get
            {
                return new InternalCallEnumerable();
            }
        }

        public static IEnumerable<PtxFunction> PtxFunctions
        {
            get
            {
                return _ptx_functions;
            }
        }

        public static string FindNativeCoreLib()
        {
            try
            {
                // Let's try the obvious, in the same directory as Campy.Utils.dll.
                var path_of_campy = Campy.Utils.CampyInfo.PathOfCampy();
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "Campy.Runtime.Native.lib";
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Try something else...

            try
            {
                // Let's say it's in the Nuget packages directory. Go up a few levels and look for it in "contents" directory.
                // .../.nuget/packages/campy/0.0.4/lib/netstandard2.0/Campy.Utils.dll
                // =>
                // .../.nuget/packages/campy/0.0.4/contents/corlib.dll.
                var path_of_campy = Campy.Utils.CampyInfo.PathOfCampy();
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar
                                                       + ".." + Path.DirectorySeparatorChar
                                                       + ".." + Path.DirectorySeparatorChar
                                                       + "content" + Path.DirectorySeparatorChar
                                                       + "Campy.Runtime.Native.lib";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Try something else...

            try
            {
                // Let's try the calling executable directory.
                var dir = Path.GetDirectoryName(Path.GetFullPath(System.Reflection.Assembly.GetEntryAssembly().Location));
                string full_path_assem = dir + Path.DirectorySeparatorChar
                                                       + "Campy.Runtime.Native.lib";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Try something else...

            try
            {
                // This could be a unit test in Campy. If that is true, then look in the standard directory structure
                // for Campy source/object. It should have actually copied the damn corlib.dll to the output executable directory,
                // but someone set up the test wrong. Anyways, assume that the project is up to date, and load from Campy.Runtime.
                // ../../../../../Campy.Runtime/Corlib/bin/Debug/net20/
                var path_of_campy = @"../../../../../x64/Debug";
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "Campy.Runtime.Native.lib";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Fuck. I have no idea.
            return null;
        }

        public static string FindCoreLib()
        {
            try
            {
                // Let's try the obvious, in the same directory as Campy.Utils.dll.
                var path_of_campy = Campy.Utils.CampyInfo.PathOfCampy();
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "corlib.dll";
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Try something else...

            try
            {
                // Let's say it's in the Nuget packages directory. Go up a few levels and look for it in "contents" directory.
                // .../.nuget/packages/campy/0.0.4/lib/netstandard2.0/Campy.Utils.dll
                // =>
                // .../.nuget/packages/campy/0.0.4/contents/corlib.dll.
                var path_of_campy = Campy.Utils.CampyInfo.PathOfCampy();
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar
                                                       + ".." + Path.DirectorySeparatorChar
                                                       + ".." + Path.DirectorySeparatorChar
                                                       + "content" + Path.DirectorySeparatorChar
                                                       + "corlib.dll";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            try
            {
                // Let's try the calling executable directory.
                var dir = Path.GetDirectoryName(Path.GetFullPath(System.Reflection.Assembly.GetEntryAssembly().Location));
                string full_path_assem = dir + Path.DirectorySeparatorChar
                                                       + "corlib.dll";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Try something else...

            try
            {
                // This could be a unit test in Campy. If that is true, then look in the standard directory structure
                // for Campy source/object. It should have actually copied the damn corlib.dll to the output executable directory,
                // but someone set up the test wrong. Anyways, assume that the project is up to date, and load from Campy.Runtime.
                // ../../../../../Campy.Runtime/Corlib/bin/Debug/net20/
                var path_of_campy = @"../../../../../Campy.Runtime/Corlib/bin/Debug/net20";
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "corlib.dll";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception e)
            {
            }

            // Fuck. I have no idea.
            return null;
        }

        public static void Initialize()
        {
            // Load C# library for BCL, and grab all types and methods.
            // The tables that this method sets up are:
            // _substituted_bcl -- maps types in program (represented in Mono.Cecil) into GPU BCL types (represented in Mono.Cecil).
            // _system_type_to_mono_type_for_bcl -- associates types in GPU BCL with NET Core/NET Framework/... in user program.
            // Note, there seems to be an underlying bug in System.Type.GetType for certain generics, like System.Collections.Generic.HashSet.
            // The method returns null.
            var xx = typeof(System.Collections.Generic.HashSet<>);
            var x2 = typeof(System.Collections.Generic.HashSet<int>);
            var yy = System.Type.GetType("System.Collections.Generic.HashSet");
            var y2 = System.Type.GetType("System.Collections.Generic.HashSet<>");
            var y3 = System.Type.GetType("System.Collections.Generic.HashSet`1");
            var y4 = System.Type.GetType("System.Collections.Generic.HashSet<T>");
            var y5 = System.Type.GetType(xx.FullName);
            var y6 = System.Type.GetType(@"System.Collections.Generic.HashSet`1[System.Int32]");
            var y7 = System.Type.GetType(@"System.Collections.Generic.Dictionary`2[System.String,System.String]");
            var y8 = System.Type.GetType(x2.FullName);

            // Set up _substituted_bcl.
            var runtime = new RUNTIME();

            // Find corlib.dll. It could be anywhere, but let's check the usual spots.
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(FindCoreLib());
            foreach (var bcl_type in md.GetTypes())
            {
                // Filter out <Module> and <PrivateImplementationDetails>, among possible others.
                Regex regex = new Regex(@"^[<]\w+[>]");
                if (regex.IsMatch(bcl_type.FullName)) continue;

                // Try to map the type into native NET type. Some things just won't.
                var t_system_type = System.Type.GetType(bcl_type.FullName);
                if (t_system_type == null) continue;

                var to_mono = t_system_type.ToMonoTypeReference();

                // Add entry for converting intrinsic NET BCL type to GPU BCL type.
                _substituted_bcl.Add(to_mono, bcl_type);

                foreach (var m in bcl_type.Methods)
                {
                    var x = m.ImplAttributes;
                    if ((x & MethodImplAttributes.InternalCall) != 0)
                    {
                        if (Campy.Utils.Options.IsOn("runtime_trace"))
                            System.Console.WriteLine("Internal call set up " + bcl_type + " " + m);

                        _internalCalls.Add(new BclNativeMethod(bcl_type, m));
                    }
                }
            }

            // Set up _system_type_to_mono_type_for_bcl.
            // There really isn't any good way to set this up because NET Core System.Reflection does not work
            // on things like System.Int32. We will manually set up it here, checking then to see if we miss
            // something.

            // Parse PTX files for all "visible" functions, and create LLVM declarations.
            // For "Internal Calls", these functions appear here, but also on the _internalCalls list.
            var assembly = Assembly.GetAssembly(typeof(Campy.Compiler.RUNTIME));
            var resource_names = assembly.GetManifestResourceNames();
            foreach (var resource_name in resource_names)
            {
                using (Stream stream = assembly.GetManifestResourceStream(resource_name))
                using (StreamReader reader = new StreamReader(stream))
                {
                    string gpu_bcl_ptx = reader.ReadToEnd();
                    // Parse the PTX for ".visible" functions, and enter each in
                    // the runtime table.

                    // This should match ".visible" <spaces> ".func" <spaces> <function-return>? <function-name>
                    // over many lines, many times.
                    Regex regex = new Regex(
          @"\.visible\s+\.func\s+(?<return>[(][^)]*[)]\s+)?(?<name>\w+)\s*(?<params>[(][^)]*[)]\s+)");
                    foreach (Match match in regex.Matches(gpu_bcl_ptx))
                    {
                        Regex space = new Regex(@"\s\s+");
                        string mangled_name = match.Groups["name"].Value.Trim();
                        string return_type = match.Groups["return"].Value.Trim();
                        return_type = space.Replace(return_type, " ");
                        string parameters = match.Groups["params"].Value.Trim();
                        parameters = space.Replace(parameters, " ");

                        if (Campy.Utils.Options.IsOn("runtime_trace"))
                            System.Console.WriteLine(mangled_name + " " + return_type + " " + parameters);

                        if (JITER.functions_in_internal_bcl_layer.ContainsKey(mangled_name)) continue;

                        TypeRef llvm_return_type = default(TypeRef);
                        TypeRef[] args;

                        // Construct LLVM extern that corresponds to type of function.
                        // Parse return_type and parameters strings...
                        Regex param_regex = new Regex(@"\.param(\s+\.align\s+\d+)?\s+(?<type>\S+)\s");

                        {
                            // parse return.
                            if (return_type == "")
                                llvm_return_type = LLVM.VoidType();
                            else
                            {
                                foreach (Match ret in param_regex.Matches(return_type))
                                {
                                    var x = ret.Groups["type"].Value;
                                    _ptx_type_to_llvm_typeref.TryGetValue(
                                        x, out TypeRef y);
                                    if (Campy.Utils.Options.IsOn("runtime_trace"))
                                        System.Console.WriteLine("name " + x + "  value " + y.ToString());
                                    llvm_return_type = y;
                                }
                            }
                        }

                        {
                            // parse parameters.
                            List<TypeRef> args_list = new List<TypeRef>();
                            foreach (Match ret in param_regex.Matches(parameters))
                            {
                                var x = ret.Groups["type"].Value;
                                if (!_ptx_type_to_llvm_typeref.TryGetValue(
                                    x, out TypeRef y))
                                    throw new Exception("Unknown type syntax in ptx parameter.");
                                if (Campy.Utils.Options.IsOn("runtime_trace"))
                                {
                                    System.Console.Write("parameter ");

                                    System.Console.WriteLine("name " + x + "  value " + y.ToString());
                                }
                                args_list.Add(y);
                            }
                            args = args_list.ToArray();
                        }

                        var decl = LLVM.AddFunction(
                            JITER.global_llvm_module,
                            mangled_name,
                            LLVM.FunctionType(
                                llvm_return_type,
                                args,
                                false));

                        PtxFunction ptxf = new PtxFunction(mangled_name, decl);
                        if (Campy.Utils.Options.IsOn("runtime_trace"))
                            System.Console.WriteLine(ptxf._mangled_name
                                                 + " "
                                                 + ptxf._short_name
                                                 + " "
                                                 + ptxf._valueref);

                        JITER.functions_in_internal_bcl_layer.Add(mangled_name, decl);
                        _ptx_functions.Add(ptxf);
                    }
                }
            }
        }

        public static IntPtr GetMetaDataType(Mono.Cecil.TypeReference type)
        {
            IntPtr result = IntPtr.Zero;
            CUresult res = CUresult.CUDA_SUCCESS;

            // Get meta data from type from the GPU, as that is where it resides.
            // tMetaData* pTypeMetaData;
            // BCL_CLIFile_GetMetaDataForAssembly("ConsoleApp1.exe", &pTypeMetaDatanew FileStream();

            // Set up the type's assembly in file system.
            String assembly_location = Path.GetFullPath(type.Resolve().Module.FullyQualifiedName);
            string assem = Path.GetFileName(assembly_location);
            string full_path_assem = assembly_location;
            Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
            var corlib_bytes_handle_len = stream.Length;
            var corlib_bytes = new byte[corlib_bytes_handle_len];
            stream.Read(corlib_bytes, 0, (int)corlib_bytes_handle_len);
            var corlib_bytes_handle = GCHandle.Alloc(corlib_bytes, GCHandleType.Pinned);
            var corlib_bytes_intptr = corlib_bytes_handle.AddrOfPinnedObject();
            stream.Close();
            stream.Dispose();

            unsafe
            {
                // Set up parameters.
                int count = 4;
                IntPtr parm1; // Name of assembly.
                IntPtr parm2; // Contents
                IntPtr parm3; // Length
                IntPtr parm4; // result

                var ptr = Marshal.StringToHGlobalAnsi(assem);
                BUFFERS buffers = new BUFFERS();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr pointer2 = buffers.New((int)corlib_bytes_handle_len);
                BUFFERS.Cp(pointer2, corlib_bytes_intptr, (int)corlib_bytes_handle_len);
                IntPtr[] x2 = new IntPtr[] { pointer2 };
                GCHandle handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                parm2 = handle2.AddrOfPinnedObject();

                IntPtr[] x3 = new IntPtr[] { new IntPtr(corlib_bytes_handle_len) };
                GCHandle handle3 = GCHandle.Alloc(x3, GCHandleType.Pinned);
                parm3 = handle3.AddrOfPinnedObject();

                var pointer4 = buffers.New(sizeof(long));
                IntPtr[] x4 = new IntPtr[] { pointer4 };
                GCHandle handle4 = GCHandle.Alloc(x4, GCHandleType.Pinned);
                parm4 = handle4.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1, parm2, parm3, parm4 };

                CUmodule module = RUNTIME.RuntimeModule;
                CUfunction _Z16Bcl_Gfs_add_filePcS_yPi = RUNTIME._Z16Bcl_Gfs_add_filePcS_yPi(module);
                Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                    out Campy.Utils.CudaHelpers.dim3 tile_size,
                    out Campy.Utils.CudaHelpers.dim3 tiles);
                fixed (IntPtr* kernelParams = kp)
                {
                    res = Cuda.cuLaunchKernel(
                        _Z16Bcl_Gfs_add_filePcS_yPi,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr)kernelParams,
                        (IntPtr)IntPtr.Zero
                    );
                }
                Utils.CudaHelpers.CheckCudaError(res);
                res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                Utils.CudaHelpers.CheckCudaError(res);
            }

            unsafe
            {
                // Set up parameters.
                int count = 1;
                IntPtr parm1; // Name of assembly.

                var ptr = Marshal.StringToHGlobalAnsi(assem);
                BUFFERS buffers = new BUFFERS();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1 };

                CUmodule module = RUNTIME.RuntimeModule;
                CUfunction _Z34BCL_CLIFile_GetMetaDataForAssemblyPc = RUNTIME._Z34BCL_CLIFile_GetMetaDataForAssemblyPc(module);
                Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                    out Campy.Utils.CudaHelpers.dim3 tile_size,
                    out Campy.Utils.CudaHelpers.dim3 tiles);
                fixed (IntPtr* kernelParams = kp)
                {
                    res = Cuda.cuLaunchKernel(
                        _Z34BCL_CLIFile_GetMetaDataForAssemblyPc,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr)kernelParams,
                        (IntPtr)IntPtr.Zero
                    );
                }
                Utils.CudaHelpers.CheckCudaError(res);
                res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                Utils.CudaHelpers.CheckCudaError(res);
            }

            return result;
        }

        public static void LoadAssemblyOfTypeOntoGpu(Mono.Cecil.TypeReference type)
        {
            CUresult res = CUresult.CUDA_SUCCESS;

            // Get meta data from type from the GPU, as that is where it resides.
            // tMetaData* pTypeMetaData;
            // BCL_CLIFile_GetMetaDataForAssembly("ConsoleApp1.exe", &pTypeMetaDatanew FileStream();

            // Set up the type's assembly in file system.
            String assembly_location = Path.GetFullPath(type.Resolve().Module.FullyQualifiedName);
            string assem = Path.GetFileName(assembly_location);
            string full_path_assem = assembly_location;
            Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
            var corlib_bytes_handle_len = stream.Length;
            var corlib_bytes = new byte[corlib_bytes_handle_len];
            stream.Read(corlib_bytes, 0, (int)corlib_bytes_handle_len);
            var corlib_bytes_handle = GCHandle.Alloc(corlib_bytes, GCHandleType.Pinned);
            var corlib_bytes_intptr = corlib_bytes_handle.AddrOfPinnedObject();
            stream.Close();
            stream.Dispose();

            unsafe
            {
                // Set up parameters.
                int count = 4;
                IntPtr parm1; // Name of assembly.
                IntPtr parm2; // Contents
                IntPtr parm3; // Length
                IntPtr parm4; // result

                var ptr = Marshal.StringToHGlobalAnsi(assem);
                BUFFERS buffers = new BUFFERS();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr pointer2 = buffers.New((int)corlib_bytes_handle_len);
                BUFFERS.Cp(pointer2, corlib_bytes_intptr, (int)corlib_bytes_handle_len);
                IntPtr[] x2 = new IntPtr[] { pointer2 };
                GCHandle handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                parm2 = handle2.AddrOfPinnedObject();

                IntPtr[] x3 = new IntPtr[] { new IntPtr(corlib_bytes_handle_len) };
                GCHandle handle3 = GCHandle.Alloc(x3, GCHandleType.Pinned);
                parm3 = handle3.AddrOfPinnedObject();

                var pointer4 = buffers.New(sizeof(long));
                IntPtr[] x4 = new IntPtr[] { pointer4 };
                GCHandle handle4 = GCHandle.Alloc(x4, GCHandleType.Pinned);
                parm4 = handle4.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1, parm2, parm3, parm4 };

                CUmodule module = RUNTIME.RuntimeModule;
                CUfunction _Z16Bcl_Gfs_add_filePcS_yPi = RUNTIME._Z16Bcl_Gfs_add_filePcS_yPi(module);
                Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                    out Campy.Utils.CudaHelpers.dim3 tile_size,
                    out Campy.Utils.CudaHelpers.dim3 tiles);
                fixed (IntPtr* kernelParams = kp)
                {
                    res = Cuda.cuLaunchKernel(
                        _Z16Bcl_Gfs_add_filePcS_yPi,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr)kernelParams,
                        (IntPtr)IntPtr.Zero
                    );
                }
                Utils.CudaHelpers.CheckCudaError(res);
                res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                Utils.CudaHelpers.CheckCudaError(res);
            }

            unsafe
            {
                // Set up parameters.
                int count = 1;
                IntPtr parm1; // Name of assembly.

                var ptr = Marshal.StringToHGlobalAnsi(assem);
                BUFFERS buffers = new BUFFERS();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1 };

                CUmodule module = RUNTIME.RuntimeModule;
                CUfunction _Z34BCL_CLIFile_GetMetaDataForAssemblyPc = RUNTIME._Z34BCL_CLIFile_GetMetaDataForAssemblyPc(module);
                Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                    out Campy.Utils.CudaHelpers.dim3 tile_size,
                    out Campy.Utils.CudaHelpers.dim3 tiles);
                fixed (IntPtr* kernelParams = kp)
                {
                    res = Cuda.cuLaunchKernel(
                        _Z34BCL_CLIFile_GetMetaDataForAssemblyPc,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr)kernelParams,
                        (IntPtr)IntPtr.Zero
                    );
                }
                Utils.CudaHelpers.CheckCudaError(res);
                res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                Utils.CudaHelpers.CheckCudaError(res);
            }
        }

        public static void LoadBclCode()
        {
        }

        public static IntPtr RuntimeCubinImage
        {
            get; private set;
        }

        public static ulong RuntimeCubinImageSize
        {
            get; private set;
        }

        private static Dictionary<IntPtr, CUmodule> cached_modules = new Dictionary<IntPtr, CUmodule>();

        public static CUmodule InitializeModule(IntPtr cubin)
        {
            if (cached_modules.TryGetValue(cubin, out CUmodule value))
            {
                return value;
            }
            uint num_ops = 0;
            var op = new CUjit_option[num_ops];
            ulong[] op_values = new ulong[num_ops];

            var op_values_link_handle = GCHandle.Alloc(op_values, GCHandleType.Pinned);
            var op_values_link_intptr = op_values_link_handle.AddrOfPinnedObject();

            CUresult res = Cuda.cuModuleLoadDataEx(out CUmodule module, cubin, 0, op, op_values_link_intptr);
            CudaHelpers.CheckCudaError(res);
            cached_modules[cubin] = module;
            return module;
        }

        public static CUmodule RuntimeModule
        {
            get; set;
        }

        public static TypeReference FindBCLType(System.Type type)
        {
            var runtime = new RUNTIME();
            TypeReference result = null;
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(FindCoreLib());
            foreach (var bcl_type in md.GetTypes())
            {
                if (bcl_type.FullName == type.FullName)
                    return bcl_type;
            }
            return result;
        }

        public static CUfunction _Z22Initialize_BCL_GlobalsPvyiPP6_BCL_t(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z22Initialize_BCL_GlobalsPvyiPP6_BCL_t"));
            return function;
        }

        public static CUfunction _Z15Initialize_BCL1v(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z15Initialize_BCL1v"));
            return function;
        }

        public static CUfunction _Z15Initialize_BCL2v(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z15Initialize_BCL2v"));
            return function;
        }

        public static CUfunction _Z12Bcl_Gfs_initv(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z12Bcl_Gfs_initv"));
            return function;
        }

        public static CUfunction _Z16Bcl_Gfs_add_filePcS_yPi(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z16Bcl_Gfs_add_filePcS_yPi"));
            return function;
        }

        public static CUfunction _Z14Bcl_Heap_AllocPcS_S_(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z14Bcl_Heap_AllocPcS_S_"));
            return function;
        }

        public static CUfunction _Z34BCL_CLIFile_GetMetaDataForAssemblyPc(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z34BCL_CLIFile_GetMetaDataForAssemblyPc"));
            return function;
        }

        public static CUfunction _Z15Set_BCL_GlobalsP6_BCL_t(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z15Set_BCL_GlobalsP6_BCL_t"));
            return function;
        }
        
        public static IntPtr BclPtr { get; set; }

        public static TypeReference RewriteType(TypeReference tr)
        {
            foreach (var kv in _substituted_bcl)
            {
                if (kv.Key.FullName == tr.FullName)
                    tr = kv.Value;
            }
            return tr;
        }


        public static MethodDefinition SubstituteMethod(MethodReference method_reference)
        {
            // Look up base class.
            TypeReference mr_dt = method_reference.DeclaringType;
            MethodDefinition method_definition = method_reference.Resolve();
            // Find in Campy.Runtime, assuming it exists in the same
            // directory as the Campy compiler assembly.
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(FindCoreLib());
            // Find type/method in order to do a substitution. If there
            // is no substitution, continue on with the method.
            if (mr_dt != null)
            {
                foreach (var type in md.Types)
                {
                    if (type.Name == mr_dt.Name && type.Namespace == mr_dt.Namespace)
                    {
                        var fn = mr_dt.Module.Assembly.FullName;
                        var sub = mr_dt.SubstituteMonoTypeReference(md);
                        foreach (var meth in sub.Methods)
                        {
                            if (meth.Name != method_reference.Name) continue;
                            if (meth.Parameters.Count != method_reference.Parameters.Count) continue;

                            var mrdt_resolve = mr_dt.Resolve();
                            if (mrdt_resolve != null && mrdt_resolve.FullName != sub.FullName)
                                continue;

                            for (int i = 0; i < meth.Parameters.Count; ++i)
                            {
                                var p1 = meth.Parameters[i];
                                var p2 = method_reference.Parameters[i];
                            }

                            method_reference = meth;
                            method_definition = method_reference.Resolve();
                            break;
                        }
                    }
                }
            }

            if (method_definition == null)
            {
                return null;
            }
            else if (method_definition.Body == null)
            {
                return null;
            }

            return method_definition;
        }

        public static void RewriteCilCodeBlock(Mono.Cecil.Cil.MethodBody body)
        {
            List<Instruction> result = new List<Instruction>();
            for (int j = 0; j < body.Instructions.Count; ++j)
            {
                Instruction i = body.Instructions[j];

                var inst_to_insert = i;

                if (i.OpCode.FlowControl == FlowControl.Call)
                {
                    object method = i.Operand;

                    if (method as Mono.Cecil.MethodReference == null)
                        throw new Exception();

                    var method_reference = method as Mono.Cecil.MethodReference;
                    TypeReference mr_dt = method_reference.DeclaringType;

                    var bcl_substitute = SubstituteMethod(method_reference);
                    if (bcl_substitute != null)
                    {
                        CallSite cs = new CallSite(typeof(void).ToMonoTypeReference());
                        body.Instructions.RemoveAt(j);
                        var worker = body.GetILProcessor();
                        Instruction new_inst = worker.Create(i.OpCode, bcl_substitute);
                        body.Instructions.Insert(j, new_inst);
                    }
                }
            }
        }
    }
}

