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
using MethodImplAttributes = Mono.Cecil.MethodImplAttributes;

namespace Campy.Compiler
{
    public class Runtime
    {
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

            public PtxFunction(string mangled_name)
            {
                _mangled_name = mangled_name;

                // Construct LLVM extern that corresponds to type of function.
                Regex regex = new Regex(@"^_Z(?<len>[\d]+)(?<name>.+)$");
                Match m = regex.Match(_mangled_name);
                if (m.Success)
                {
                    var len_string = m.Groups["len"].Value;
                    var rest = m.Groups["name"].Value;
                    var len = Int32.Parse(len_string);
                    var name = rest.Substring(0, len);
                    var suffix = rest.Substring(len);
                    _short_name = name;

                    if (suffix == "i")
                    {

                    }
                    else if (suffix == "c")
                    { }
                    else if (suffix == "PKc")
                    { }
                    else if (suffix == "Pvy")
                    { }
                    else if (suffix == "y")
                    { }
                    else if (suffix == "Pc")
                    {
                        var decl = LLVM.AddFunction(
                                CampyConverter.global_llvm_module,
                                _mangled_name,
                                LLVM.FunctionType(LLVM.Int64Type(),
                                    new TypeRef[]
                                    {
                                        LLVM.PointerType(LLVM.Int8Type(), 0) // return value block.
                                    }, false));
                        CampyConverter.built_in_functions.Add(_mangled_name, decl);
                        this._valueref = decl;
                    }
                    else if (suffix == "Ph")
                    { }
                    else if (suffix == "Pv")
                    { }
                    else if (suffix == "v")
                    { }
                    else if (suffix == "P9tCLIFile_iPPc")
                    { }
                    else if (suffix == "P11tHeapRoots_")
                    { }
                    else if (suffix == "PvPPhPS_")
                    { }
                    else if (suffix == "P14tMD_MethodDef_")
                    { }
                    else if (suffix == "P11tHeapRoots_P12tMD_TypeDef_")
                    { }
                    else if (suffix == "P10tMetaData_PPhPP12tMD_TypeDef_S5_")
                    { }
                    else if (suffix == "P12tMD_TypeDef_jPS0_")
                    { }
                    else if (suffix == "P15tMD_MethodSpec_PP12tMD_TypeDef_S3_")
                    { }
                    else if (suffix == "P14tMD_MethodDef_P12tMD_TypeDef_jPS2_")
                    { }
                    else if (suffix == "PKcS0_y")
                    { }
                    else if (suffix == "PKcS0_")
                    { }
                    else if (suffix == "PcPKc")
                    { }
                    else if (suffix == "PcPKcy")
                    { }
                    else if (suffix == "PvPKvy")
                    { }
                    else if (suffix == "PKci")
                    { }
                    else if (suffix == "PKcy")
                    { }
                    else if (suffix == "PPcPKc")
                    { }
                    else if (suffix == "Pviy")
                    { }
                    else if (suffix == "PKvS0_y")
                    { }
                    else if (suffix == "PKviy")
                    { }
                    else if (suffix == "PcyPKcS_")
                    { }
                    else if (suffix == "PPcPKcS_")
                    { }
                    else if (suffix == "PPcPKcz")
                    { }
                    else if (suffix == "PcPKcS_")
                    { }
                    else if (suffix == "PcPKcz")
                    { }
                    else if (suffix == "PKcz")
                    { }
                    else if (suffix == "PKcPc")
                    { }
                    else if (suffix == "P11tHeapRoots_Pvj")
                    { }
                    else if (suffix == "P12tMD_TypeDef_j")
                    { }
                    else if (suffix == "P12tMD_TypeDef_")
                    { }
                    else if (suffix == "P12tMD_TypeDef_Ph")
                    { }
                    else if (suffix == "PhS_")
                    { }
                    else if (suffix == "P14tMD_MethodDef_j")
                    { }
                    else if (suffix == "P8tThread_j")
                    { }
                    else if (suffix == "PPh")
                    { }
                    else if (suffix == "P10tMetaData_Pvj")
                    { }
                    else if (suffix == "P10tMetaData_hPPh")
                    { }
                    else if (suffix == "PhS_S_")
                    {
                        var decl = LLVM.AddFunction(
                                CampyConverter.global_llvm_module,
                                _mangled_name,
                                LLVM.FunctionType(LLVM.Int64Type(),
                                    new TypeRef[]
                                    {
                                                        LLVM.PointerType(LLVM.VoidType(), 0), // "this"
                                                        LLVM.PointerType(LLVM.VoidType(), 0), // params in a block.
                                                        LLVM.PointerType(LLVM.VoidType(), 0) // return value block.
                                    }, false));
                        CampyConverter.built_in_functions.Add(_mangled_name, decl);
                        this._valueref = decl;
                    }
                    else if (suffix == "PcS_S_")
                    {
                        var decl = LLVM.AddFunction(
                                CampyConverter.global_llvm_module,
                                _mangled_name,
                                LLVM.FunctionType(
                                    LLVM.PointerType(LLVM.VoidType(),0),
                                    new TypeRef[]
                                    {
                                        LLVM.PointerType(LLVM.Int8Type(), 0),
                                        LLVM.PointerType(LLVM.Int8Type(), 0),
                                        LLVM.PointerType(LLVM.Int8Type(), 0)
                                    }, false));
                        CampyConverter.built_in_functions.Add(_mangled_name, decl);
                        this._valueref = decl;
                    }
                    else;

                }
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

        // This table encode runtime type information for rewriting BCL types. Use this to determine
        // what the type maps into in the GPU BCL.
        private static Dictionary<TypeReference, TypeReference> _substituted_bcl = new Dictionary<TypeReference, TypeReference>();

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

        public static TypeReference RewriteType(TypeReference tr)
        {
            foreach (var kv in _substituted_bcl)
            {
                if (kv.Key.FullName == tr.FullName)
                    tr = kv.Value;
            }
            return tr;
        }

        public static void Initialize()
        {
            // Load C# library for BCL, and grab all types and methods.
            string yopath = @"C:\Users\kenne\Documents\Campy2\Campy.Runtime\Corlib\bin\Debug\netstandard1.3\corlib.dll";
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(yopath);
            foreach (var bcl_type in md.GetTypes())
            {
                var t_system_type = System.Type.GetType(bcl_type.FullName);
                if (t_system_type == null) continue;

                System.Console.WriteLine("BCL type added: " + bcl_type.FullName);

                var to_mono = t_system_type.ToMonoTypeReference();

                _substituted_bcl.Add(to_mono, bcl_type);

                foreach (var m in bcl_type.Methods)
                {
                    var x = m.ImplAttributes;
                    if ((x & MethodImplAttributes.InternalCall) != 0)
                    {
                        System.Console.WriteLine("BCL internal method added: " + m.FullName);

                        _internalCalls.Add(new BclNativeMethod(bcl_type, m));
                    }
                }
            }

            // Parse PTX files for all "visible" functions, and create LLVM declarations.
            // For "Internal Calls", these functions appear here, but also on the _internalCalls list.
            var assembly = Assembly.GetAssembly(typeof(Campy.Compiler.Runtime));
            var resource_names = assembly.GetManifestResourceNames();
            foreach (var resource_name in resource_names)
            {
                using (Stream stream = assembly.GetManifestResourceStream(resource_name))
                using (StreamReader reader = new StreamReader(stream))
                {
                    string gpu_bcl_ptx = reader.ReadToEnd();
                    // Parse the PTX for ".visible" functions, and enter each in
                    // the runtime table.
                    string[] lines = gpu_bcl_ptx.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        Regex regex = new Regex(@"\.visible.*[ ](?<name>\w+)\($");
                        Match m = regex.Match(line);
                        if (m.Success)
                        {
                            var mangled_name = m.Groups["name"].Value;

                            _ptx_functions.Add(new PtxFunction(mangled_name));

                            System.Console.WriteLine("Adding PTX method " + mangled_name);
                        }
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
                Buffers buffers = new Buffers();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                Buffers.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr pointer2 = buffers.New((int)corlib_bytes_handle_len);
                Buffers.Cp(pointer2, corlib_bytes_intptr, (int)corlib_bytes_handle_len);
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

                CUmodule module = Runtime.RuntimeModule;
                CUfunction _Z16Bcl_Gfs_add_filePcS_yPi = Runtime._Z16Bcl_Gfs_add_filePcS_yPi(module);
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
                Buffers buffers = new Buffers();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                Buffers.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1 };

                CUmodule module = Runtime.RuntimeModule;
                CUfunction _Z34BCL_CLIFile_GetMetaDataForAssemblyPc = Runtime._Z34BCL_CLIFile_GetMetaDataForAssemblyPc(module);
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
                Buffers buffers = new Buffers();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                Buffers.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr pointer2 = buffers.New((int)corlib_bytes_handle_len);
                Buffers.Cp(pointer2, corlib_bytes_intptr, (int)corlib_bytes_handle_len);
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

                CUmodule module = Runtime.RuntimeModule;
                CUfunction _Z16Bcl_Gfs_add_filePcS_yPi = Runtime._Z16Bcl_Gfs_add_filePcS_yPi(module);
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
                Buffers buffers = new Buffers();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                Buffers.Cp(pointer1, ptr, assem.Length + 1);
                IntPtr[] x1 = new IntPtr[] { pointer1 };
                GCHandle handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                parm1 = handle1.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1 };

                CUmodule module = Runtime.RuntimeModule;
                CUfunction _Z34BCL_CLIFile_GetMetaDataForAssemblyPc = Runtime._Z34BCL_CLIFile_GetMetaDataForAssemblyPc(module);
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
            return;
            Utils.CudaHelpers.CheckCudaError(Cuda.cuCtxGetLimit(out ulong pvalue, CUlimit.CU_LIMIT_STACK_SIZE));
            Utils.CudaHelpers.CheckCudaError(Cuda.cuCtxSetLimit(CUlimit.CU_LIMIT_STACK_SIZE, (uint)pvalue * 25));
            System.Console.WriteLine("Stack size " + pvalue);

            CUresult res = CUresult.CUDA_SUCCESS;

            uint num_ops_link = 5;
            var op_link = new CUjit_option[num_ops_link];
            ulong[] op_values_link = new ulong[num_ops_link];

            int size = 1024 * 100;
            op_link[0] = CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
            op_values_link[0] = (ulong)size;

            op_link[1] = CUjit_option.CU_JIT_INFO_LOG_BUFFER;
            byte[] info_log_buffer = new byte[size];
            var info_log_buffer_handle = GCHandle.Alloc(info_log_buffer, GCHandleType.Pinned);
            var info_log_buffer_intptr = info_log_buffer_handle.AddrOfPinnedObject();
            op_values_link[1] = (ulong)info_log_buffer_intptr;

            op_link[2] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
            op_values_link[2] = (ulong)size;

            op_link[3] = CUjit_option.CU_JIT_ERROR_LOG_BUFFER;
            byte[] error_log_buffer = new byte[size];
            var error_log_buffer_handle = GCHandle.Alloc(error_log_buffer, GCHandleType.Pinned);
            var error_log_buffer_intptr = error_log_buffer_handle.AddrOfPinnedObject();
            op_values_link[3] = (ulong)error_log_buffer_intptr;

            op_link[4] = CUjit_option.CU_JIT_LOG_VERBOSE;
            op_values_link[4] = (ulong)1;

            //op_link[5] = CUjit_option.CU_JIT_TARGET;
            //op_values_link[5] = (ulong)CUjit_target.CU_TARGET_COMPUTE_35;

            var op_values_link_handle = GCHandle.Alloc(op_values_link, GCHandleType.Pinned);
            var op_values_link_intptr = op_values_link_handle.AddrOfPinnedObject();
            res = Cuda.cuLinkCreate_v2(num_ops_link, op_link, op_values_link_intptr, out CUlinkState linkState);
            CudaHelpers.CheckCudaError(res);

            // Go to a standard directory, for now hardwired....
            var dir = @"C:\Users\kenne\Documents\Campy2\Campy.Runtime\Native\x64\Debug";
            var resource_names = Directory.GetFiles(dir);
            uint num_ops = 0;
            var op = new CUjit_option[num_ops];
            var op_values = new ulong[num_ops];
            var op_values_handle = GCHandle.Alloc(op_values, GCHandleType.Pinned);
            var op_values_intptr = op_values_handle.AddrOfPinnedObject();
            foreach (var resource_name in resource_names)
            {
                var last_index_of = resource_name.LastIndexOf('.');
                if (last_index_of < 0) continue;
                if (resource_name.Contains("device-link")) continue;
                if (resource_name.Substring(last_index_of) != ".obj") continue;

                using (Stream stream = new FileStream(resource_name, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    var len = stream.Length;
                    var gpu_bcl_obj = new byte[len];
                    stream.Read(gpu_bcl_obj, 0, (int)len);

                    var gpu_bcl_obj_handle = GCHandle.Alloc(gpu_bcl_obj, GCHandleType.Pinned);
                    var gpu_bcl_obj_intptr = gpu_bcl_obj_handle.AddrOfPinnedObject();

                    res = Cuda.cuLinkAddData_v2(linkState, CUjitInputType.CU_JIT_INPUT_OBJECT,
                        gpu_bcl_obj_intptr, (uint)len,
                        "", num_ops, op, op_values_intptr);
                    {
                        string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                        System.Console.WriteLine(info);
                        string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                        System.Console.WriteLine(error);
                    }
                    Utils.CudaHelpers.CheckCudaError(res);
                }
            }

            IntPtr image;
            res = Cuda.cuLinkComplete(linkState, out image, out ulong sz);

            {
                string info = Marshal.PtrToStringAnsi(info_log_buffer_intptr);
                System.Console.WriteLine(info);
                string error = Marshal.PtrToStringAnsi(error_log_buffer_intptr);
                System.Console.WriteLine(error);
            }
            CudaHelpers.CheckCudaError(res);

            RuntimeCubinImage = image;
            RuntimeCubinImageSize = sz;
            Runtime.RuntimeModule = Runtime.InitializeModule(Runtime.RuntimeCubinImage);
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
            TypeReference result = null;
            string yopath = @"C:\Users\kenne\Documents\Campy2\Campy.Runtime\Corlib\bin\Debug\netstandard1.3\corlib.dll";
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(yopath);
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
    }
}

