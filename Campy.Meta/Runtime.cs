using System.Text;

namespace Campy.Meta
{
    using MethodImplAttributes = Mono.Cecil.MethodImplAttributes;
    using Mono.Cecil;
    using Swigged.Cuda;
    using Swigged.LLVM;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using System.Runtime.InteropServices;
    using System.Text.RegularExpressions;
    using System;
    using Utils;

    public class RUNTIME
    {
        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclHeapAlloc")]
        public static extern System.IntPtr BclHeapAlloc(System.IntPtr bcl_type);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclSizeOf")]
        public static extern System.IntPtr BclSizeOf(System.IntPtr bcl_type);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclSetOptions")]
        public static extern void BclSetOptions(UInt64 options);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitTheBcl")]
        public static extern void InitTheBcl(System.IntPtr a1, long a2, long a3, int a4);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclInitFileSystem")]
        public static extern void BclInitFileSystem();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclAddFile")]
        public static extern void BclAddFile(System.IntPtr name, System.IntPtr file, long length, System.IntPtr result);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclContinueInit")]
        public static extern void BclContinueInit();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclCheckHeap")]
        public static extern void BclCheckHeap();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGcCollect")]
        public static extern void BclGcCollect();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclAllocString")]
        public static extern IntPtr BclAllocString(int length, IntPtr chars);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclArrayAlloc")]
        public static extern System.IntPtr BclArrayAlloc(System.IntPtr bcl_type, int rank, uint[] lengths);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetMetaOfType")]
        public static extern System.IntPtr BclGetMetaOfType(
            [MarshalAs(UnmanagedType.LPStr)] string assemblyName,
            [MarshalAs(UnmanagedType.LPStr)] string nameSpace,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            System.IntPtr nested);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclConstructArrayType")]
        public static extern System.IntPtr BclConstructArrayType(IntPtr element_type, int rank);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclConstructGenericInstanceType")]
        public static extern System.IntPtr BclConstructGenericInstanceType(IntPtr base_type, int count, IntPtr[] args);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclHeapGetType")]
        public static extern System.IntPtr BclHeapGetType(IntPtr ptr);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclFindFieldInType")]
        public static extern System.IntPtr BclFindFieldInType(IntPtr ptr, [MarshalAs(UnmanagedType.LPStr)] string name);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclFindFieldInTypeAll")]
        public static extern System.IntPtr BclFindFieldInTypeAll(IntPtr ptr, [MarshalAs(UnmanagedType.LPStr)] string name);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetField")]
        public static extern System.IntPtr BclGetField(IntPtr ptr, IntPtr bcl_field);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetFieldOffset")]
        public static extern int BclGetFieldOffset(IntPtr bcl_field);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetFieldSize")]
        public static extern int BclGetFieldSize(IntPtr bcl_field);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetStaticField")]
        public static extern System.IntPtr BclGetStaticField(IntPtr bcl_field);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetFields")]
        public static extern unsafe void BclGetFields(IntPtr bcl_type, IntPtr** buf, int* len);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetFieldName")]
        public static extern IntPtr BclGetFieldName(IntPtr bcl_field);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetFieldType")]
        public static extern IntPtr BclGetFieldType(IntPtr bcl_field);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclSystemArrayGetRank")]
        public static extern int BclSystemArrayGetRank(IntPtr bcl_object);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclSystemArrayGetDims")]
        public static extern IntPtr BclSystemArrayGetDims(IntPtr bcl_object);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclSystemArrayLoadElementIndices")]
        public static extern IntPtr BclSystemArrayLoadElementIndices(IntPtr bcl_object, uint dim, IntPtr indices, IntPtr value);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclSystemArrayLoadElementIndicesAddress")]
        public static extern IntPtr BclSystemArrayLoadElementIndicesAddress(IntPtr bcl_object, IntPtr indices, IntPtr value_address);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclMetaDataGetMethodJit")]
        public static extern IntPtr BclMetaDataGetMethodJit(IntPtr bcl_object, int table_ref);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclMetaDataSetMethodJit")]
        public static extern IntPtr BclMetaDataSetMethodJit(IntPtr method_ptr, IntPtr bcl_object, int table_ref);



        public static ModuleRef global_llvm_module;
        public static List<ModuleRef> all_llvm_modules;
        public static Mono.Cecil.Cil.ILProcessor worker;
        public static Dictionary<string, ValueRef> _bcl_runtime_csharp_internal_to_valueref;
        public static Dictionary<string, TypeReference> all_types = new Dictionary<string, TypeReference>();
        // This table encodes runtime type information for rewriting internal calls in the native portion of
        // the BCL for the GPU. It was originally encoded in dna/internal.c. However, it's easier and
        // safer to derive the information from the C# portion of the BCL using System.Reflection.
        //
        // Why is this information needed? In Inst.c, I need to make a call of a function to the runtime.
        // I only have PTX files, which removes the type information from the signature of
        // the original call (it is all three parameters of void*).
        private static List<BclNativeMethod> _bcl_runtime_csharp_methods_labeled_internal = new List<BclNativeMethod>();

        // This table is a record of all '.visible' functions in a generated PTX file. Use this name when calling
        // functions in PTX/LLVM.
        private static List<PtxFunction> _ptx_functions = new List<PtxFunction>();


        private static Dictionary<string, TypeRef> _ptx_type_to_llvm_typeref = new Dictionary<string, TypeRef>()
        {
            {"i8", LLVM.Int8Type()},
            {"i16", LLVM.Int16Type()},
            {"i32", LLVM.Int32Type()},
            {"i64", LLVM.Int64Type()},
        };

        public RUNTIME()
        {
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
                        var arg1 = a.ConstructorArguments[0];
                        var v1 = arg1.Value;
                        var s1 = (string)v1;
                        _short_name = s1;
                        var arg2 = a.ConstructorArguments[1];
                        var v2 = arg2.Value;
                        var s2 = (string)v2;
                        _native_name = s2;
                    }
                }

                if (_native_name == null && !md.IsInternalCall)
                    ;
            }
        }

        public class PtxFunction
        {
            public string _mangled_name;
            public ValueRef _valueref;

            public PtxFunction(string mangled_name, ValueRef value_ref)
            {
                _mangled_name = mangled_name;
                _valueref = value_ref;
            }
        }

        private class InternalCallEnumerable : IEnumerable<BclNativeMethod>
        {
            public InternalCallEnumerable()
            {
            }

            public IEnumerator<BclNativeMethod> GetEnumerator()
            {
                foreach (var key in _bcl_runtime_csharp_methods_labeled_internal)
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
                var suffix = Campy.Utils.CampyEnv.IsLinux ? ".a" : ".lib";
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "campy-runtime-native" + suffix;
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception)
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
                var suffix = Campy.Utils.CampyEnv.IsLinux ? ".a" : ".lib";
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar
                                                       + ".." + Path.DirectorySeparatorChar
                                                       + ".." + Path.DirectorySeparatorChar
                                                       + "content" + Path.DirectorySeparatorChar
                                                       + "campy-runtime-native"
                                                       + suffix;
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception)
            {
            }

            // Try something else...

            try
            {
                // Let's try the calling executable directory.
                var dir = Path.GetDirectoryName(Path.GetFullPath(System.Reflection.Assembly.GetEntryAssembly().Location));
                var suffix = Campy.Utils.CampyEnv.IsLinux ? ".a" : ".lib";
                string full_path_assem = dir + Path.DirectorySeparatorChar + "campy-runtime-native" + suffix;
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception)
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
                var suffix = Campy.Utils.CampyEnv.IsLinux ? ".a" : ".lib";
                string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "campy-runtime-native" + suffix;
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception)
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
            catch (Exception)
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
                                                       + "lib" + Path.DirectorySeparatorChar
                                                       + "native" + Path.DirectorySeparatorChar
                                                       + "corlib.dll";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception)
            {
            }

            System.Diagnostics.StackTrace callStack = new System.Diagnostics.StackTrace();
            for (int i = 0; i < callStack.FrameCount; i++)
            {
                System.Diagnostics.StackFrame sf = callStack.GetFrame(i);
                try
                {
                    // Let's try going up the stack, looking for anything.
                    var m = sf.GetMethod();
                    var p = m.DeclaringType.Assembly.Location;
                    var dir = Path.GetDirectoryName(Path.GetFullPath(p));
                    string full_path_assem = dir + Path.DirectorySeparatorChar
                                                           + "corlib.dll";
                    full_path_assem = Path.GetFullPath(full_path_assem);
                    Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                    stream.Close();
                    return full_path_assem;
                }
                catch (Exception)
                {
                }
            }

            try
            {
                // Let's try the calling executable directory.
                var p = typeof(RUNTIME).Assembly.Location;
                var dir = Path.GetDirectoryName(Path.GetFullPath(p));
                string full_path_assem = dir + Path.DirectorySeparatorChar
                                                       + "corlib.dll";
                full_path_assem = Path.GetFullPath(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                stream.Close();
                return full_path_assem;
            }
            catch (Exception)
            {
            }

            // Try something else...
            // Panic, assume this is somewhere within the Campy directory.
            // go up the tree and try to find.
            for (int level = 0; level < 10; ++level)
            {
                try
                {
                    // This could be a unit test in Campy. If that is true, then look in the standard directory structure
                    // for Campy source/object. It should have actually copied the damn corlib.dll to the output executable directory,
                    // but someone set up the test wrong. Anyways, assume that the project is up to date, and load from Campy.Runtime.
                    // ../../../../../Campy.Runtime/Corlib/bin/Debug/net20/
                    var prefix = string.Concat(Enumerable.Repeat("../", level));
                    var path_of_campy = prefix + @"Campy.Runtime/Corlib/bin/Debug/netstandard2.0";
                    string full_path_assem = path_of_campy + Path.DirectorySeparatorChar + "corlib.dll";
                    full_path_assem = Path.GetFullPath(full_path_assem);
                    Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                    stream.Close();
                    return full_path_assem;
                }
                catch (Exception)
                {
                }
            }

            // Fuck. I have no idea.
            return null;
        }


        private static string runtime_decls = $@"
declare void @_Z15Initialize_BCL0yyi(i64, i64, i32)
declare i64 @_Z7Gmallocy(i64)
declare i32 @_Z7roundUpii(i32, i32)
declare i64 @_Z13simple_mallocy(i64)
declare void @_Z5GfreePv(i64)
declare void @_Z16CommonInitTheBclPvyyi(i64, i64, i64, i32)
declare void @_Z18InternalInitTheBclPvyyi(i64, i64, i64, i32)
declare void @_Z15Get_BCL_GlobalsPP6_BCL_t(i64)
declare void @_Z7gpuexiti(i32)
declare void @_Z21check_heap_structuresv()
declare void @_Z17InternalCheckHeapv()
declare i64 @_Z8GreallocPvy(i64, i64)
declare void @_Z22InternalInitializeBCL1v()
declare void @_Z22InternalInitializeBCL2v()
declare i64 @_Z15Bcl_Array_AllocP12tMD_TypeDef_iPj(i64, i32, i64)
declare i32 @_Z21get_kernel_base_indexv()
declare void @_Z18InternalSetOptionsy(i64)
declare i64 @_Z33CLIFile_GetMetaDataForAssemblyAuxPcS_tt(i64, i64, i32, i32)
declare i64 @_Z12CLIFile_LoadPc(i64)
declare i64 @_Z30CLIFile_GetMetaDataForAssemblyP16tMD_AssemblyRef_(i64)
declare i32 @_Z15CLIFile_ExecuteP9tCLIFile_iPPc(i64, i32, i64)
declare void @_Z20CLIFile_GetHeapRootsP11tHeapRoots_(i64)
declare i64 @_Z18Delegate_GetMethodPv(i64)
declare i64 @_Z26Delegate_GetMethodAndStorePvPPhPS_(i64, i64, i64)
declare i64 @_Z12Map_DelegateP14tMD_MethodDef_(i64)
declare void @_Z14Finalizer_Initv()
declare void @_Z12AddFinalizerPh(i64)
declare i64 @_Z16GetNextFinalizerv()
declare i64 @_Z35Generics_GetGenericTypeFromCoreTypeP12tMD_TypeDef_jPS0_(i64, i32, i64)
declare i64 @_Z35Generics_GetMethodDefFromCoreMethodP14tMD_MethodDef_P12tMD_TypeDef_jPS2_(i64, i64, i32, i64)
declare void @_Z20Generic_GetHeapRootsP11tHeapRoots_P12tMD_TypeDef_(i64, i64)
declare i64 @_Z30Generics_GetGenericTypeFromSigP10tMetaData_PPhPP12tMD_TypeDef_S5_(i64, i64, i64, i64)
declare i64 @_Z29Generics_GetMethodDefFromSpecP15tMD_MethodSpec_PP12tMD_TypeDef_S3_(i64, i64, i64)
declare i32 @_Z8Gtolowerc(i32)
declare i64 @_Z7GstrlenPKc(i64)
declare i64 @_Z7GmemcpyPvPKvy(i64, i64, i64)
declare i32 @_Z12GstrncasecmpPKcS0_y(i64, i64, i64)
declare i32 @_Z11GstrcasecmpPKcS0_(i64, i64)
declare i64 @_Z7GstrcpyPcPKc(i64, i64)
declare i64 @_Z8GstrncpyPcPKcy(i64, i64, i64)
declare i64 @_Z8GstrlcpyPcPKcy(i64, i64, i64)
declare i64 @_Z7GstrcatPcPKc(i64, i64)
declare i64 @_Z8GstrncatPcPKcy(i64, i64, i64)
declare i32 @_Z7GstrcmpPKcS0_(i64, i64)
declare i32 @_Z8GstrncmpPKcS0_y(i64, i64, i64)
declare i64 @_Z7GstrchrPKci(i64, i32)
declare i64 @_Z8GstrrchrPKci(i64, i32)
declare i64 @_Z8GstrnlenPKcy(i64, i64)
declare i64 @_Z7GstrdupPKc(i64)
declare i64 @_Z7GstrspnPKcS0_(i64, i64)
declare i64 @_Z8GstrpbrkPKcS0_(i64, i64)
declare i64 @_Z7GstrtokPcPKc(i64, i64)
declare i64 @_Z7GstrsepPPcPKc(i64, i64)
declare i64 @_Z7strswabPKc(i64)
declare i64 @_Z7GmemsetPviy(i64, i32, i64)
declare i64 @_Z8GmemmovePvPKvy(i64, i64, i64)
declare i32 @_Z7GmemcmpPKvS0_y(i64, i64, i64)
declare i64 @_Z8GmemscanPviy(i64, i32, i64)
declare i64 @_Z7GstrstrPKcS0_(i64, i64)
declare i64 @_Z7GmemchrPKviy(i64, i32, i64)
declare void @_Z9GstoupperPc(i64)
declare void @_Z9GstolowerPc(i64)
declare i32 @_Z8Gtoupperc(i32)
declare i32 @_Z10GvsnprintfPcyPKcS_(i64, i64, i64, i64)
declare i32 @_Z10GvasprintfPPcPKcS_(i64, i64, i64)
declare i32 @_Z9GasprintfPPcPKcz(i64, i64, i8)
declare i32 @_Z9GvsprintfPcPKcS_(i64, i64, i64)
declare i32 @_Z8GsprintfPcPKcz(i64, i64, i8)
declare i32 @_Z7GprintfPKcz(i64, i8)
declare i32 @_Z8GvprintfPKcPc(i64, i64)
declare void @_Z13Heap_SetRootsP11tHeapRoots_Pvj(i64, i64, i32)
declare i64 @_Z14Heap_AllocTypeP12tMD_TypeDef_(i64)
declare void @_Z9Heap_Initv()
declare void @_Z20Heap_UnmarkFinalizerPh(i64)
declare void @_Z19Heap_GarbageCollectv()
declare i32 @_Z19Heap_NumCollectionsv()
declare i32 @_Z19Heap_GetTotalMemoryv()
declare i64 @_Z10Heap_AllocP12tMD_TypeDef_j(i64, i32)
declare i64 @_Z23Heap_AllocTypeVoidStarsPv(i64)
declare i64 @_Z12Heap_GetTypePh(i64)
declare void @_Z20Heap_MakeUndeletablePh(i64)
declare void @_Z18Heap_MakeDeletablePh(i64)
declare i64 @_Z8Heap_BoxP12tMD_TypeDef_Ph(i64, i64)
declare i64 @_Z10Heap_ClonePh(i64)
declare i32 @_Z17Heap_SyncTryEnterPh(i64)
declare i32 @_Z13Heap_SyncExitPh(i64)
declare i64 @_Z21Heap_SetWeakRefTargetPhS_(i64, i64)
declare i64 @_Z22Heap_GetWeakRefAddressPh(i64)
declare void @_Z25Heap_RemovedWeakRefTargetPh(i64)
declare i64 @_Z16InternalCall_MapP14tMD_MethodDef_(i64)
declare i32 @_Z11JIT_ExecuteP8tThread_j(i64, i32)
declare void @_Z16JIT_Execute_Initv()
declare i64 @_Z20MetaData_GetTableRowP10tMetaData_j(i64, i32)
declare i32 @_Z35MetaData_DecodeUnsigned32BitIntegerPPh(i64)
declare i32 @_Z34MetaData_DecodeUnsigned8BitIntegerPPh(i64)
declare i32 @_Z28MetaData_DecodeSigEntryTokenPPh(i64)
declare i64 @_Z24MetaData_DecodePublicKeyPh(i64)
declare i64 @_Z8MetaDatav()
declare void @_Z20MetaData_LoadStringsP10tMetaData_Pvj(i64, i64, i32)
declare i32 @_Z30MetaData_DecodeHeapEntryLengthPPh(i64)
declare void @_Z18MetaData_LoadBlobsP10tMetaData_Pvj(i64, i64, i32)
declare void @_Z24MetaData_LoadUserStringsP10tMetaData_Pvj(i64, i64, i32)
declare void @_Z18MetaData_LoadGUIDsP10tMetaData_Pvj(i64, i64, i32)
declare void @_Z13MetaData_Initv()
declare i32 @_Z6GetU16Ph(i64)
declare i32 @_Z6GetU32Ph(i64)
declare i64 @_Z6GetU64Ph(i64)
declare i32 @_Z10CodedIndexP10tMetaData_hPPh(i64, i32, i64)
declare i32 @_Z11Coded2IndexP10tMetaData_iPPh(i64, i32, i64)
declare void @_Z15OutputSignaturePh(i64)
declare void @_Z17ModuleTableReaderiiP10tMetaData_P5tRVA_PPhPv(i32, i32, i64, i64, i64, i64)
declare void @_Z12OutputModuleP11tMD_Module_(i64)
declare void @_Z18TypeRefTableReaderiiP10tMetaData_P5tRVA_PPhPv(i32, i32, i64, i64, i64, i64)
declare void @_Z13OutputTypeRefP12tMD_TypeRef_(i64)
declare void @_Z18TypeDefTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z13OutputTypeDefP12tMD_TypeDef_(i64)
declare void @_Z19FieldPtrTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputFieldPtrP13tMD_FieldPtr_(i64)
declare void @_Z19FieldDefTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputFieldDefP13tMD_FieldDef_(i64)
declare void @_Z20MethodDefTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z15OutputMethodDefP14tMD_MethodDef_(i64)
declare void @_Z16ParamTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z11OutputParamP10tMD_Param_(i64)
declare void @_Z24InterfaceImplTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z19OutputInterfaceImplP18tMD_InterfaceImpl_(i64)
declare void @_Z20MemberRefTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z15OutputMemberRefP14tMD_MemberRef_(i64)
declare void @_Z19ConstantTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputConstantP13tMD_Constant_(i64)
declare void @_Z26CustomAttributeTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z21OutputCustomAttributeP20tMD_CustomAttribute_(i64)
declare void @_Z23FieldMarshalTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z18OutputFieldMarshalP17tMD_FieldMarshal_(i64)
declare void @_Z23DeclSecurityTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z18OutputDeclSecurityP17tMD_DeclSecurity_(i64)
declare void @_Z22ClassLayoutTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z17OutputClassLayoutP16tMD_ClassLayout_(i64)
declare void @_Z22FieldLayoutTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z17OutputFieldLayoutP16tMD_FieldLayout_(i64)
declare void @_Z24StandAloneSigTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z19OutputStandAloneSigP18tMD_StandAloneSig_(i64)
declare void @_Z19EventMapTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputEventMapP13tMD_EventMap_(i64)
declare void @_Z16EventTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z11OutputEventP10tMD_Event_(i64)
declare void @_Z22PropertyMapTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z17OutputPropertyMapP16tMD_PropertyMap_(i64)
declare void @_Z19PropertyTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputPropertyP13tMD_Property_(i64)
declare void @_Z26MethodSemanticsTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z21OutputMethodSemanticsP20tMD_MethodSemantics_(i64)
declare void @_Z21MethodImplTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z16OutputMethodImplP15tMD_MethodImpl_(i64)
declare void @_Z20ModuleRefTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z15OutputModuleRefP14tMD_ModuleRef_(i64)
declare void @_Z19TypeSpecTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputTypeSpecP13tMD_TypeSpec_(i64)
declare void @_Z18ImplMapTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z13OutputImplMapP12tMD_ImplMap_(i64)
declare void @_Z19FieldRVATableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputFieldRVAP13tMD_FieldRVA_(i64)
declare void @_Z19AssemblyTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z14OutputAssemblyP13tMD_Assembly_(i64)
declare void @_Z22AssemblyRefTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z17OutputAssemblyRefP16tMD_AssemblyRef_(i64)
declare void @_Z23ExportedTypeTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z18OutputExportedTypeP17tMD_ExportedType_(i64)
declare void @_Z27ManifestResourceTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z22OutputManifestResourceP21tMD_ManifestResource_(i64)
declare void @_Z22NestedClassTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z17OutputNestedClassP16tMD_NestedClass_(i64)
declare void @_Z23GenericParamTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z18OutputGenericParamP17tMD_GenericParam_(i64)
declare void @_Z21MethodSpecTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z16OutputMethodSpecP15tMD_MethodSpec_(i64)
declare void @_Z33GenericParamConstraintTableReaderiiP10tMetaData_P5tRVA_PPhPvi(i32, i32, i64, i64, i64, i64, i32)
declare void @_Z28OutputGenericParamConstraintP27tMD_GenericParamConstraint_(i64)
declare void @_Z19MetaData_LoadTablesP10tMetaData_P5tRVA_Phj(i64, i64, i64, i32)
declare i64 @_Z16MetaData_GetBlobPhPj(i64, i64)
declare i64 @_Z22MetaData_GetUserStringP10tMetaData_jPj(i64, i32, i64)
declare void @_Z20MetaData_GetConstantP10tMetaData_jPh(i64, i32, i64)
declare void @_Z21MetaData_GetHeapRootsP11tHeapRoots_P10tMetaData_(i64, i64)
declare void @_Z22MetaData_PrintMetaDataP10tMetaData_(i64)
declare void @_Z17MetaData_SetFieldPhP13tMD_FieldDef_S_(i64, i64, i64)
declare i64 @_Z17MetaData_GetFieldPhP13tMD_FieldDef_(i64, i64)
declare void @_Z18MetaData_GetFieldsP12tMD_TypeDef_PPP13tMD_FieldDef_Pi(i64, i64, i64)
declare i64 @_Z21MetaData_GetFieldNameP13tMD_FieldDef_(i64)
declare i64 @_Z21MetaData_GetFieldTypeP13tMD_FieldDef_(i64)
declare i64 @_Z21MetaData_GetMethodJitPvi(i64, i32)
declare void @_Z21MetaData_SetMethodJitPvS_i(i64, i64, i32)
declare void @_Z21MetaData_Fill_TypeDefP12tMD_TypeDef_PS0_S1_(i64, i64, i64)
declare void @_Z23MetaData_Fill_MethodDefP12tMD_TypeDef_P14tMD_MethodDef_PS0_S3_(i64, i64, i64, i64)
declare void @_Z22MetaData_Fill_FieldDefP12tMD_TypeDef_P13tMD_FieldDef_jPS0_(i64, i64, i32, i64)
declare i64 @_Z35MetaData_GetTypeDefFromDefRefOrSpecP10tMetaData_jPP12tMD_TypeDef_S3_(i64, i32, i64, i64)
declare i64 @_Z27MetaData_GetTypeDefFromNameP10tMetaData_PcS1_P12tMD_TypeDef_(i64, i64, i64, i64)
declare i64 @_Z31MetaData_GetTypeDefFromFullNamePcS_S_(i64, i64, i64)
declare i32 @_Z26MetaData_CompareNameAndSigPcPhP10tMetaData_PP12tMD_TypeDef_S5_P14tMD_MethodDef_S5_S5_(i64, i64, i64, i64, i64, i64, i64, i64)
declare i64 @_Z24MetaData_FindFieldInTypeP12tMD_TypeDef_Pc(i64, i64)
declare i64 @_Z35MetaData_GetResolutionScopeMetaDataP10tMetaData_jPP12tMD_TypeDef_(i64, i32, i64)
declare i64 @_Z44MetaData_GetTypeDefFromFullNameAndNestedTypePcS_S_P12tMD_TypeDef_(i64, i64, i64, i64)
declare i64 @_Z32MetaData_GetTypeDefFromMethodDefP14tMD_MethodDef_(i64)
declare i64 @_Z31MetaData_GetTypeDefFromFieldDefP13tMD_FieldDef_(i64)
declare i64 @_Z37MetaData_GetMethodDefFromDefRefOrSpecP10tMetaData_jPP12tMD_TypeDef_S3_(i64, i32, i64, i64)
declare i64 @_Z32MetaData_GetFieldDefFromDefOrRefP10tMetaData_jPP12tMD_TypeDef_S3_(i64, i32, i64, i64)
declare i64 @_Z27MetaData_GetTypeMethodFieldP10tMetaData_jPjPP12tMD_TypeDef_S4_(i64, i32, i64, i64, i64)
declare i64 @_Z19MetaData_GetImplMapP10tMetaData_j(i64, i32)
declare i64 @_Z25MetaData_GetModuleRefNameP10tMetaData_j(i64, i32)
declare i64 @_Z18MethodState_DirectP8tThread_P14tMD_MethodDef_P13tMethodState_j(i64, i64, i64, i32)
declare i64 @_Z11MethodStateP8tThread_P10tMetaData_jP13tMethodState_(i64, i64, i32, i64)
declare void @_Z18MethodState_DeleteP8tThread_PP13tMethodState_(i64, i64)
declare i64 @_Z19PInvoke_GetFunctionP10tMetaData_P12tMD_ImplMap_(i64, i64)
declare i64 @_Z3RVAv()
declare i64 @_Z10RVA_CreateP5tRVA_PvS1_(i64, i64, i64)
declare i64 @_Z12RVA_FindDataP5tRVA_j(i64, i32)
declare void @_Z5CrashPKcz(i64, i8)
declare void @_Z5log_fjPKcz(i32, i64, i8)
declare i64 @_Z17Sys_GetMethodDescP14tMD_MethodDef_(i64)
declare i64 @_Z13mallocForeverj(i32)
declare i64 @_Z6msTimev()
declare i64 @_Z9microTimev()
declare void @_Z7SleepMSj(i32)
declare i64 @_Z21SystemArray_NewVectorP12tMD_TypeDef_jPj(i64, i32, i64)
declare i64 @_Z31System_Array_Internal_GetLengthPhS_S_(i64, i64, i64)
declare i64 @_Z30System_Array_Internal_GetValuePhS_S_(i64, i64, i64)
declare i64 @_Z30System_Array_Internal_SetValuePhS_S_(i64, i64, i64)
declare i64 @_Z21System_Array_GetValuePhS_S_(i64, i64, i64)
declare i64 @_Z21System_Array_SetValuePhS_S_(i64, i64, i64)
declare i64 @_Z18System_Array_ClearPhS_S_(i64, i64, i64)
declare i64 @_Z26System_Array_Internal_CopyPhS_S_(i64, i64, i64)
declare i64 @_Z19System_Array_ResizePhS_S_(i64, i64, i64)
declare i64 @_Z20System_Array_ReversePhS_S_(i64, i64, i64)
declare void @_Z24SystemArray_StoreElementPhjS_(i64, i32, i64)
declare void @_Z23SystemArray_LoadElementPhjS_(i64, i32, i64)
declare void @_Z30SystemArray_LoadElementIndicesPhPyS0_(i64, i64, i64)
declare void @_Z31SystemArray_StoreElementIndicesPhPyS0_(i64, i64, i64)
declare void @_Z37SystemArray_LoadElementIndicesAddressPhPyPS_(i64, i64, i64)
declare i64 @_Z30SystemArray_LoadElementAddressPhj(i64, i32)
declare i32 @_Z23SystemArray_GetNumBytesPhP12tMD_TypeDef_(i64, i64)
declare i32 @_Z19SystemArray_GetRankPh(i64)
declare void @_Z19SystemArray_SetRankPhi(i64, i32)
declare i64 @_Z19SystemArray_GetDimsPh(i64)
declare i64 @_Z30System_Char_GetUnicodeCategoryPhS_S_(i64, i64, i64)
declare i64 @_Z28System_Char_ToLowerInvariantPhS_S_(i64, i64, i64)
declare i64 @_Z28System_Char_ToUpperInvariantPhS_S_(i64, i64, i64)
declare i64 @_Z20System_Console_WritePhS_S_(i64, i64, i64)
declare i64 @_Z31System_Console_Internal_ReadKeyPhS_S_(i64, i64, i64)
declare i64 @_Z36System_Console_Internal_KeyAvailablePhS_S_(i64, i64, i64)
declare i64 @_Z30System_DateTime_InternalUtcNowPhS_S_(i64, i64, i64)
declare i64 @_Z33System_Diagnostics_Debugger_BreakPhS_S_(i64, i64, i64)
declare i64 @_Z29System_Enum_Internal_GetValuePhS_S_(i64, i64, i64)
declare i64 @_Z28System_Enum_Internal_GetInfoPhS_S_(i64, i64, i64)
declare i64 @_Z32System_Environment_get_TickCountPhS_S_(i64, i64, i64)
declare i64 @_Z37System_Environment_GetOSVersionStringPhS_S_(i64, i64, i64)
declare i64 @_Z31System_Environment_get_PlatformPhS_S_(i64, i64, i64)
declare i64 @_Z17System_GC_CollectPhS_S_(i64, i64, i64)
declare i64 @_Z34System_GC_Internal_CollectionCountPhS_S_(i64, i64, i64)
declare i64 @_Z24System_GC_GetTotalMemoryPhS_S_(i64, i64, i64)
declare i64 @_Z26System_GC_SuppressFinalizePhS_S_(i64, i64, i64)
declare i64 @_Z27System_IO_FileInternal_OpenPhS_S_(i64, i64, i64)
declare i64 @_Z27System_IO_FileInternal_ReadPhS_S_(i64, i64, i64)
declare i64 @_Z28System_IO_FileInternal_ClosePhS_S_(i64, i64, i64)
declare i64 @_Z42System_IO_FileInternal_GetCurrentDirectoryPhS_S_(i64, i64, i64)
declare i64 @_Z40System_IO_FileInternal_GetFileAttributesPhS_S_(i64, i64, i64)
declare i64 @_Z43System_IO_FileInternal_GetFileSystemEntriesPhS_S_(i64, i64, i64)
declare i64 @_Z15System_Math_SinPhS_S_(i64, i64, i64)
declare i64 @_Z15System_Math_CosPhS_S_(i64, i64, i64)
declare i64 @_Z15System_Math_TanPhS_S_(i64, i64, i64)
declare i64 @_Z15System_Math_PowPhS_S_(i64, i64, i64)
declare i64 @_Z16System_Math_SqrtPhS_S_(i64, i64, i64)
declare i64 @_Z34System_Net_Dns_Internal_GetHostEntPhS_S_(i64, i64, i64)
declare void @_Z11Socket_Initv()
declare i64 @_Z40System_Net_Sockets_Internal_CreateSocketPhS_S_(i64, i64, i64)
declare i64 @_Z32System_Net_Sockets_Internal_BindPhS_S_(i64, i64, i64)
declare i64 @_Z33System_Net_Sockets_Internal_ClosePhS_S_(i64, i64, i64)
declare i64 @_Z34System_Net_Sockets_Internal_ListenPhS_S_(i64, i64, i64)
declare i64 @_Z34System_Net_Sockets_Internal_AcceptPhS_S_(i64, i64, i64)
declare i64 @_Z35System_Net_Sockets_Internal_ConnectPhS_S_(i64, i64, i64)
declare i64 @_Z35System_Net_Sockets_Internal_ReceivePhS_S_(i64, i64, i64)
declare i64 @_Z32System_Net_Sockets_Internal_SendPhS_S_(i64, i64, i64)
declare i64 @_Z20System_Object_EqualsPhS_S_(i64, i64, i64)
declare i64 @_Z19System_Object_ClonePhS_S_(i64, i64, i64)
declare i64 @_Z25System_Object_GetHashCodePhS_S_(i64, i64, i64)
declare i64 @_Z21System_Object_GetTypePhS_S_(i64, i64, i64)
declare i64 @_Z47System_Runtime_CompilerServices_InitializeArrayPhS_S_(i64, i64, i64)
declare i64 @_Z15RuntimeType_NewP12tMD_TypeDef_(i64)
declare i64 @_Z27System_RuntimeType_get_NamePhS_S_(i64, i64, i64)
declare i64 @_Z32System_RuntimeType_get_NamespacePhS_S_(i64, i64, i64)
declare i64 @_Z39System_RuntimeType_GetNestingParentTypePhS_S_(i64, i64, i64)
declare i64 @_Z31System_RuntimeType_get_BaseTypePhS_S_(i64, i64, i64)
declare i64 @_Z29System_RuntimeType_get_IsEnumPhS_S_(i64, i64, i64)
declare i64 @_Z36System_RuntimeType_get_IsGenericTypePhS_S_(i64, i64, i64)
declare i64 @_Z52System_RuntimeType_Internal_GetGenericTypeDefinitionPhS_S_(i64, i64, i64)
declare i64 @_Z38System_RuntimeType_GetGenericArgumentsPhS_S_(i64, i64, i64)
declare i64 @_Z17RuntimeType_DeRefPh(i64)
declare i64 @_Z28System_String_ctor_CharInt32PhS_S_(i64, i64, i64)
declare i64 @_Z24System_String_ctor_CharAPhS_S_(i64, i64, i64)
declare i64 @_Z30System_String_ctor_CharAIntIntPhS_S_(i64, i64, i64)
declare i64 @_Z30System_String_ctor_CharAIntIntPhS_S_(i64, i64, i64)
declare i64 @_Z31System_String_ctor_StringIntIntPhS_S_(i64, i64, i64)
declare i64 @_Z23System_String_get_CharsPhS_S_(i64, i64, i64)
declare i64 @_Z28System_String_InternalConcatPhS_S_(i64, i64, i64)
declare i64 @_Z26System_String_InternalTrimPhS_S_(i64, i64, i64)
declare i64 @_Z20System_String_EqualsPhS_S_(i64, i64, i64)
declare i64 @_Z25System_String_GetHashCodePhS_S_(i64, i64, i64)
declare i64 @_Z29System_String_InternalReplacePhS_S_(i64, i64, i64)
declare i64 @_Z29System_String_InternalIndexOfPhS_S_(i64, i64, i64)
declare i64 @_Z32System_String_InternalIndexOfAnyPhS_S_(i64, i64, i64)
declare i64 @_Z28SystemString_FromUserStringsP10tMetaData_j(i64, i32)
declare i64 @_Z29SystemString_FromCharPtrASCIIPc(i64)
declare i64 @_Z29SystemString_FromCharPtrUTF16Pt(i64)
declare i64 @_Z38Internal_SystemString_FromCharPtrUTF16iPt(i32, i64)
declare i64 @_Z22SystemString_GetStringPhPj(i64, i64)
declare i32 @_Z24SystemString_GetNumBytesPh(i64)
declare i64 @_Z50System_Threading_Interlocked_CompareExchange_Int32PhS_S_(i64, i64, i64)
declare i64 @_Z44System_Threading_Interlocked_Increment_Int32PhS_S_(i64, i64, i64)
declare i64 @_Z44System_Threading_Interlocked_Decrement_Int32PhS_S_(i64, i64, i64)
declare i64 @_Z38System_Threading_Interlocked_Add_Int32PhS_S_(i64, i64, i64)
declare i64 @_Z43System_Threading_Interlocked_Exchange_Int32PhS_S_(i64, i64, i64)
declare i64 @_Z42System_Threading_Monitor_Internal_TryEnterPhS_S_(i64, i64, i64)
declare i64 @_Z38System_Threading_Monitor_Internal_ExitPhS_S_(i64, i64, i64)
declare i64 @_Z28System_Threading_Thread_ctorPhS_S_(i64, i64, i64)
declare i64 @_Z33System_Threading_Thread_ctorParamPhS_S_(i64, i64, i64)
declare i64 @_Z29System_Threading_Thread_StartPhS_S_(i64, i64, i64)
declare i64 @_Z29System_Threading_Thread_SleepPhS_S_(i64, i64, i64)
declare i64 @_Z41System_Threading_Thread_get_CurrentThreadPhS_S_(i64, i64, i64)
declare i64 @_Z29System_Type_GetTypeFromHandlePhS_S_(i64, i64, i64)
declare i64 @_Z27System_Type_get_IsValueTypePhS_S_(i64, i64, i64)
declare i64 @_Z26System_ValueType_GetFieldsPhS_S_(i64, i64, i64)
declare i64 @_Z31System_WeakReference_get_TargetPhS_S_(i64, i64, i64)
declare i64 @_Z31System_WeakReference_set_TargetPhS_S_(i64, i64, i64)
declare void @_Z30SystemWeakReference_TargetGonePPhj(i64, i32)
declare i64 @_Z6Threadv()
declare i64 @_Z17Thread_StackAllocP8tThread_j(i64, i32)
declare void @_Z16Thread_StackFreeP8tThread_Pv(i64, i64)
declare void @_Z20Thread_SetEntryPointP8tThread_P10tMetaData_jPhj(i64, i64, i32, i64, i32)
declare i32 @_Z14Thread_Executev()
declare i64 @_Z17Thread_GetCurrentv()
declare void @_Z19Thread_GetHeapRootsP11tHeapRoots_(i64)
declare i64 @_Z20Type_GetArrayTypeDefP12tMD_TypeDef_iPS0_S1_(i64, i32, i64, i64)
declare i32 @_Z16Type_IsValueTypeP12tMD_TypeDef_(i64)
declare i64 @_Z19Type_GetTypeFromSigP10tMetaData_PPhPP12tMD_TypeDef_S5_(i64, i64, i64, i64)
declare void @_Z9Type_Initv()
declare i32 @_Z13Type_IsMethodP14tMD_MethodDef_PcP12tMD_TypeDef_jPh(i64, i64, i64, i32, i64)
declare i32 @_Z24Type_IsDerivedFromOrSameP12tMD_TypeDef_S0_(i64, i64)
declare i32 @_Z18Type_IsImplementedP12tMD_TypeDef_S0_(i64, i64)
declare i32 @_Z21Type_IsAssignableFromP12tMD_TypeDef_S0_(i64, i64)
declare i64 @_Z18Type_GetTypeObjectP12tMD_TypeDef_(i64)
declare i64 @_Z62System_Runtime_CompilerServices_RuntimeHelpers_InitializeArrayPhS_S_(i64, i64, i64)
";
        public static void Initialize()
        {
            // Load C# library for BCL, and grab all types and methods.
            // The tables that this method sets up are:
            // _substituted_bcl -- maps types in program (represented in Mono.Cecil) into GPU BCL types (represented in Mono.Cecil).
            // _system_type_to_mono_type_for_bcl -- associates types in GPU BCL with NET Core/NET Framework/... in user program.
            // Note, there seems to be an underlying bug in System.Type.GetType for certain generics, like System.Collections.Generic.HashSet.
            // The method returns null.

            // Set up _substituted_bcl.
            var runtime = new RUNTIME();

            // Find corlib.dll. It could be anywhere, but let's check the usual spots.
            Mono.Cecil.ModuleDefinition md = Campy.Meta.StickyReadMod.StickyReadModule(FindCoreLib());
            foreach (var bcl_type in md.GetTypes())
            {
                // Filter out <Module> and <PrivateImplementationDetails>, among possible others.
                Regex regex = new Regex(@"^[<]\w+[>]");
                if (regex.IsMatch(bcl_type.FullName)) continue;
                foreach (var m in bcl_type.Methods)
                {
                    var x = m.ImplAttributes;
                    if ((x & MethodImplAttributes.InternalCall) != 0)
                    {
                        if (Campy.Utils.Options.IsOn("runtime_trace"))
                            System.Console.WriteLine("Internal call set up " + bcl_type + " " + m);
                        _bcl_runtime_csharp_methods_labeled_internal.Add(new BclNativeMethod(bcl_type, m));
                    }
                }
            }

            // Set up _system_type_to_mono_type_for_bcl.
            // There really isn't any good way to set this up because NET Core System.Reflection does not work
            // on .LIB files. So begins the kludge...
            // set up decls of runtime from above string.

            var assembly = Assembly.GetAssembly(typeof(Campy.Meta.RUNTIME));
            List<MethodReference> bcl_runtime_csharp_methods = new List<MethodReference>();

            Mono.Cecil.ModuleDefinition campy_bcl_runtime = Campy.Meta.StickyReadMod.StickyReadModule(RUNTIME.FindCoreLib());
            foreach (var tt in campy_bcl_runtime.Types)
            {
                all_types.Add(tt.FullName, tt);
            }

            Stack<TypeReference> consider_list = new Stack<TypeReference>();
            Stack<TypeReference> types_in_bcl = new Stack<TypeReference>();
            foreach (var type in campy_bcl_runtime.Types)
            {
                consider_list.Push(type);
            }
            while (consider_list.Any())
            {
                var type = consider_list.Pop();
                types_in_bcl.Push(type);
                var r = type.Resolve();
                if (r == null) continue;
                foreach (var nested in r.NestedTypes)
                    consider_list.Push(nested);
            }
            while (types_in_bcl.Any())
            {
                var type = types_in_bcl.Pop();
                var r = type.Resolve();
                if (r == null) continue;
                foreach (var method in r.Methods)
                {
                    // add into list of method calls for BCL.
                    bcl_runtime_csharp_methods.Add(method);
                }
            }

            using (Stream stream = new MemoryStream(Encoding.UTF8.GetBytes(runtime_decls)))
            using (StreamReader reader = new StreamReader(stream))
            {
                string decls = reader.ReadToEnd();
                // Parse the declarations for ptx visible functions.
                // This should match "declare" <spaces> <function-return>? <function-name>
                // over many lines, many times.
                Regex regex = new Regex(
      @"declare\s+(?<return>\w+)\s+[@](?<name>\w+)\s*[(](?<params>[^)]*[)]\s*)");
                foreach (Match match in regex.Matches(decls))
                {
                    Regex space = new Regex(@"\s\s+");
                    string mangled_name = match.Groups["name"].Value.Trim();
                    string return_type = match.Groups["return"].Value.Trim();
                    return_type = space.Replace(return_type, " ");
                    string parameters = match.Groups["params"].Value.Trim();
                    parameters = space.Replace(parameters, " ");

                    if (Campy.Utils.Options.IsOn("runtime_trace"))
                        System.Console.WriteLine(mangled_name + " " + return_type + " " + parameters);

                    if (RUNTIME._bcl_runtime_csharp_internal_to_valueref.ContainsKey(mangled_name)) continue;

                    TypeRef llvm_return_type = default(TypeRef);
                    TypeRef[] args;

                    // Construct LLVM extern that corresponds to type of function.
                    // Parse return_type and parameters strings...
                    {
                        // parse return.
                        if (return_type == "" || return_type == "void")
                            llvm_return_type = LLVM.VoidType();
                        else
                        {
                            _ptx_type_to_llvm_typeref.TryGetValue(
                                return_type, out TypeRef y);
                            if (Campy.Utils.Options.IsOn("runtime_trace"))
                                System.Console.WriteLine("  value " + y.ToString());
                            llvm_return_type = y;
                        }
                    }

                    {
                        Regex param_regex = new Regex(@"(?<type>[^,)]+)[,)]");
                        // parse parameters.
                        List<TypeRef> args_list = new List<TypeRef>();
                        foreach (Match ret in param_regex.Matches(parameters))
                        {
                            var x = ret.Groups["type"].Value;
                            x = x.Trim();
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
                        RUNTIME.global_llvm_module,
                        mangled_name,
                        LLVM.FunctionType(
                            llvm_return_type,
                            args,
                            false));

                    PtxFunction ptxf = new PtxFunction(mangled_name, decl);
                    if (Campy.Utils.Options.IsOn("runtime_trace"))
                        System.Console.WriteLine(ptxf._mangled_name
                                             + " "
                                             + ptxf._valueref);

                    RUNTIME._bcl_runtime_csharp_internal_to_valueref.Add(mangled_name, decl);
                    _ptx_functions.Add(ptxf);
                }
            }
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
            Mono.Cecil.ModuleDefinition md = Campy.Meta.StickyReadMod.StickyReadModule(FindCoreLib());
            foreach (var bcl_type in md.GetTypes())
            {
                if (bcl_type.FullName == type.FullName)
                    return bcl_type;
            }
            return result;
        }

        public static CUfunction _Z15Set_BCL_GlobalsP6_BCL_t(CUmodule module)
        {
            CudaHelpers.CheckCudaError(Cuda.cuModuleGetFunction(out CUfunction function, module, "_Z15Set_BCL_GlobalsP6_BCL_t"));
            return function;
        }

        public static IntPtr BclPtr { get; set; }
        public static ulong BclPtrSize { get; set; }

        private static Dictionary<TypeReference, IntPtr> _type_to_bcltype = new Dictionary<TypeReference, IntPtr>();
        public static IntPtr MonoBclMap_GetBcl(TypeReference type)
        {
            if (_type_to_bcltype.ContainsKey(type))
                return _type_to_bcltype[type];

            // Using the BCL, find the type. Note, Mono has a tendency to list something
            // in a namespace in which the type's metadata says it has no namespace.
            // As far as I can tell, this is for generated methods corresponding to the kernels.
            // Due to the disparity of what the "namespace" means, look up the type from
            // the top-most declaring type, and use that as context for the sub-class search.
            Stack<TypeReference> chain = new Stack<TypeReference>();
            while (type != null)
            {
                chain.Push(type);
                type = type.DeclaringType;
            }
            System.IntPtr result = System.IntPtr.Zero;
            while (chain.Any())
            {
                var tr = chain.Pop();
                var mt = tr.SwapInBclType();
                var mt_assembly_name = mt.Scope.Name;
                var mt_name_space = mt.Namespace;
                var mt_name = mt.Name;

                if (mt.IsGenericInstance)
                {
                    result = BclGetMetaOfType(mt_assembly_name, mt_name_space, mt_name, result);
                    // Look up each argument type of generic instance and construct array of these args.
                    var g_mt = mt as GenericInstanceType;
                    var generic_arguments = g_mt.GenericArguments;
                    var count = generic_arguments.Count;
                    System.IntPtr[] args = new System.IntPtr[count];
                    for (int i = 0; i < count; ++i)
                        args[i] = (System.IntPtr)MonoBclMap_GetBcl(generic_arguments[i]);
                    RUNTIME.BclCheckHeap();
                    result = BclConstructGenericInstanceType(result, count, args);
                }
                else if (mt.IsArray)
                {
                    var a = mt as ArrayType;
                    var et = a.ElementType;
                    var bcl_et = MonoBclMap_GetBcl(et);
                    result = BclConstructArrayType(bcl_et, a.Rank);
                }
                else
                {
                    result = BclGetMetaOfType(mt_assembly_name, mt_name_space, mt_name, result);
                }
                if (!_type_to_bcltype.Where(t => t.Key.FullName == mt.FullName).Any())
                {
                    _type_to_bcltype[mt] = result;
                }
            }
            return result;
        }
        public static TypeReference MonoBclMap_GetMono(IntPtr bcl_type)
        {
            var possible = _type_to_bcltype.Where(t => t.Value == bcl_type);
            if (!possible.Any()) return null;
            return possible.First().Key;
        }
    }
}



