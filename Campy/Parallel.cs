using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Campy.Compiler;
using Campy.Utils;
using Mono.Cecil;
using Swigged.Cuda;
using Type = System.Type;

namespace Campy
{
    public class Parallel
    {
        private static Parallel _singleton;
        private CFG _graph;
        private Importer _importer;
        private Converter _converter;

        private Parallel()
        {
            Swigged.LLVM.Helper.Adjust.Path();
            _importer = new Importer();
            _graph = _importer.Cfg;
            _converter = new Campy.Compiler.Converter(_graph);
           // var ok = GC.TryStartNoGCRegion(200000000);
        }

        public static Parallel Singleton()
        {
            if (_singleton == null)
            {
                _singleton = new Parallel();
            }
            return _singleton;
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        public static void For(int number_of_threads, KernelType kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, new Extent(number_of_threads), kernel);
        }

        //public static void For(int inclusive_start, int exclusive_end, KernelType kernel)
        //{
        //    AcceleratorView view = Accelerator.GetAutoSelectionView();
        //    For(view, inclusive_start, exclusive_end, kernel);
        //}

        public static void For(Extent extent, KernelType kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, extent, kernel);
        }

        private static void For(AcceleratorView view, int inclusive_start, int exclusive_end, KernelType kernel)
        {

            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            try
            {
                unsafe
                {
                    // Parse kernel instructions to determine basic block representation of all the code to compile.
                    int change_set_id = Singleton()._graph.StartChangeSet();
                    Singleton()._importer.AnalyzeMethod(kernel.Method);
                    if (Singleton()._importer.Failed)
                    {
                        throw new Exception("Failure to find all methods in GPU code. Cannot continue.");
                    }
                    List<CFG.Vertex> cs = Singleton()._graph.PopChangeSet(change_set_id);

                    MethodInfo method = kernel.Method;
                    object target = kernel.Target;

                    // Get basic block of entry.
                    CFG.Vertex bb;
                    if (!cs.Any())
                    {
                        // Compiled previously. Look for basic block of entry.
                        CFG.Vertex vvv = Singleton()._graph.Entries.Where(v =>
                            v.IsEntry && v.ExpectedCalleeSignature.Name == method.Name).FirstOrDefault();

                        bb = vvv;
                    }
                    else
                    {
                        bb = cs.First();
                    }

                    // Very important note: Although we have the control flow graph of the code that is to
                    // be compiled, there is going to be generics used, e.g., ArrayView<int>, within the body
                    // of the code and in the called runtime library. We need to record the types for compiling
                    // and add that to compilation.
                    // https://stackoverflow.com/questions/5342345/how-do-generics-get-compiled-by-the-jit-compiler

                    // Create a list of generics called with types passed.
                    List<Type> list_of_data_types_used = new List<Type>();
                    list_of_data_types_used.Add(target.GetType());

                    // Convert list into Mono data types.
                    List<Mono.Cecil.TypeReference> list_of_mono_data_types_used = new List<TypeReference>();
                    foreach (System.Type data_type_used in list_of_data_types_used)
                    {
                        list_of_mono_data_types_used.Add(
                            data_type_used.ToMonoTypeReference());
                    }

                    // In the same, in-order discovery of all methods, we're going to pass on
                    // type information. As we spread the type info from basic block to successors,
                    // copy the node with the type information associated with it if the type info
                    // results in a different interpretation/compilation of the function.
                    cs = Singleton()._converter.InstantiateGenerics(
                        cs, list_of_data_types_used, list_of_mono_data_types_used);

                    // Associate "this" with entry.
                    Dictionary<Tuple<TypeReference, GenericParameter>, Type> ops = bb.OpsFromOriginal;

                    // Compile methods with added type information.
                    string ptx = Singleton()._converter.CompileToLLVM(cs, list_of_mono_data_types_used,
                        bb.Name, inclusive_start);

                    Converter.CheckCudaError(Cuda.cuInit(0));
                    Converter.CheckCudaError(Cuda.cuCtxGetCurrent(out CUcontext pctx));
                    


                    var ptr_to_kernel = Singleton()._converter.GetCudaFunction(bb.Name, ptx);

                    Buffers buffer = new Buffers();

                    // Set up parameters.
                    int count = kernel.Method.GetParameters().Length;
                    if (bb.HasThis) count++;
                    if (!(count == 1 || count == 2))
                        throw new Exception("Expecting at least one parameter for kernel.");

                    IntPtr[] parm1 = new IntPtr[1];
                    IntPtr[] parm2 = new IntPtr[1];
                    IntPtr ptr = IntPtr.Zero;

                    if (bb.HasThis)
                    {
                        Type type = kernel.Target.GetType();
                        Type btype = buffer.CreateImplementationType(type);
                        ptr = buffer.New(Buffers.SizeOf(btype));
                        buffer.DeepCopyToImplementation(kernel.Target, ptr);
                        parm1[0] = ptr;
                    }

                    {
                        Type btype = buffer.CreateImplementationType(typeof(Index));
                        var s = Buffers.SizeOf(btype);
                        var ptr2 = buffer.New(s);
                        // buffer.DeepCopyToImplementation(index, ptr2);
                        parm2[0] = ptr2;
                    }

                    IntPtr[] x1 = parm1;
                    handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                    IntPtr pointer1 = handle1.AddrOfPinnedObject();

                    IntPtr[] x2 = parm2;
                    handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                    IntPtr pointer2 = handle2.AddrOfPinnedObject();

                    // Compute number of total number of threads.
                    int number_of_threads = exclusive_end - inclusive_start;

                    IntPtr[] kp = new IntPtr[] { pointer1, pointer2 };
                    var res = CUresult.CUDA_SUCCESS;
                    fixed (IntPtr* kernelParams = kp)
                    {
                        //MakeLinearTiling(number_of_threads, out dim3 tile_size, out dim3 tiles);
                        MakeLinearTiling(1, out dim3 tile_size, out dim3 tiles);

                        res = Cuda.cuLaunchKernel(
                            ptr_to_kernel,
                            tiles.x, tiles.y, tiles.z, // grid has one block.
                            tile_size.x, tile_size.y, tile_size.z, // n threads.
                            0, // no shared memory
                            default(CUstream),
                            (IntPtr)kernelParams,
                            (IntPtr)IntPtr.Zero
                        );
                    }
                    Converter.CheckCudaError(res);
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    Converter.CheckCudaError(res);
                    buffer.DeepCopyFromImplementation(ptr, out object to, kernel.Target.GetType());
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw e;
            }
            finally
            {
                handle1.Free();
                handle2.Free();
            }
        }
        private static void For(AcceleratorView view, Extent extent, KernelType kernel)
        {

            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            try
            {
                unsafe
                {
                    Stopwatch stopwatch_discovery = new Stopwatch();
                    stopwatch_discovery.Reset();
                    stopwatch_discovery.Start();

                    // Parse kernel instructions to determine basic block representation of all the code to compile.
                    int change_set_id = Singleton()._graph.StartChangeSet();
                    Singleton()._importer.AnalyzeMethod(kernel.Method);
                    if (Singleton()._importer.Failed)
                    {
                        throw new Exception("Failure to find all methods in GPU code. Cannot continue.");
                    }
                    List<CFG.Vertex> cs = Singleton()._graph.PopChangeSet(change_set_id);

                    stopwatch_discovery.Stop();
                    var elapse_discovery = stopwatch_discovery.Elapsed;

                    MethodInfo method = kernel.Method;
                    object target = kernel.Target;

                    // Get basic block of entry.
                    CFG.Vertex bb;
                    if (!cs.Any())
                    {
                        // Compiled previously. Look for basic block of entry.
                        CFG.Vertex vvv = Singleton()._graph.Entries.Where(v =>
                            v.IsEntry && v.ExpectedCalleeSignature.Name == method.Name).FirstOrDefault();

                        bb = vvv;
                    }
                    else
                    {
                        bb = cs.First();
                    }


                    // Very important note: Although we have the control flow graph of the code that is to
                    // be compiled, there is going to be generics used, e.g., ArrayView<int>, within the body
                    // of the code and in the called runtime library. We need to record the types for compiling
                    // and add that to compilation.
                    // https://stackoverflow.com/questions/5342345/how-do-generics-get-compiled-by-the-jit-compiler

                    // Create a list of generics called with types passed.
                    List<Type> list_of_data_types_used = new List<Type>();
                    list_of_data_types_used.Add(target.GetType());
                    //Singleton._converter.FindAllTargets(kernel));

                    // Convert list into Mono data types.
                    List<Mono.Cecil.TypeReference> list_of_mono_data_types_used = new List<TypeReference>();
                    foreach (System.Type data_type_used in list_of_data_types_used)
                    {
                        list_of_mono_data_types_used.Add(
                            data_type_used.ToMonoTypeReference());
                    }

                    // In the same, in-order discovery of all methods, we're going to pass on
                    // type information. As we spread the type info from basic block to successors,
                    // copy the node with the type information associated with it if the type info
                    // results in a different interpretation/compilation of the function.
                    cs = Singleton()._converter.InstantiateGenerics(
                        cs, list_of_data_types_used, list_of_mono_data_types_used);

                    // Associate "this" with entry.
                    Dictionary<Tuple<TypeReference, GenericParameter>, Type> ops = bb.OpsFromOriginal;

                    var stopwatch_compiler = new Stopwatch();
                    stopwatch_compiler.Reset();
                    stopwatch_compiler.Start();

                    // Compile methods with added type information.
                    string ptx = Singleton()._converter.CompileToLLVM(cs, list_of_mono_data_types_used,
                        bb.Name);

                    stopwatch_compiler.Stop();
                    var elapse_compiler = stopwatch_compiler.Elapsed;

                    Converter.CheckCudaError(Cuda.cuInit(0));
                    Converter.CheckCudaError(Cuda.cuDevicePrimaryCtxReset(0));
                    Converter.CheckCudaError(Cuda.cuCtxCreate_v2(out CUcontext pctx, 0, 0));
                    Converter.CheckCudaError(Cuda.cuMemGetInfo_v2(out ulong free_memory, out ulong total_memory));
                    System.Console.WriteLine("total memory " + total_memory + " free memory " + free_memory);
                    Converter.CheckCudaError(Cuda.cuCtxGetLimit(out ulong pvalue, CUlimit.CU_LIMIT_STACK_SIZE));
                    Converter.CheckCudaError(Cuda.cuCtxSetLimit(CUlimit.CU_LIMIT_STACK_SIZE, (uint)pvalue*25));
                    System.Console.WriteLine("Stack size " + pvalue);
                    var stopwatch_cuda_compile = new Stopwatch();
                    stopwatch_cuda_compile.Reset();
                    stopwatch_cuda_compile.Start();

                    var ptr_to_kernel = Singleton()._converter.GetCudaFunction(bb.Name, ptx);

                    stopwatch_cuda_compile.Start();
                    var elapse_cuda_compile = stopwatch_cuda_compile.Elapsed;

                    Index index = new Index(extent);
                    Buffers buffer = new Buffers();

                    var stopwatch_deep_copy_to = new Stopwatch();
                    stopwatch_deep_copy_to.Reset();
                    stopwatch_deep_copy_to.Start();

                    // Set up parameters.
                    int count = kernel.Method.GetParameters().Length;
                    if (bb.HasThis) count++;
                    if (!(count == 1 || count == 2))
                        throw new Exception("Expecting at least one parameter for kernel.");

                    IntPtr[] parm1 = new IntPtr[1];
                    IntPtr[] parm2 = new IntPtr[1];
                    IntPtr ptr = IntPtr.Zero;

                    if (bb.HasThis)
                    {
                        Type type = kernel.Target.GetType();
                        Type btype = buffer.CreateImplementationType(type);
                        ptr = buffer.New(Buffers.SizeOf(btype));
                        buffer.DeepCopyToImplementation(kernel.Target, ptr);
                        parm1[0] = ptr;
                    }

                    {
                        Type btype = buffer.CreateImplementationType(typeof(Index));
                        var s = Buffers.SizeOf(btype);
                        var ptr2 = buffer.New(s);
                        // buffer.DeepCopyToImplementation(index, ptr2);
                        parm2[0] = ptr2;
                    }

                    stopwatch_deep_copy_to.Start();
                    var elapse_deep_copy_to = stopwatch_cuda_compile.Elapsed;

                    var stopwatch_call_kernel = new Stopwatch();
                    stopwatch_call_kernel.Reset();
                    stopwatch_call_kernel.Start();

                    IntPtr[] x1 = parm1;
                    handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                    IntPtr pointer1 = handle1.AddrOfPinnedObject();

                    IntPtr[] x2 = parm2;
                    handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                    IntPtr pointer2 = handle2.AddrOfPinnedObject();

                    // Compute number of total number of threads.
                    int number_of_threads = extent.Aggregate(1, (acc, x) => acc * x);

                    IntPtr[] kp = new IntPtr[] { pointer1, pointer2 };
                    var res = CUresult.CUDA_SUCCESS;
                    fixed (IntPtr* kernelParams = kp)
                    {
                        MakeLinearTiling(number_of_threads, out dim3 tile_size, out dim3 tiles);

                        //MakeLinearTiling(1, out dim3 tile_size, out dim3 tiles);

                        res = Cuda.cuLaunchKernel(
                            ptr_to_kernel,
                            tiles.x, tiles.y, tiles.z, // grid has one block.
                            tile_size.x, tile_size.y, tile_size.z, // n threads.
                            0, // no shared memory
                            default(CUstream),
                            (IntPtr)kernelParams,
                            (IntPtr)IntPtr.Zero
                        );
                    }
                    Converter.CheckCudaError(res);
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    Converter.CheckCudaError(res);

                    stopwatch_call_kernel.Stop();
                    var elapse_call_kernel = stopwatch_call_kernel.Elapsed;

                    var stopwatch_deep_copy_back = new Stopwatch();
                    stopwatch_deep_copy_back.Reset();
                    stopwatch_deep_copy_back.Start();

                    buffer.DeepCopyFromImplementation(ptr, out object to, kernel.Target.GetType());

                    stopwatch_deep_copy_back.Stop();
                    var elapse_deep_copy_back = stopwatch_deep_copy_back.Elapsed;

                    Cuda.cuCtxDestroy_v2(pctx);

                    System.Console.WriteLine("discovery     " + elapse_discovery);
                    System.Console.WriteLine("compiler      " + elapse_compiler);
                    System.Console.WriteLine("cuda compile  " + elapse_cuda_compile);
                    System.Console.WriteLine("deep copy in  " + elapse_deep_copy_to);
                    System.Console.WriteLine("cuda kernel   " + elapse_call_kernel);
                    System.Console.WriteLine("deep copy out " + elapse_deep_copy_back);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw e;
            }
            finally
            {
                handle1.Free();
                handle2.Free();
            }
        }

        private static void For(TiledExtent extent, KernelTiledType kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, extent, kernel);
        }

        private static void For(AcceleratorView view, TiledExtent extent, KernelTiledType kernel)
        {
        }

        public struct dim3
        {
            public uint x;
            public uint y;
            public uint z;
        }

        public static void MakeLinearTiling(int size, out dim3 tile_size, out dim3 tiles)
        {
            int max_dimensionality = 3;
            int[] blocks = new int[10];
            for (int j = 0; j < max_dimensionality; ++j)
                blocks[j] = 1;
            int[] max_threads = new int[]{ 1024, 1024, 64 };
            int[] max_blocks = new int[] { 65535, 65535, 65535 };
            int[] threads = new int[10];
            for (int j = 0; j < max_dimensionality; ++j)
                threads[j] = 1;
            int b = size / (max_threads[0] * max_blocks[0]);
            if (b == 0)
            {
                b = size / max_threads[0];
                if (size % max_threads[0] != 0)
                    b++;

                if (b == 1)
                    max_threads[0] = size;

                // done. return the result.
                blocks[0] = b;
                threads[0] = max_threads[0];
                SetBlockAndThreadDim(blocks, threads, max_dimensionality, out tile_size, out tiles);
                return;
            }

            int sqrt_size = (int)Math.Sqrt((float)size / max_threads[0]);
            sqrt_size++;

            int b2 = sqrt_size / max_blocks[1];
            if (b2 == 0)
            {
                b = sqrt_size;

                // done. return the result.
                blocks[0] = blocks[1] = b;
                threads[0] = max_threads[0];
                SetBlockAndThreadDim(blocks, threads, max_dimensionality, out tile_size, out tiles);
                return;
            }
            throw new Exception();
        }

        private static void SetBlockAndThreadDim(int[] blocks, int[] threads, int max_dimensionality, out dim3 tile_size, out dim3 tiles)
        {
            tiles.x = (uint)blocks[0];
            tiles.y = (uint)blocks[1];
            tiles.z = (uint)blocks[2];
            tile_size.x = (uint)threads[0];
            tile_size.y = (uint)threads[1];
            tile_size.z = (uint)threads[2];
        }
    }
}