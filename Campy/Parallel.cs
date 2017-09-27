using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Campy.Types;
using Campy.Compiler;
using Campy.LCFG;
using Campy.Types.Utils;
using Mono.Cecil;
using Swigged.Cuda;
using Type = System.Type;

namespace Campy
{
    public class Parallel
    {
        private static Parallel _singleton;
        private CFG _graph;
        private Reader _reader;
        private Converter _converter;

        private Parallel()
        {
            Swigged.LLVM.Helper.Adjust.Path();
            Cuda.cuInit(0);
            _reader = new Reader();
            _graph = _reader.Cfg;
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

        public static void For(int number_of_threads, _Kernel_type kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, number_of_threads, kernel);
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        static public unsafe void For(AcceleratorView view, int number_of_threads, _Kernel_type kernel)
        {

            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            try
            {
                // Parse kernel instructions to determine basic block representation of all the code to compile.
                int change_set_id = Singleton()._graph.StartChangeSet();
                Singleton()._reader.AnalyzeMethod(kernel);
                List<CFG.Vertex> cs = Singleton()._graph.PopChangeSet(change_set_id);

                MethodInfo method = kernel.Method;
                object target = kernel.Target;

                // Get basic block of entry.
                CFG.Vertex bb;
                if (!cs.Any())
                {
                    // Compiled previously. Look for basic block of entry.
                    CFG.Vertex vvv = Singleton()._graph.Entries.Where(v =>
                        v.IsEntry && v.Method.Name == method.Name).FirstOrDefault();

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
                Dictionary<TypeReference, Type> ops = bb.OpsFromOriginal;

                // Compile methods with added type information.
                Singleton()._converter.CompileToLLVM(cs, list_of_mono_data_types_used);

                var ptr_to_kernel = Singleton()._converter.GetCudaFunction(bb.Name);

                Index index = new Index(number_of_threads);
                Buffers buffer = new Buffers();

                // Set up parameters.
                int count = kernel.Method.GetParameters().Length;
                if (bb.HasThis) count++;
                if (!(count == 1 || count == 2)) throw new Exception("Expecting at least one parameter for kernel.");

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

                IntPtr[] kp = new IntPtr[] {pointer1, pointer2};
                var res = CUresult.CUDA_SUCCESS;
                fixed (IntPtr* kernelParams = kp)
                {
                    linear_to_tile(number_of_threads, out dim3 tile_size, out dim3 tiles);
                    //linear_to_tile(1, out dim3 tile_size, out dim3 tiles);

                    res = Cuda.cuLaunchKernel(
                        ptr_to_kernel,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(CUstream),
                        (IntPtr) kernelParams,
                        (IntPtr) IntPtr.Zero
                    );
                }
                Converter.CheckCudaError(res);
                res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                Converter.CheckCudaError(res);
                buffer.DeepCopyFromImplementation(ptr, out object to, kernel.Target.GetType());
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }
            finally
            {
                handle1.Free();
                handle2.Free();
            }
        }

        static public void For(TiledExtent extent, _Kernel_tiled_type kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, extent, kernel);
        }

        static public void For(AcceleratorView view, TiledExtent extent, _Kernel_tiled_type kernel)
        {
        }

        struct dim3
        {
            public uint x;
            public uint y;
            public uint z;
        }

        private static void linear_to_tile(int size,
            out dim3 tile_size, out dim3 tiles)
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
                make_results(blocks, threads, max_dimensionality, out tile_size, out tiles);
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
                make_results(blocks, threads, max_dimensionality, out tile_size, out tiles);
                return;
            }
            throw new Exception();
        }

        private static void make_results(int[] blocks, int[] threads, int max_dimensionality,
            out dim3 tile_size, out dim3 tiles)
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