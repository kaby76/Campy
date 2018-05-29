using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Campy.Compiler;
using Campy.Utils;
using Swigged.Cuda;
using Type = System.Type;

namespace Campy
{
    public class Parallel
    {
        private static Parallel _singleton;
        private JITER _converter;
        private BUFFERS Buffer { get; }

        private Parallel()
        {
            _converter = JITER.Singleton;
            Buffer = new BUFFERS();
            //InitCuda();
            // var ok = GC.TryStartNoGCRegion(200000000);
        }

        private static Parallel Singleton()
        {
            if (_singleton == null)
            {
                _singleton = new Parallel();
            }
            return _singleton;
        }

        /// <summary>
        /// Make "obj" stay on the GPU until Sync() is called. In other words, do not
        /// copy it back to the CPU until then. "obj" should be a reference type, and used
        /// by the GPU kernel. Otherwise, it won't be copied to the GPU in the first place.
        /// The closure object for the kernel delegate is always copied to and from the GPU
        /// after each call.
        /// </summary>
        /// <param name="obj"></param>
        public static void Sticky(object obj)
        {
            Singleton().Buffer.Delay(obj);
        }

        /// <summary>
        /// Make "obj" stay on the GPU, and do not copy it back to the CPU.
        /// The closure object for the kernel delegate is always copied to and from the GPU
        /// after each call.
        /// </summary>
        /// <param name="obj"></param>
        public static void ReadOnly(object obj)
        {
            Singleton().Buffer.ReadOnly(obj);
        }

        /// <summary>
        /// Copy all "sticky" objects used in the kernel back to the CPU from the GPU.
        /// The closure object for the kernel delegate is always copied to and from the GPU
        /// after each call.
        /// </summary>
        public static void Sync()
        {
            Singleton().Buffer.FullSynch();
        }

        public static void For(int number_of_threads, SimpleKernel simpleKernel)
        {
            JITER.InitCuda();

            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            try
            {
                unsafe
                {

                    //////// COMPILE KERNEL INTO GPU CODE ///////
                    /////////////////////////////////////////////
                    var stopwatch_cuda_compile = new Stopwatch();
                    stopwatch_cuda_compile.Start();
                    IntPtr image = Singleton()._converter.JitCodeToImage(simpleKernel.Method, simpleKernel.Target);
                    CUfunction ptr_to_kernel = Singleton()._converter.GetCudaFunction(simpleKernel.Method, image);
                    var elapse_cuda_compile = stopwatch_cuda_compile.Elapsed;

                    BUFFERS.CheckHeap();

                    //////// COPY DATA INTO GPU /////////////////
                    /////////////////////////////////////////////
                    var stopwatch_deep_copy_to = new Stopwatch();
                    stopwatch_deep_copy_to.Reset();
                    stopwatch_deep_copy_to.Start();
                    BUFFERS buffer = Singleton().Buffer;

                    // Set up parameters.
                    int count = simpleKernel.Method.GetParameters().Length;
                    var bb = Singleton()._converter.GetBasicBlock(simpleKernel.Method);
                    if (bb.HasThis) count++;
                    if (!(count == 1 || count == 2))
                        throw new Exception("Expecting at least one parameter for kernel.");

                    IntPtr[] parm1 = new IntPtr[1];
                    IntPtr[] parm2 = new IntPtr[1];
                    IntPtr ptr = IntPtr.Zero;

                    // The method really should have a "this" because it's a closure
                    // object.
                    if (bb.HasThis)
                    {
                        BUFFERS.CheckHeap();
                        ptr = buffer.AddDataStructure(simpleKernel.Target);
                        parm1[0] = ptr;
                    }

                    {
                        Type btype = typeof(int);
                        var s = BUFFERS.SizeOf(btype);
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

                    BUFFERS.CheckHeap();

                    IntPtr[] kp = new IntPtr[] { pointer1, pointer2 };
                    var res = CUresult.CUDA_SUCCESS;
                    fixed (IntPtr* kernelParams = kp)
                    {
                        Campy.Utils.CudaHelpers.MakeLinearTiling(number_of_threads, out Campy.Utils.CudaHelpers.dim3 tile_size, out Campy.Utils.CudaHelpers.dim3 tiles);

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
                    CudaHelpers.CheckCudaError(res);
                    res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                    CudaHelpers.CheckCudaError(res);

                    stopwatch_call_kernel.Stop();
                    var elapse_call_kernel = stopwatch_call_kernel.Elapsed;

                    if (Campy.Utils.Options.IsOn("jit_trace"))
                    {
                        System.Console.WriteLine("cuda compile  " + elapse_cuda_compile);
                        System.Console.WriteLine("deep copy in  " + elapse_deep_copy_to);
                        System.Console.WriteLine("cuda kernel   " + elapse_call_kernel);
                    }

                    {
                        var stopwatch_deep_copy_back = new Stopwatch();
                        stopwatch_deep_copy_back.Reset();

                        BUFFERS.CheckHeap();

                        stopwatch_deep_copy_back.Start();

                        buffer.SynchDataStructures();
                        
                        stopwatch_deep_copy_back.Stop();

                        BUFFERS.CheckHeap();

                        var elapse_deep_copy_back = stopwatch_deep_copy_back.Elapsed;
                        if (Campy.Utils.Options.IsOn("jit_trace"))
                            System.Console.WriteLine("deep copy out " + elapse_deep_copy_back);
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw e;
            }
            finally
            {
                if (default(GCHandle) != handle1) handle1.Free();
                if (default(GCHandle) != handle2) handle2.Free();
            }
        }

        public static void Options(UInt64 options)
        {
            JITER.Options(options);
        }
    }
}