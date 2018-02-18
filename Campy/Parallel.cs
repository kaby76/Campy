using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        private CampyConverter _converter;
        private Buffers Buffer { get; }

        private Parallel()
        {
            _converter = new Campy.Compiler.CampyConverter();
            Buffer = new Buffers();
            //InitCuda();
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

        public static void Delay()
        {
            Singleton().Buffer.Delay = true;
        }

        public static void Synch()
        {
            Singleton().Buffer.Delay = false;
            Singleton().Buffer.SynchDataStructures();
        }

        public static void Managed(ManagedMemoryBlock block)
        {
            block();
            
            var stopwatch_deep_copy_back = new Stopwatch();
            stopwatch_deep_copy_back.Reset();
            stopwatch_deep_copy_back.Start();

            // Copy back all referenced.

            stopwatch_deep_copy_back.Stop();
            var elapse_deep_copy_back = stopwatch_deep_copy_back.Elapsed;
            System.Console.WriteLine("deep copy out " + elapse_deep_copy_back);
        }

        public static void For(int number_of_threads, KernelType kernel)
        {
            CampyConverter.InitCuda();

            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            //bool managed = false;
            //StackTrace st = new StackTrace(true);
            //for (int i = 0; i < st.FrameCount; i++)
            //{
            //    // Note that high up the call stack, there is only
            //    // one stack frame.
            //    StackFrame sf = st.GetFrame(i);
            //    MethodBase met = sf.GetMethod();
            //    string nae = met.Name;
            //    if (nae.Contains("Managed"))
            //    {
            //        managed = true;
            //        break;
            //    }
            //}

            try
            {
                unsafe
                {
                    IntPtr image = Singleton()._converter.JitCodeToImage(kernel.Method, kernel.Target);
                    CUfunction ptr_to_kernel = Singleton()._converter.GetCudaFunction(kernel.Method, image);
                    var stopwatch_cuda_compile = new Stopwatch();
                    stopwatch_cuda_compile.Start();
                    var elapse_cuda_compile = stopwatch_cuda_compile.Elapsed;
                    Index index = new Index();
                    Buffers buffer = Singleton().Buffer;
                    var stopwatch_deep_copy_to = new Stopwatch();
                    stopwatch_deep_copy_to.Reset();
                    stopwatch_deep_copy_to.Start();

                    // Set up parameters.
                    int count = kernel.Method.GetParameters().Length;
                    var bb = Singleton()._converter.GetBasicBlock(kernel.Method);
                    if (bb.HasThis) count++;
                    if (!(count == 1 || count == 2))
                        throw new Exception("Expecting at least one parameter for kernel.");

                    IntPtr[] parm1 = new IntPtr[1];
                    IntPtr[] parm2 = new IntPtr[1];
                    IntPtr ptr = IntPtr.Zero;

                    if (bb.HasThis)
                    {
                        ptr = buffer.AddDataStructure(kernel.Target);
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

                    System.Console.WriteLine("cuda compile  " + elapse_cuda_compile);
                    System.Console.WriteLine("deep copy in  " + elapse_deep_copy_to);
                    System.Console.WriteLine("cuda kernel   " + elapse_call_kernel);

                    {
                        var stopwatch_deep_copy_back = new Stopwatch();
                        stopwatch_deep_copy_back.Reset();
                        stopwatch_deep_copy_back.Start();

                        if (!buffer.Delay)
                            buffer.SynchDataStructures();
                        
                        stopwatch_deep_copy_back.Stop();
                        var elapse_deep_copy_back = stopwatch_deep_copy_back.Elapsed;
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
                handle1.Free();
                handle2.Free();
            }
        }

        private static void Finish(Buffers buffer, KernelType kernel, IntPtr ptr)
        {
            try
            {
                unsafe
                {
                    var stopwatch_deep_copy_back = new Stopwatch();
                    stopwatch_deep_copy_back.Reset();
                    stopwatch_deep_copy_back.Start();

                    buffer.DeepCopyFromImplementation(ptr, out object to, kernel.Target.GetType());

                    stopwatch_deep_copy_back.Stop();
                    var elapse_deep_copy_back = stopwatch_deep_copy_back.Elapsed;

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
            }

        }
    }
}