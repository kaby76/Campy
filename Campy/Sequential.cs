namespace Campy
{
    using System;
    using System.Runtime.InteropServices;
    using Campy.Utils;
    using Campy.Meta;

    public class Sequential
    {
        public static void For(int number_of_threads, SimpleKernel simpleKernel)
        {
            // Semantics of this method: run the kernel number_of_threads times on
            // the given accelerator. It is not simply running it on a
            // CPU. The problem is that memory could be on accelerator,
            // not CPU, which would force the data to be transfered
            // back to C# in the CPU.
            //
            // Note, we need to pass a unique thread id even though we are iterating
            // for a given number of threads.
            for (int i = 0; i < number_of_threads; ++i)
            {
                SetBaseIndex(i);
                Campy.Parallel.For(1, simpleKernel);
            }
        }

        private static void SetBaseIndex(int base_index)
        {
            unsafe
            {
                IntPtr[] x1 = new IntPtr[] { new IntPtr(base_index) };
                GCHandle handle2 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                var parm1 = handle2.AddrOfPinnedObject();

                IntPtr[] kp = new IntPtr[] { parm1 };

                Swigged.Cuda.CUmodule module = RUNTIME.RuntimeModule;
                CudaHelpers.CheckCudaError(Swigged.Cuda.Cuda.cuModuleGetFunction(out Swigged.Cuda.CUfunction function, module, "_Z21set_kernel_base_indexi"));
                Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                    out Campy.Utils.CudaHelpers.dim3 tile_size,
                    out Campy.Utils.CudaHelpers.dim3 tiles);
                Swigged.Cuda.CUresult res;
                fixed (IntPtr* kernelParams = kp)
                {
                    res = Swigged.Cuda.Cuda.cuLaunchKernel(
                        function,
                        tiles.x, tiles.y, tiles.z, // grid has one block.
                        tile_size.x, tile_size.y, tile_size.z, // n threads.
                        0, // no shared memory
                        default(Swigged.Cuda.CUstream),
                        (IntPtr)kernelParams,
                        (IntPtr)IntPtr.Zero
                    );
                }
                Utils.CudaHelpers.CheckCudaError(res);
                res = Swigged.Cuda.Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                Utils.CudaHelpers.CheckCudaError(res);
            }
        }
    }
}
