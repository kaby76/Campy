namespace Campy.Utils
{
    using System;
    using System.Runtime.InteropServices;

    public class CudaHelpers
    {
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
            int[] max_threads = new int[] { 1024/4, 1024/4, 64/4 };
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

        public static void CheckCudaError(CUresult res)
        {
            if (res.Value != cudaError_enum.CUDA_SUCCESS)
            {
                IntPtr pStr = IntPtr.Zero;
                Functions.cuGetErrorString(res, ref pStr);
                var cuda_error = Marshal.PtrToStringAnsi(pStr);
                throw new Exception("CUDA error: " + cuda_error);
            }
        }

        public enum CU_MEMHOSTALLOC
        {
            /**
             * If set, host memory is portable between CUDA contexts.
             * Flag for ::cuMemHostAlloc()
             */
            CU_MEMHOSTALLOC_PORTABLE = 0x01,

            /**
             * If set, host memory is mapped into CUDA address space and
             * ::cuMemHostGetDevicePointer() may be called on the host pointer.
             * Flag for ::cuMemHostAlloc()
             */
            CU_MEMHOSTALLOC_DEVICEMAP = 0x02,

            /**
             * If set, host memory is allocated as write-combined - fast to write,
             * faster to DMA, slow to read except via SSE4 streaming load instruction
             * (MOVNTDQA).
             * Flag for ::cuMemHostAlloc()
             */
            CU_MEMHOSTALLOC_WRITECOMBINED = 0x04,
        }

        public enum CU_MEMHOSTREGISTER
        {
            /**
             * If set, host memory is portable between CUDA contexts.
             * Flag for ::cuMemHostRegister()
             */
            CU_MEMHOSTREGISTER_PORTABLE = 0x01,

            /**
             * If set, host memory is mapped into CUDA address space and
             * ::cuMemHostGetDevicePointer() may be called on the host pointer.
             * Flag for ::cuMemHostRegister()
             */
            CU_MEMHOSTREGISTER_DEVICEMAP = 0x02,

            /**
             * If set, the passed memory pointer is treated as pointing to some
             * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
             * On Windows the flag is a no-op.
             * On Linux that memory is marked as non cache-coherent for the GPU and
             * is expected to be physically contiguous. It may return
             * CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
             * CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
             * On all other platforms, it is not supported and CUDA_ERROR_NOT_SUPPORTED
             * is returned.
             * Flag for ::cuMemHostRegister()
             */
            CU_MEMHOSTREGISTER_IOMEMORY = 0x04
        }
    }
}
