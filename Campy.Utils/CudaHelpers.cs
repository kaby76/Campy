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

        public static void CheckCudaError(Swigged.Cuda.CUresult res)
        {
            if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS)
            {
                Swigged.Cuda.Cuda.cuGetErrorString(res, out IntPtr pStr);
                var cuda_error = Marshal.PtrToStringAnsi(pStr);
                throw new Exception("CUDA error: " + cuda_error);
            }
        }
    }
}
