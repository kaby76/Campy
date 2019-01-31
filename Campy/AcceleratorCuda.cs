using System.Text;
using Campy.Utils;
using Mono.Cecil;

namespace Campy
{
    using System.Collections.Generic;

    internal class AcceleratorCuda : Accelerator
    {
        internal static unsafe List<Accelerator> AllCudaDevices()
        {
            List<Accelerator> results = new List<Accelerator>();
            int count = default(int);
            var res = Functions.cuDeviceGetCount(ref count);
            if (res.Value != cudaError_enum.CUDA_SUCCESS) return results;
            if (count == 0) return results;
            for (int i = 0; i < count; ++i)
            {
                CUcontext ctx = default(CUcontext);
                res = Functions.cuCtxCreate_v2(ref ctx, 0, new CUdevice(i));
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                CUdevice dev = default(CUdevice);
                res = Functions.cuDeviceGet(ref dev, i);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_WARP_SIZE = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_WARP_SIZE, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_WARP_SIZE), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_CLOCK_RATE = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_CLOCK_RATE, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CLOCK_RATE), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                int CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = default(int);
                res = Functions.cuDeviceGetAttribute(ref CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, new CUdevice_attribute(CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK), dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                byte[] sb_name = new byte[1000];
                fixed (void* namex = sb_name)
                {
                    res = Functions.cuDeviceGetName((System.IntPtr)namex, 1000, dev);
                }
                System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
                var name = enc.GetString(sb_name).Replace("\0", "");
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                SizeT bytes = default(SizeT);
                res = Functions.cuDeviceTotalMem_v2(ref bytes, dev);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) continue;
                var acc = new Accelerator();
                acc.Description = name + " | " + bytes.ToString();
                results.Add(acc);
            }
            return results;
        }
    }
}
