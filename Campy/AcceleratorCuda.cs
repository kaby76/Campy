namespace Campy
{
    using System.Collections.Generic;

    internal class AcceleratorCuda : Accelerator
    {
        internal static List<Accelerator> AllCudaDevices()
        {
            List<Accelerator> results = new List<Accelerator>();
            var res = Swigged.Cuda.Cuda.cuDeviceGetCount(out int count);
            if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) return results;
            if (count == 0) return results;
            for (int i = 0; i < count; ++i)
            {
                res = Swigged.Cuda.Cuda.cuCtxCreate_v2(out Swigged.Cuda.CUcontext ctx, 0, i);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGet(out int dev, i);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_WARP_SIZE, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CLOCK_RATE, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, Swigged.Cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceGetName(out string name, 1000, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                res = Swigged.Cuda.Cuda.cuDeviceTotalMem_v2(out ulong bytes, dev);
                if (res != Swigged.Cuda.CUresult.CUDA_SUCCESS) continue;
                var acc = new Accelerator();
                acc.Description = name + " | " + bytes.ToString();
                results.Add(acc);
            }
            return results;
        }
    }
}
