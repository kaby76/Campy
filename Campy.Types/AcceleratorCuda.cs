using System;
using System.Collections.Generic;
using System.Text;
using Swigged.Cuda;

namespace Campy.Types
{
    internal class AcceleratorCuda : Accelerator
    {
        internal static List<Accelerator> AllCudaDevices()
        {
            List<Accelerator> results = new List<Accelerator>();
            Cuda.cuInit(0);
            var res = Cuda.cuDeviceGetCount(out int count);
            if (res != CUresult.CUDA_SUCCESS) return results;
            if (count == 0) return results;
            for (int i = 0; i < count; ++i)
            {
                res = Cuda.cuCtxCreate_v2(out CUcontext ctx, 0, i);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGet(out int dev, i);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_WARP_SIZE, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CLOCK_RATE, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;
                res = Cuda.cuDeviceGetAttribute(out int CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, dev);
                if (res != CUresult.CUDA_SUCCESS) continue;

                //res = Cuda.cuDeviceGetName(naem, )
                var acc = new Accelerator();
                acc.Description = CU_DEVICE_ATTRIBUTE_WARP_SIZE + "|" + CU_DEVICE_ATTRIBUTE_COMPUTE_MODE;
                results.Add(acc);
            }
            return results;
        }
    }
}
