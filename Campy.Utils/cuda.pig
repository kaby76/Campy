using 'ClangSupport.pig';
using 'Enums.pig';
using 'Structs.pig';
using 'Funcs.pig';
using 'Namespace.pig';
using 'Typedefs.pig';

template CudaClangSupport : ClangSupport
{
    init {{
        namespace_name = "Campy.Utils";
        limit = ".*\\.*GPU.*\\.*";
        var list = new List<string>() {
            "^CUcontext$",
            "^cuCtxCreate_v2$",
            "^cuCtxDestroy_v2$",
            "^cuCtxGetLimit$",
            "^cuCtxSynchronize$",
            "^cudaError_enum$",
            "^CUdevice$",
            "^CUdevice_attribute$",
            "^CUdevice_attribute_enum$",
            "^cuDeviceGet$",
            "^cuDeviceGetAttribute$",
            "^cuDeviceGetCount$",
            "^cuDeviceGetName$",
            "^cuDeviceGetPCIBusId$",
            "^cuDeviceGetProperties$",
            "^cuDevicePrimaryCtxReset$",
            "^CUdeviceptr$",
            "^cuDeviceTotalMem_v2$",
            "^CUdevprop$",
            "^CUdevprop_st$",
            "^CUfunction$",
            "^cuGetErrorString$",
            "^cuInit$",
            "^CUjit_option$",
            "^CUjit_option_enum$",
            "^CUjitInputType$",
            "^CUjitInputType_enum$",
            "^cuLaunchKernel$",
            "^CUlimit$",
            "^CUlimit_enum$",
            "^cuLinkAddData_v2$",
            "^cuLinkAddFile_v2$",
            "^cuLinkComplete$",
            "^cuLinkCreate_v2$",
            "^CUlinkState$",
            "^cuMemAlloc_v2$",
            "^CUmemAttach_flags_enum$",
            "^cuMemcpyDtoH_v2$",
            "^cuMemcpyHtoD_v2$",
            "^cuMemFreeHost$",
            "^cuMemGetInfo_v2$",
            "^cuMemHostAlloc$",
            "^CUmodule$",
            "^cuModuleGetFunction$",
            "^cuModuleGetGlobal_v2$",
            "^cuModuleLoadData$",
            "^CUresult$",
            "^CUstream$",
			"^cuCtxSetLimit$",
           };
        generate_for_only = String.Join("|", list);
        dllname = "nvcuda";
    }}
}

template CudaFuncs : Funcs
{
    init {{
        details = new List<generate_type>()
            {
                { new generate_type()
                    {
                        name = ".*",
                        convention = System.Runtime.InteropServices.CallingConvention.Cdecl,
                        special_args = null
                    }
                }
            }; // default for everything.
    }}

    pass Functions {
        ( FunctionDecl SrcRange=$"{ClangSupport.limit}" Name="cuModuleLoadDataEx"
            [[
				[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "cuModuleLoadDataEx")]
				public static extern CUresult cuModuleLoadDataEx(out CUmodule jarg1, IntPtr jarg2, uint jarg3, CUjit_option[] jarg4, IntPtr jarg5);
            ]]
        )
        ( FunctionDecl SrcRange=$"{ClangSupport.limit}" Name="cuLaunchKernel"
            [[
				[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "cuLaunchKernel")]
				public static extern CUresult cuLaunchKernel(CUfunction f, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, CUstream hStream, IntPtr kernelParams, IntPtr extra);
            ]]
        )
                ( FunctionDecl Name="cuLinkCreate_v2"
				 [[
					 [global::System.Runtime.InteropServices.DllImport("nvcuda", EntryPoint="cuLinkCreate_v2")]
					 public static extern CUresult cuLinkCreate_v2(uint jarg1, CUjit_option[] jarg2, System.IntPtr jarg3, out CUlinkState jarg4);
				 ]]
                )
                ( FunctionDecl Name="cuLinkAddData_v2"
                [[
					[global::System.Runtime.InteropServices.DllImport("nvcuda", EntryPoint="cuLinkAddData_v2")]
					public static extern CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, System.IntPtr jarg3, uint jarg4, string jarg5, uint jarg6, CUjit_option[] jarg7, System.IntPtr jarg8);
                ]]
                )
                ( FunctionDecl Name="cuLinkAddFile_v2"
                [[
					[global::System.Runtime.InteropServices.DllImport("nvcuda", EntryPoint="cuLinkAddFile_v2")]
					public static extern CUresult cuLinkAddFile_v2(CUlinkState jarg1, CUjitInputType jarg2, string jarg3, uint jarg4, CUjit_option[] jarg5, System.IntPtr jarg6);
                ]]
                )

    }
}

application
    CudaClangSupport.Start
    Namespace.GenerateStart
    Enums.GenerateEnums
    Typedefs.GeneratePointerTypes
    Structs.GenerateStructs
    Typedefs.GenerateTypedefs
    CudaFuncs.Start
    CudaFuncs.Functions
    CudaFuncs.End
    Namespace.GenerateEnd
    ;
