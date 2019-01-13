using 'Enums.pig';
using 'Structs.pig';
using 'Funcs.pig';
using 'Namespace.pig';
using 'Typedefs.pig';

template CudaNamespace : Namespace
{
    init {{
        namespace_name = "Campy.Utils";
        PiggyRuntime.TemplateHelpers.ModParamUsageType(
            new Dictionary<string, string>() {
            { "const char **", "out IntPtr" },
            { "char *", "[Out] byte[]"},
            { "unsigned int *", "out uint" },
            { "void **", "out IntPtr" },
            { "void *", "IntPtr" },
            { "const char *", "string" },
            { "const void *", "IntPtr" },
            { "const <type> *", "in <type>"},
        });
        PiggyRuntime.TemplateHelpers.ModNonParamUsageType(
            new Dictionary<string, string>() {
            { "char *", "byte[]"},
            { "size_t", "SizeT" },
            { "int", "int"},
            { "uint", "uint"},
            { "short", "short"},
            { "ushort", "ushort"},
            { "long", "long"},
            { "unsigned char", "byte" },
            { "unsigned short", "UInt16"},
            { "unsigned int", "uint"},
            { "unsigned long", "ulong"},
            { "unsigned long long", "ulong"},
            { "long long", "long"},
            { "float", "float"},
            { "double", "double"},
            { "bool", "bool"},
            { "char", "byte"},
            { "const char *", "string" },
    });
    }}
}

template CudaEnums : Enums
{
    init {{
        // Override limits in matching.
        limit = ".*\\.*GPU.*\\.*";
        var list = new List<string>() {
            "cudaError_enum",
            "CUdevice_attribute_enum",
            "CUjit_option_enum",
            "CUmemAttach_flags_enum",
            "CUjitInputType_enum",
			"CUlimit_enum",
			"^CUjitInputType_enum$",
            };
        generate_for_only = String.Join("|", list);
    }}
}

template CudaStructs : Structs
{
    init {{
        // Override limits in matching.
        limit = ".*\\.*GPU.*\\.*";
        var list = new List<string>() {
            "CUdevprop",
            };
        generate_for_only = String.Join("|", list);
    }}
}

template CudaTypedefs : Typedefs
{
    init {{
        // Override limits in matching.
        limit = ".*\\.*GPU.*\\.*";
        var list = new List<string>() {
            "^CUresult$",
            "^CUcontext$",
            "^CUfunction$",
            "^CUlinkState$",
            "^CUmodule$",
            "^CUstream$",
            "^CUdevice$",
            "^CUjit_option$",
            "^CUdeviceptr$",
            "^CUdevprop$",
			"^CUlimit$",
			"^CUjitInputType$",
			"^CUdevice_attribute$",
            };
        generate_for_only = String.Join("|", list);
    }}
}


template CudaFuncs : Funcs
{
    init {{
        limit = ".*\\.*GPU.*\\.*";
        var list = new List<string>() {
            "^cuCtxCreate_v2$",
            "^cuCtxDestroy_v2",
            "^cuCtxSynchronize$",
            "^cuDeviceGet$",
            "^cuDeviceGetCount$",
            "^cuDeviceGetName$",
            "^cuDeviceGetPCIBusId$",
            "^cuDeviceGetProperties$",
            "^cuDevicePrimaryCtxReset$",
            "^cuDeviceTotalMem_v2$",
            "^cuGetErrorString$",
            "^cuInit$",
            "^cuLaunchKernel$",
            "^cuLinkComplete$",
            "^cuMemAlloc_v2$",
            "^cuMemcpyDtoH_v2$",
            "^cuMemcpyHtoD_v2$",
            "^cuMemFreeHost$",
            "^cuMemGetInfo_v2$",
            "^cuModuleGetFunction$",
            "^cuModuleGetGlobal_v2$",
            "^cuModuleLoadData$",
			"^cuMemHostAlloc$",
			"^cuCtxGetLimit$",
			"^cuCtxSetLimit$",
			"^cuLinkAddData_v2$",
			"^cuLinkAddFile_v2$",
			"^cuLinkCreate_v2$",
			"^cuDeviceGetAttribute$",
            };
        generate_for_only = String.Join("|", list);
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
        dllname = "nvcuda";
    }}

    pass Functions {
        ( FunctionDecl SrcRange=$"{CudaFuncs.limit}" Name="cuModuleLoadDataEx"
            [[ [DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "cuModuleLoadDataEx")]
            public static extern CUresult cuModuleLoadDataEx(out CUmodule jarg1, IntPtr jarg2, uint jarg3, CUjit_option[] jarg4, IntPtr jarg5);
            
            ]]
        )
        ( FunctionDecl SrcRange=$"{CudaFuncs.limit}" Name="cuLaunchKernel"
            [[ [DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "cuLaunchKernel")]
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
    CudaNamespace.GenerateStart
    CudaEnums.GenerateEnums
    CudaTypedefs.GeneratePointerTypes
    CudaStructs.GenerateStructs
    CudaTypedefs.GenerateTypedefs
    CudaFuncs.Start
    CudaFuncs.Functions
    CudaFuncs.End
    Namespace.GenerateEnd
    ;
