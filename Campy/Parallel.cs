using System;
using System.Diagnostics;
using System.IO;
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
        private JITER _converter;
        private BUFFERS Buffer { get; }

        private Parallel()
        {
            _converter = JITER.Singleton;
            Buffer = new BUFFERS();
        }

        private static Parallel Singleton
        {
            get
            {
                if (_singleton == null)
                {
                    _singleton = new Parallel();
                }
                return _singleton;
            }
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
            Singleton.Buffer.Delay(obj);
        }

        /// <summary>
        /// Make "obj" stay on the GPU, and do not copy it back to the CPU.
        /// The closure object for the kernel delegate is always copied to and from the GPU
        /// after each call.
        /// </summary>
        /// <param name="obj"></param>
        public static void ReadOnly(object obj)
        {
            Singleton.Buffer.ReadOnly(obj);
        }

        /// <summary>
        /// Copy all "sticky" objects used in the kernel back to the CPU from the GPU.
        /// The closure object for the kernel delegate is always copied to and from the GPU
        /// after each call.
        /// </summary>
        public static void Sync()
        {
            Singleton.Buffer.FullSynch();
        }

        public static void For(int number_of_threads, SimpleKernel simpleKernel)
        {
            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            try
            {
                unsafe
                {
                    System.Reflection.MethodInfo method_info = simpleKernel.Method;
                    String kernel_assembly_file_name = method_info.DeclaringType.Assembly.Location;
                    string p = Path.GetDirectoryName(kernel_assembly_file_name);
                    var resolver = new DefaultAssemblyResolver();
                    resolver.AddSearchDirectory(p);
                    Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(
                        kernel_assembly_file_name,
                        new ReaderParameters { AssemblyResolver = resolver, ReadSymbols = true });
                    MethodReference method_reference = md.ImportReference(method_info);

                    CUfunction ptr_to_kernel = default(CUfunction);

                    //////// COMPILE KERNEL INTO GPU CODE ///////
                    /////////////////////////////////////////////
                    Campy.Utils.TimePhase.Time("compile     ", () =>
                    {
                        IntPtr image = Singleton._converter.Compile(method_reference, simpleKernel.Target);
                        ptr_to_kernel = Singleton._converter.GetCudaFunction(method_reference, image);
                    });

                    RUNTIME.CheckHeap();

                    IntPtr[] parm1 = new IntPtr[1];
                    IntPtr[] parm2 = new IntPtr[1];
                    BUFFERS buffer = Singleton.Buffer;

                    //////// COPY DATA INTO GPU /////////////////
                    /////////////////////////////////////////////
                    Campy.Utils.TimePhase.Time("deep copy     ", () =>
                    {
                        // Set up parameters.
                        int count = simpleKernel.Method.GetParameters().Length;
                        var bb = Singleton._converter.GetBasicBlock(method_reference);
                        if (bb.HasThis) count++;
                        if (!(count == 1 || count == 2))
                            throw new Exception("Expecting at least one parameter for kernel.");

                        IntPtr ptr = IntPtr.Zero;

                        // The method really should have a "this" because it's a closure
                        // object.
                        if (bb.HasThis)
                        {
                            RUNTIME.CheckHeap();
                            ptr = buffer.AddDataStructure(simpleKernel.Target);
                            parm1[0] = ptr;
                            RUNTIME.CheckHeap();
                        }

                        {
                            RUNTIME.CheckHeap();
                            Type btype = typeof(int);
                            var s = BUFFERS.SizeOf(btype);
                            var ptr2 = buffer.New(s);
                            // buffer.DeepCopyToImplementation(index, ptr2);
                            parm2[0] = ptr2;
                            RUNTIME.CheckHeap();
                        }
                    });

                    Campy.Utils.TimePhase.Time("kernel call", () =>
                    {
                        IntPtr[] x1 = parm1;
                        handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                        IntPtr pointer1 = handle1.AddrOfPinnedObject();

                        IntPtr[] x2 = parm2;
                        handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                        IntPtr pointer2 = handle2.AddrOfPinnedObject();

                        RUNTIME.CheckHeap();

                        IntPtr[] kp = new IntPtr[] {pointer1, pointer2};
                        var res = CUresult.CUDA_SUCCESS;
                        fixed (IntPtr* kernelParams = kp)
                        {
                            Campy.Utils.CudaHelpers.MakeLinearTiling(number_of_threads,
                                out Campy.Utils.CudaHelpers.dim3 tile_size, out Campy.Utils.CudaHelpers.dim3 tiles);

                            //MakeLinearTiling(1, out dim3 tile_size, out dim3 tiles);

                            res = Cuda.cuLaunchKernel(
                                ptr_to_kernel,
                                tiles.x, tiles.y, tiles.z, // grid has one block.
                                tile_size.x, tile_size.y, tile_size.z, // n threads.
                                0, // no shared memory
                                default(CUstream),
                                (IntPtr) kernelParams,
                                (IntPtr) IntPtr.Zero
                            );
                        }

                        CudaHelpers.CheckCudaError(res);
                        res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                        CudaHelpers.CheckCudaError(res);
                    });

                    //if (Campy.Utils.Options.IsOn("jit_trace"))
                    //{
                    //    System.Console.WriteLine("cuda compile  " + elapse_cuda_compile);
                    //    System.Console.WriteLine("deep copy in  " + elapse_deep_copy_to);
                    //    System.Console.WriteLine("cuda kernel   " + elapse_call_kernel);
                    //    System.Console.WriteLine("deep copy out " + elapse_deep_copy_back);
                    //}

                    Campy.Utils.TimePhase.Time("kernel call", () =>
                    {
                        RUNTIME.CheckHeap();
                        buffer.SynchDataStructures();
                        RUNTIME.CheckHeap();
                    });
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