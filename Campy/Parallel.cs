using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using Campy.Compiler;
using Campy.Utils;
using Campy.Meta;
using Mono.Cecil;
using Swigged.Cuda;
using Type = System.Type;

namespace Campy
{
    public class Parallel
    {
        private static Parallel _singleton;
        private COMPILER _compiler;
        private BUFFERS Buffer { get; }

        private Parallel()
        {
            _compiler = COMPILER.Singleton;
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

        private static void JustImport(SimpleKernel simpleKernel)
        {
            System.Reflection.MethodInfo method_info = simpleKernel.Method;
            String kernel_assembly_file_name = method_info.DeclaringType.Assembly.Location;
            Mono.Cecil.ModuleDefinition md = Campy.Meta.StickyReadMod.StickyReadModule(
                kernel_assembly_file_name, new ReaderParameters { ReadSymbols = true });
            MethodReference method_reference = md.ImportReference(method_info);
            Campy.Utils.TimePhase.Time("compile     ", () =>
            {
                Singleton._compiler.ImportOnlyCompile(method_reference, simpleKernel.Target);
            });
        }

        public static void For(int number_of_threads, SimpleKernel simpleKernel)
        {
            if (Campy.Utils.Options.IsOn("import-only"))
            {
                JustImport(simpleKernel);
                return;
            }

            GCHandle handle1 = default(GCHandle);
            GCHandle handle2 = default(GCHandle);

            try
            {
                unsafe
                {
                    System.Reflection.MethodInfo method_info = simpleKernel.Method;
                    String kernel_assembly_file_name = method_info.DeclaringType.Assembly.Location;
                    Mono.Cecil.ModuleDefinition md = Campy.Meta.StickyReadMod.StickyReadModule(
                        kernel_assembly_file_name, new ReaderParameters { ReadSymbols = true });
                    MethodReference method_reference = md.ImportReference(method_info);

                    CUfunction ptr_to_kernel = default(CUfunction);
                    CUmodule module = default(CUmodule);

                    Campy.Utils.TimePhase.Time("compile     ", () =>
                    {
                        IntPtr image = Singleton._compiler.Compile(method_reference, simpleKernel.Target);
                        module = Singleton._compiler.SetModule(method_reference, image);
                        Singleton._compiler.StoreJits(module);
                        ptr_to_kernel = Singleton._compiler.GetCudaFunction(method_reference, module);
                    });

                    RUNTIME.BclCheckHeap();

                    BUFFERS buffer = Singleton.Buffer;
                    IntPtr kernel_target_object = IntPtr.Zero;

                    Campy.Utils.TimePhase.Time("deep copy ", () =>
                    {
                        int count = simpleKernel.Method.GetParameters().Length;
                        var bb = Singleton._compiler.GetBasicBlock(method_reference);
                        if (bb.HasThis) count++;
                        if (!(count == 1 || count == 2))
                            throw new Exception("Expecting at least one parameter for kernel.");

                        if (bb.HasThis)
                        {
                            kernel_target_object = buffer.AddDataStructure(simpleKernel.Target);
                        }
                    });

                    Campy.Utils.TimePhase.Time("kernel cctor set up", () =>
                    {
                        // For each cctor, run on GPU.
                        foreach (var bb in Singleton._compiler.AllCctors())
                        {
                            var cctor = Singleton._compiler.GetCudaFunction(bb, module);

                            var res = CUresult.CUDA_SUCCESS;
                            Campy.Utils.CudaHelpers.MakeLinearTiling(1,
                                out Campy.Utils.CudaHelpers.dim3 tile_size, out Campy.Utils.CudaHelpers.dim3 tiles);

                            res = Cuda.cuLaunchKernel(
                                cctor,
                                tiles.x, tiles.y, tiles.z, // grid has one block.
                                tile_size.x, tile_size.y, tile_size.z, // n threads.
                                0, // no shared memory
                                default(CUstream),
                                (IntPtr) IntPtr.Zero,
                                (IntPtr) IntPtr.Zero
                            );

                            CudaHelpers.CheckCudaError(res);
                            res = Cuda.cuCtxSynchronize(); // Make sure it's copied back to host.
                            CudaHelpers.CheckCudaError(res);
                        }
                    });

                    Campy.Utils.TimePhase.Time("kernel call ", () =>
                    {
                        IntPtr[] parm1 = new IntPtr[1];
                        IntPtr[] parm2 = new IntPtr[1];

                        parm1[0] = kernel_target_object;
                        parm2[0] = buffer.New(BUFFERS.SizeOf(typeof(int)));

                        IntPtr[] x1 = parm1;
                        handle1 = GCHandle.Alloc(x1, GCHandleType.Pinned);
                        IntPtr pointer1 = handle1.AddrOfPinnedObject();

                        IntPtr[] x2 = parm2;
                        handle2 = GCHandle.Alloc(x2, GCHandleType.Pinned);
                        IntPtr pointer2 = handle2.AddrOfPinnedObject();

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

                    Campy.Utils.TimePhase.Time("deep copy return ", () =>
                    {
                        buffer.SynchDataStructures();
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
            COMPILER.Options(options);
        }

        public static void Compile(Type type)
        {
            Singleton._compiler.Add(type);
        }
    }
}