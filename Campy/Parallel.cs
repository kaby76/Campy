using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using Campy.Types;
using Campy.ControlFlowGraph;
using Campy.Types.Utils;
using DeepCopyGPU;
using Mono.Cecil;
using Swigged.Cuda;
using Type = System.Type;
using Swigged.LLVM;

namespace Campy
{
    public class Parallel
    {

        public static void For(Extent extent, _Kernel_type kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, extent, kernel);
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        static public unsafe void For(AcceleratorView view, Extent extent, _Kernel_type kernel)
        {
            Swigged.LLVM.Helper.Adjust.Path();
            Cuda.cuInit(0);

            Reader r = new Reader();
            CFG g = r.Cfg;
            CUresult res;
            
            Converter c = new Campy.ControlFlowGraph.Converter(g);

            // Parse kernel instructions to determine basic block representation of all the code to compile.
            int change_set_id = g.StartChangeSet();
            r.AnalyzeMethod(kernel);
            List<CFG.Vertex> cs = g.PopChangeSet(change_set_id);

            // Very important note: Although we have the control flow graph of the code that is to
            // be compiled, there is going to be generics used, e.g., ArrayView<int>, within the body
            // of the code and in the called runtime library. We need to record the types for compiling
            // and add that to compilation.
            // https://stackoverflow.com/questions/5342345/how-do-generics-get-compiled-by-the-jit-compiler

            // Create a list of generics called with types passed.
            List<Type> list_of_data_types_used = c.FindAllTargets(kernel);

            // Convert list into Mono data types.
            List<Mono.Cecil.TypeDefinition> list_of_mono_data_types_used = new List<TypeDefinition>();
            foreach (System.Type data_type_used in list_of_data_types_used)
            {
                var mono_type = Campy.Types.Utils.ReflectionCecilInterop.ConvertToMonoCecilTypeDefinition(data_type_used);
                //if (mono_type == null) continue;
                list_of_mono_data_types_used.Add(mono_type);
            }
            if (list_of_mono_data_types_used.Count != list_of_data_types_used.Count) throw new Exception("Cannot convert types properly to Mono.");

            // Instantiate all generics at this point.
            cs = c.InstantiateGenerics(cs, list_of_data_types_used, list_of_mono_data_types_used);

            // Compile methods with added type information.
            c.CompileToLLVM(cs, list_of_mono_data_types_used);

            // Get basic block of entry.
            var bb = cs.First();
            var method = bb.Method;

            var helloWorld = c.GetPtr(cs.First().Name);

            var ok = GC.TryStartNoGCRegion(200000000);
            var rank = extent._Rank;
            Index index = new Index(extent.Size());
            Buffers buffer = new Buffers();

            // Set up parameters.
            var parameters = method.Parameters;
            int count = parameters.Count;
            if (bb.HasThis) count++;

            IntPtr[] parms1 = new IntPtr[1];
            IntPtr[] parms2 = new IntPtr[1];
            int current = 0;
            IntPtr ptr = IntPtr.Zero;
            if (count > 0)
            {
                if (bb.HasThis)
                {
                    // kernel.Target is a class. Copy the entire class to managed memory.
                    Type type = kernel.Target.GetType();
                    Type btype = buffer.CreateImplementationType(type);
                    ptr = buffer.New(Marshal.SizeOf(btype));
                    buffer.DeepCopyToImplementation(kernel.Target, ptr);

                    parms1[0] = ptr;

                    current++;
                }

                //foreach (var p in parameters)
                {
                    Type btype = buffer.CreateImplementationType(typeof(Index));
                    var s = Marshal.SizeOf(btype);
                    var ptr2 = buffer.New(s);
                //    buffer.DeepCopyToImplementation(index, ptr2);
                    parms2[0] = ptr2;
                }
            }

            //int[] v = { 'G', 'd', 'k', 'k', 'n', (char)31, 'v', 'n', 'q', 'k', 'c' };
            //GCHandle handle = GCHandle.Alloc(v, GCHandleType.Pinned);
            //IntPtr pointer = IntPtr.Zero;
            //pointer = handle.AddrOfPinnedObject();

            //IntPtr dptr = buffer.New(11 * sizeof(int));
            // res = Cuda.cuMemcpyHtoD_v2(dptr, pointer, 11*sizeof(int));
            // if (res != CUresult.CUDA_SUCCESS) throw new Exception();

            //IntPtr[] x = new IntPtr[] { dptr };
            IntPtr[] x = parms1;
            GCHandle handle1 = GCHandle.Alloc(x, GCHandleType.Pinned);
            IntPtr pointer1 = IntPtr.Zero;
            pointer1 = handle1.AddrOfPinnedObject();

            IntPtr[] x2 = parms2;
            GCHandle handle2 = GCHandle.Alloc(x, GCHandleType.Pinned);
            IntPtr pointer2 = IntPtr.Zero;
            pointer2 = handle2.AddrOfPinnedObject();

            IntPtr[] kp = new IntPtr[] { pointer1, pointer2 };
            res = CUresult.CUDA_SUCCESS;
            fixed (IntPtr* kernelParams = kp)
            {
                res = Cuda.cuLaunchKernel(helloWorld,
                    1, 1, 1, // grid has one block.
                    2, 1, 1, // block has 2 threads.
                    0, // no shared memory
                    default(CUstream),
                    (IntPtr)kernelParams,
                    (IntPtr)IntPtr.Zero
                );
            }
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            res = Cuda.cuCtxSynchronize();
            if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            buffer.DeepCopyFromImplementation(ptr, out object to, kernel.Target.GetType());
            //res = Cuda.cuMemcpyDtoH_v2(pointer, dptr, 11 * sizeof(int));
            //if (res != CUresult.CUDA_SUCCESS) throw new Exception();
            //Cuda.cuCtxDestroy_v2(cuContext);
            //return default(IntPtr);
        }

        static public void For(TiledExtent extent, _Kernel_tiled_type kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, extent, kernel);
        }

        static public void For(AcceleratorView view, TiledExtent extent, _Kernel_tiled_type kernel)
        {
        }
    }
}