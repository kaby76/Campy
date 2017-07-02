using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using Campy.Types;
using Campy.ControlFlowGraph;

namespace Campy
{
    public class Parallel
    {

        public static void For(Extent extent, _Kernel_type kernel)
        {
            AcceleratorView view = Accelerator.GetAutoSelectionView();
            For(view, extent, kernel);
        }

        static public void For(AcceleratorView view, Extent extent, _Kernel_type kernel)
        {
            Swigged.LLVM.Helper.Adjust.Path();

            Reader r = new Reader();
            CFG g = r.Cfg;
            Converter c = new Campy.ControlFlowGraph.Converter(g);
            int change_set_id = g.StartChangeSet();
            r.AnalyzeMethod(kernel);

            List<CFG.Vertex> cs = g.PopChangeSet(change_set_id);
            c.ConvertToLLVM(cs);
            IntPtr p = c.GetPtr(cs.First().Name);

            //DFoo2 f = (DFoo2)Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
            //f(k);
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