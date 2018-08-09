using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;

namespace ConsoleApp1
{
    public class TwoDimArrayInts
    {
        public static void TwoDimArrayIntsT()
        {
            int e = 10;
            int ex0 = 3;
            int ex1 = 5;
            int[,] b = new int[ex0, ex1];
            for (int i = 0; i < ex0; ++i)
                for (int j = 0; j < ex1; ++j)
                    b[i, j] = (i + 1) * (j + 1);
            Campy.Parallel.For(5, d =>
            {
                b[d % 3, d] = 33 + d;
            });
            if (b[0, 0] != 33) throw new Exception();
            if (b[1, 1] != 34) throw new Exception();
            if (b[2, 2] != 35) throw new Exception();
            if (b[0, 3] != 36) throw new Exception();
            if (b[1, 4] != 37) throw new Exception();
        }
    }
    class Program
    {
        static void StartDebugging()
        {
            Campy.Utils.Options.Set("graph_trace");
            Campy.Utils.Options.Set("module_trace");
            Campy.Utils.Options.Set("name_trace");
            Campy.Utils.Options.Set("cfg_construction_trace");
            Campy.Utils.Options.Set("dot_graph");
            Campy.Utils.Options.Set("jit_trace");
            Campy.Utils.Options.Set("memory_trace");
            Campy.Utils.Options.Set("ptx_trace");
            Campy.Utils.Options.Set("state_computation_trace");
            Campy.Utils.Options.Set("continue_with_no_resolve");
            Campy.Utils.Options.Set("copy_trace");
            Campy.Utils.Options.Set("runtime_trace");
        }

        static void Main(string[] args)
        {
            StartDebugging();
            TwoDimArrayInts.TwoDimArrayIntsT();
        }
    }
}
