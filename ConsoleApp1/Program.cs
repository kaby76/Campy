using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;

namespace ConsoleApp1
{
    public class ListsGaloreBasic
    {
        public static void ListsGaloreBasicT()
        {
            int n = 4;

            var t1 = new List<int>();
            for (int i = 0; i < n; ++i) t1.Add(0);
            Campy.Parallel.For(n, i => t1[i] = i);
            for (int i = 0; i < n; ++i) if (t1[i] != i) throw new Exception();

            var t2 = new List<float>();
            for (int i = 0; i < n; ++i) t2.Add(0);
            Campy.Parallel.For(n, i => t2[i] = 0.1f * i);
            for (int i = 0; i < n; ++i) if (t2[i] != 0.1f * i) throw new Exception();

            var t3 = new List<double>();
            for (int i = 0; i < n; ++i) t3.Add(0);
            Campy.Parallel.For(n, i => t3[i] = 0.1d * i);
            for (int i = 0; i < n; ++i) if (t3[i] != 0.1d * i) throw new Exception();

            var t4 = new List<ushort>();
            for (int i = 0; i < n; ++i) t4.Add(0);
            Campy.Parallel.For(n, i => t4[i] = (ushort)(i + 1));
            for (int i = 0; i < n; ++i) if (t4[i] != (ushort)(i + 1)) throw new Exception();

            var t5 = new List<int>();
            for (int i = 0; i < n; ++i) t5.Add(0);
            Campy.Parallel.For(n, i => t5[i] = t1[i] * 2);
            for (int i = 0; i < n; ++i) if (t5[i] != t1[i] * 2) throw new Exception();

            var t6 = new List<float>();
            for (int i = 0; i < n; ++i) t6.Add(0);
            Campy.Parallel.For(n, i => t6[i] = 0.1f * i + t2[i]);
            for (int i = 0; i < n; ++i) if (t6[i] != 0.1f * i + t2[i]) throw new Exception();

            var t7 = new List<double>();
            for (int i = 0; i < n; ++i) t7.Add(0);
            Campy.Parallel.For(n, i => t7[i] = 0.1f * i - t3[i]);
            for (int i = 0; i < n; ++i) if (t7[i] != 0.1f * i - t3[i]) throw new Exception();

            var t8 = new List<ushort>();
            for (int i = 0; i < n; ++i) t8.Add(0);
            Campy.Parallel.For(n, (i) =>
            {
                t8[i] = (ushort)(t4[i] + i + 1);
            });
            for (int i = 0; i < n; ++i) if (t8[i] != (ushort)(t4[i] + i + 1)) throw new Exception();
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
            ListsGaloreBasic.ListsGaloreBasicT();

        }
    }
}
