using System;
using System.Collections.Generic;
using Campy;

namespace ConsoleApp1
{
    class A
    {
        public int X { get; set; }

        public int Score(A b)
        {
            return X + b.X;
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
for (int jjj = 0; jjj < 10; ++jjj) System.Threading.Thread.Sleep(1000);


            StartDebugging();
            {
                A[] array = new A[10];
                for (int i = 0; i < 10; ++i) array[i] = new A();

                Campy.Parallel.For(10, i =>
                {
                    array[i].X = i;
                });

                for (int i = 0; i < 10; i++)
                {
                    if (array[i].X != i) throw new Exception();
                }
            }

        }
    }
}
