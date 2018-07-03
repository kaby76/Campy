using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using Campy;
using System.Linq;

namespace ConsoleApp4
{
    public class JaggedArray
    {
        public static void JaggedArrayT()
        {
            int[][] jagged_array = new int[][]
            {
                new int[] {1, 3, 5, 7, 9},
                new int[] {0, 2, 4, 6},
                new int[] {11, 22}
            };
            Campy.Parallel.For(3, i =>
            {
                jagged_array[i][0] = i; //jagged_array[i].Length;
            });
            for (int i = 0; i < 3; ++i)
                if (jagged_array[i][0] != i) // jagged_array[i].Length)
                    throw new Exception();
            Campy.Parallel.For(3, i =>
            {
                jagged_array[i][0] = jagged_array[i].Length;
            });
            for (int i = 0; i < 3; ++i)
                if (jagged_array[i][0] != jagged_array[i].Length)
                    throw new Exception();
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
            JaggedArray.JaggedArrayT();
        }
    }
}
