using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using Campy;
using System.Linq;

namespace ConsoleApp4
{
    class Program
    {
        static void StartDebugging()
        {
            //Campy.Utils.Options.Set("debug_info_off");
            //Campy.Utils.Options.Set("graph_trace");
            //Campy.Utils.Options.Set("name_trace");
            //     Campy.Utils.Options.Set("cfg_construction_trace");
 //                 Campy.Utils.Options.Set("cfg_construction_trace");
            //Campy.Utils.Options.Set("dot_graph");
            //Campy.Utils.Options.Set("jit_trace");
            //Campy.Utils.Options.Set("memory_trace");
            //Campy.Utils.Options.Set("ptx_trace");
            //Campy.Utils.Options.Set("state_computation_trace");
            Campy.Utils.Options.Set("overview_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_llvm_computation_trace");
            // Campy.Utils.Options.Set("continue_with_no_resolve");
            //Campy.Utils.Options.Set("copy_trace");
            //    Campy.Utils.Options.Set("runtime_trace");
            Campy.Utils.Options.Set("graph-output");
            Campy.Utils.Options.Set("ptx-output");
            Campy.Utils.Options.Set("llvm-output");
            Campy.Utils.Options.Set("dot-output");
            //Campy.Utils.Options.Set("import-only");
            Campy.Utils.Options.Set("trace-cctors");
        }

        static void Main(string[] args)
        {
            double[] res = new double[10];
            StartDebugging();
            Campy.Parallel.For(10, i =>
            {
                switch (i)
                {
                    case 0:
                        res[i] = Math.Sin(0.5);
                        break;
                    case 1:
                        res[i] = Math.Cos(0.5);
                        break;
                    case 2:
                        res[i] = Math.Tan(0.5);
                        break;
                    case 3:
                        res[i] = Math.Log(0.5);
                        break;
                    case 4:
                        res[i] = Math.Log10(0.5);
                        break;
                    case 5:
                        res[i] = Math.Exp(0.5);
                        break;
                    case 6:
                        res[i] = Math.Ceiling(0.5);
                        break;
                    case 7:
                        res[i] = Math.Cosh(0.5);
                        break;
                    case 8:
                        res[i] = Math.Sinh(0.5);
                        break;
                    case 9:
                        res[i] = Math.Pow(0.5, 2);
                        break;
                }
            });
            double[] gold = new double[10];
            for (int i = 0; i < 10; ++i)
                switch (i)
                {
                    case 0:
                        gold[i] = Math.Sin(0.5);
                        break;
                    case 1:
                        gold[i] = Math.Cos(0.5);
                        break;
                    case 2:
                        gold[i] = Math.Tan(0.5);
                        break;
                    case 3:
                        gold[i] = Math.Log(0.5);
                        break;
                    case 4:
                        gold[i] = Math.Log10(0.5);
                        break;
                    case 5:
                        gold[i] = Math.Exp(0.5);
                        break;
                    case 6:
                        gold[i] = Math.Ceiling(0.5);
                        break;
                    case 7:
                        gold[i] = Math.Cosh(0.5);
                        break;
                    case 8:
                        gold[i] = Math.Sinh(0.5);
                        break;
                    case 9:
                        gold[i] = Math.Pow(0.5, 2);
                        break;
                }
            for (int i = 0; i < 10; ++i)
                if (Math.Abs(res[i] - gold[i]) > 0.000001)
                    throw new Exception();
        }
    }
}
