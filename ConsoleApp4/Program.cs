﻿using System;
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
            StartDebugging();
            Campy.Parallel.For(3, i =>
            {
                System.Console.WriteLine(i.ToString());
            });
        }
    }
}
