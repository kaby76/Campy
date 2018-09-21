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
            Campy.Utils.Options.Set("jit_trace");
            //Campy.Utils.Options.Set("memory_trace");
            //Campy.Utils.Options.Set("ptx_trace");
            //Campy.Utils.Options.Set("state_computation_trace");
            Campy.Utils.Options.Set("overview_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_import_computation_trace");
            //Campy.Utils.Options.Set("detailed_import_computation_trace");
            Campy.Utils.Options.Set("detailed_llvm_computation_trace");
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

        public struct mystuff
        {
            public bool _NaN;
            public bool _infinity;
            public bool _positive;
            public int _decPointPos;
            public int _defPrecision;
            public int _defMaxPrecision;
            public int _defByteSize;
            public byte[] _digits;
            public mystuff(bool doit)
            {
                _NaN = true;
                _infinity = true;
                _positive = true;
                _decPointPos = 11;
                _defPrecision = 22;
                _defMaxPrecision = 33;
                _defByteSize = 44;
                _digits = new byte[10];
            }
            public int G_decPointPos
            {
                get
                {
                    return _decPointPos;
                }
            }
            public int hi()
            {
                return _digits.Length;
            }
        }

        static void Main(string[] args)
        {
            mystuff ms = new mystuff(true);
            ms._decPointPos = 111;
            StartDebugging();
            Campy.Parallel.For(1, i =>
            {
                var nb = new mystuff(true);
                ms._defPrecision = nb.G_decPointPos;
                ms._defMaxPrecision = nb.hi();
                nb._digits[0] = 255;
                ms._digits = nb._digits;
               // System.Console.WriteLine(2.ToString());
            });
        }
    }
}
