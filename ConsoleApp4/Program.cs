using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Messaging;
using Campy.ControlFlowGraph;
using Campy.Types;
using Swigged.LLVM;

namespace ConsoleApp4
{
    class Program
    {


        static void Main(string[] args)
        {

            var all = Campy.Types.Accelerator.GetAll();
            int[] host_data = new[] { 1, 2, 3, 4, 5 };

            GCHandle h = GCHandle.Alloc(host_data, GCHandleType.Pinned);
            var p = h.AddrOfPinnedObject();

            Campy.Parallel.For(new Extent(5), idx =>
            {
                host_data[idx] += 1;
            });
            for (int i = 0; i < host_data.Length; ++i) System.Console.WriteLine(host_data[i]);

            // FUTURE.
            //int n = Bithacks.Power2(20);
            //int[] data = new int[n];
            //Extent e = new Extent(n);
            //Campy.Parallel.For(new Extent(n), idx => data[idx] = 1);
            //for (int level = 1; level < Bithacks.Log2(n); level++)
            //{
            //    int step = Bithacks.Power2(level);
            //    Campy.Parallel.For(new Extent(n / step), idx =>
            //    {
            //        var i = step * idx;
            //        data[i] = data[i] + data[i + step / 2];
            //    });
            //}
            //for (int i = 0; i < data.Length; ++i)
            //    System.Console.WriteLine(data[i]);
        }
    }
}
