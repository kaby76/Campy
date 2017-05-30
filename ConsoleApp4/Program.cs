using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Campy.CIL;
using Campy.LCFG;
using Swigged.LLVM;

namespace ConsoleApp4
{
    class Program
    {
        static int Foo1()
        {
            return 1;
        }

        static int Foo2(int a)
        {
            return a + 1;
        }

        static int Foo3(int b)
        {
            if (b > 10)
                return b + 1;
            else
                return b + 2;
            return 99;
        }

        public delegate int DFoo2(int a);

        static void Main(string[] args)
        {
            Reader r = new Reader();
            var mg = r.Cfg;
            mg.StartChangeSet(1);
            r.AnalyzeMethod(() => Program.Foo2(1));
            List<CIL_CFG.Vertex> change_set = mg.EndChangeSet(1);
            var lg = new LLVMCFG();
            var c2 = new Campy.LCFG.Converter(mg, lg);
            Swigged.LLVM.Helper.Adjust.Path();
            c2.ConvertToLLVM(change_set);

            IntPtr p2 = c2.GetPtr(1);
            DFoo2 ff2 = (DFoo2)Marshal.GetDelegateForFunctionPointer(p2, typeof(DFoo2));
            for (int k = 0; k < 100; ++k)
            {
                int result = ff2(k);
                Console.WriteLine("Result is: " + result);
            }

            mg.StartChangeSet(2);
            r.AnalyzeMethod(() => Program.Foo3(2));
            List<CIL_CFG.Vertex> change_set2 = mg.EndChangeSet(2);
            c2.ConvertToLLVM(change_set2);
            IntPtr p3 = c2.GetPtr(3);
            DFoo2 ff3 = (DFoo2)Marshal.GetDelegateForFunctionPointer(p3, typeof(DFoo2));
            for (int k = 0; k < 100; ++k)
            {
                int result = ff3(k);
                Console.WriteLine("Result is: " + result);
            }
        }
    }
}
