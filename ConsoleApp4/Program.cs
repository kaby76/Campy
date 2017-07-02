using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Campy.ControlFlowGraph;
using Campy.Types;
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

        static int fact(int b)
        {
            if (b == 0) return 1;
            else if (b == 1) return 1;
            else return b * fact(b - 1);
        }

        static int SumOf3Or5(int high)
        {
            int result = 0;
            for (int i = 1; i < high; ++i)
            {
                if (i % 3 == 0) result += i;
                if (i % 5 == 0) result += i;
            }
            return result;
        }


        public delegate int DFoo2(int a);

        public static long Ackermann(long m, long n)
        {
            if (m > 0)
            {
                if (n > 0)
                    return Ackermann(m - 1, Ackermann(m, n - 1));
                else if (n == 0)
                    return Ackermann(m - 1, 1);
            }
            else if (m == 0)
            {
                if (n >= 0)
                    return n + 1;
            }
            return -1;
        }

        public delegate long DAck(long a, long b);

        public static int CountChar(char c, char[] a)
        {
            return a.Length;
            int result = 0;
            for (int i = 0; i < a.Length; ++i)
                if (c == a[i]) result++;
            return result;
        }
        public delegate long DCountChar(char c, char[] a);


        public static Int64 SimpleCount(int ind, Int64[] a)
        {
            return a[ind];
        }
        public delegate Int64 DSimpleCount(int ind, Int64[] a);
        public static Int32 SimpleCount2(int ind, Int32[] a)
        {
            return a[ind];
        }
        public delegate Int32 DSimpleCount2(int ind, Int32[] a);

        public static Int16 SimpleCount3(int ind, Int16[] a)
        {
            return a[ind];
        }
        public delegate Int16 DSimpleCount3(int ind, Int16[] a);

        public static Char SimpleCount4(int ind, Char[] a)
        {
            return a[ind];
        }
        public delegate Char DSimpleCount4(int ind, Char[] a);

        public static Int64 SimpleCount5(int[] a)
        {
            return a.Length;
        }
        public delegate Int64 DSimpleCount5(int[] a);


        static void Main(string[] args)
        {
            var all = Campy.Types.Accelerator.GetAll();
            int[] host_data = new[] { 1, 2, 3, 4, 5 };
            ArrayView<int> av = new ArrayView<int>(ref host_data);
            Campy.Parallel.For(av.Extent, idx => { av[idx] += 1; });
            for (int i = 0; i < host_data.Length; ++i)
                System.Console.WriteLine(av[i]);

            //Swigged.LLVM.Helper.Adjust.Path();

            //Reader r = new Reader();
            //var g = r.Cfg;
            //var c = new Campy.ControlFlowGraph.Converter(g);

            //int sz = int.Parse("33");
            //int[] Sa = new int[sz];
            //for (int kk = 0; kk < Sa.Length; ++kk) Sa[kk] = kk;
            //System.Console.WriteLine(Sa.Length);


            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.Foo2(1));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
            //    for (int k = 0; k < 100; ++k)
            //        Console.WriteLine("Result is: " + f(k));
            //}

            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.Foo3(2));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
            //    for (int k = 0; k < 100; ++k)
            //        Console.WriteLine("Result is: " + f(k));
            //}

            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.fact(2));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
            //    for (int k = 0; k < 10; ++k)
            //        Console.WriteLine("Result is: " + f(k));
            //}

            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.SumOf3Or5(2));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
            //    for (int k = 0; k < 10; ++k)
            //        Console.WriteLine("Result is: " + f(1000));
            //    int pp = SumOf3Or5(1000);
            //}

            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.Ackermann(2, 2));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DAck f = (DAck)Marshal.GetDelegateForFunctionPointer(p, typeof(DAck));
            //    for (long m = 0; m <= 3; ++m)
            //    {
            //        for (long n = 0; n <= 4; ++n)
            //        {
            //            Console.WriteLine();
            //            Console.WriteLine("Ackermann({0}, {1}) = {2}", m, n, Ackermann(m, n));
            //            long result = f(m, n);
            //            Console.WriteLine("Result is: " + result);
            //        }
            //    }
            //}

            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.SimpleCount(1, new Int64[2]));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DSimpleCount f = (DSimpleCount)Marshal.GetDelegateForFunctionPointer(p, typeof(DSimpleCount));
            //    Console.WriteLine();
            //    var dataArray = new Int64[10] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            //    for (int j = 0; j < 10; ++j)
            //    {
            //        Int64 result = f(j, dataArray);
            //        Console.WriteLine("Result is: " + result);
            //    }
            //}
            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.SimpleCount2(1, new Int32[2]));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DSimpleCount2 f = (DSimpleCount2)Marshal.GetDelegateForFunctionPointer(p, typeof(DSimpleCount2));
            //    Console.WriteLine();
            //    var dataArray = new Int32[10] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            //    for (int j = 0; j < 10; ++j)
            //    {
            //        Int64 result = f(j, dataArray);
            //        Console.WriteLine("Result is: " + result);
            //    }
            //}
            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.SimpleCount3(1, new Int16[2]));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DSimpleCount3 f = (DSimpleCount3)Marshal.GetDelegateForFunctionPointer(p, typeof(DSimpleCount3));
            //    Console.WriteLine();
            //    var dataArray = new Int16[10] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            //    for (int j = 0; j < 10; ++j)
            //    {
            //        Int64 result = f(j, dataArray);
            //        Console.WriteLine("Result is: " + result);
            //    }
            //}
            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.SimpleCount4(1, new Char[2]));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DSimpleCount4 f = (DSimpleCount4)Marshal.GetDelegateForFunctionPointer(p, typeof(DSimpleCount4));
            //    Console.WriteLine();
            //    var dataArray = new Char[10] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' };
            //    for (int j = 0; j < 10; ++j)
            //    {
            //        Char result = f(j, dataArray);
            //        Console.WriteLine("Result is: " + result);
            //    }
            //}
            //{
            //    g.StartChangeSet(r);
            //    r.AnalyzeMethod(() => Program.SimpleCount5(new int[2]));
            //    List<CFG.Vertex> cs = g.PopChangeSet(r);
            //    c.ConvertToLLVM(cs);
            //    IntPtr p = c.GetPtr(cs.First().Name);
            //    DSimpleCount5 f = (DSimpleCount5)Marshal.GetDelegateForFunctionPointer(p, typeof(DSimpleCount5));
            //    Console.WriteLine();
            //    for (int j = 0; j < 10; ++j)
            //    {
            //        unsafe
            //        {
            //            fixed (int * ffff = Sa)
            //            {
            //                System.Console.WriteLine();
            //            }
            //            Int64 result = f(Sa);
            //            Console.WriteLine("Result is: " + result);
            //        }
            //    }
            //}

        }
    }
}
