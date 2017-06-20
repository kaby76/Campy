using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Campy.ControlFlowGraph;
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


        static void Main(string[] args)
        {
            Swigged.LLVM.Helper.Adjust.Path();

            Reader r = new Reader();
            var g = r.Cfg;
            var c = new Campy.ControlFlowGraph.Converter(g);

            {
                g.StartChangeSet(r);
                r.AnalyzeMethod(() => Program.Foo2(1));
                List<CFG.Vertex> cs = g.EndChangeSet(r);
                c.ConvertToLLVM(cs);
                IntPtr p = c.GetPtr(cs.First().Name);
                DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
                for (int k = 0; k < 100; ++k)
                    Console.WriteLine("Result is: " + f(k));
            }

            {
                g.StartChangeSet(r);
                r.AnalyzeMethod(() => Program.Foo3(2));
                List<CFG.Vertex> cs = g.EndChangeSet(r);
                c.ConvertToLLVM(cs);
                IntPtr p = c.GetPtr(cs.First().Name);
                DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
                for (int k = 0; k < 100; ++k)
                    Console.WriteLine("Result is: " + f(k));
            }

            {
                g.StartChangeSet(r);
                r.AnalyzeMethod(() => Program.fact(2));
                List<CFG.Vertex> cs = g.EndChangeSet(r);
                c.ConvertToLLVM(cs);
                IntPtr p = c.GetPtr(cs.First().Name);
                DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
                for (int k = 0; k < 10; ++k)
                    Console.WriteLine("Result is: " + f(k));
            }

            {
                g.StartChangeSet(r);
                r.AnalyzeMethod(() => Program.SumOf3Or5(2));
                List<CFG.Vertex> cs = g.EndChangeSet(r);
                c.ConvertToLLVM(cs);
                IntPtr p = c.GetPtr(cs.First().Name);
                DFoo2 f = (DFoo2) Marshal.GetDelegateForFunctionPointer(p, typeof(DFoo2));
                for (int k = 0; k < 10; ++k)
                    Console.WriteLine("Result is: " + f(1000));
                int pp = SumOf3Or5(1000);
            }

            {
                g.StartChangeSet(r);
                r.AnalyzeMethod(() => Program.Ackermann(2, 2));
                List<CFG.Vertex> cs = g.EndChangeSet(r);
                c.ConvertToLLVM(cs);
                IntPtr p = c.GetPtr(cs.First().Name);
                DAck f = (DAck)Marshal.GetDelegateForFunctionPointer(p, typeof(DAck));
                for (long m = 0; m <= 3; ++m)
                {
                    for (long n = 0; n <= 4; ++n)
                    {
                        Console.WriteLine();
                        Console.WriteLine("Ackermann({0}, {1}) = {2}", m, n, Ackermann(m, n));
                        long result = f(m, n);
                        Console.WriteLine("Result is: " + result);
                    }
                }
            }

            {
                g.StartChangeSet(r);
                r.AnalyzeMethod(() => Program.SimpleCount(1, new Int64[2]));
                List<CFG.Vertex> cs = g.EndChangeSet(r);
                c.ConvertToLLVM(cs);
                IntPtr p = c.GetPtr(cs.First().Name);
                DSimpleCount f = (DSimpleCount)Marshal.GetDelegateForFunctionPointer(p, typeof(DSimpleCount));
                Console.WriteLine();
                var dataArray = new Int64[10]{1,2,3,4,5,6,7,8,9,10};
                for (int j = 0; j < 10; ++j)
                {
                    Int64 result = f(j, dataArray);
                    Console.WriteLine("Result is: " + result);
                }
            }
        }
    }
}
