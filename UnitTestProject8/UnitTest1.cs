using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Campy;
using System.Collections.Generic;

namespace UnitTestProject8
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            int n = 4;

            var t1 = new List<int>();
            for (int i = 0; i < n; ++i) t1.Add(0);
            Campy.Parallel.For(n, i => t1[i] = i);
            for (int i = 0; i < n; ++i) if (t1[i] != i) throw new Exception("unequal");

            var t2 = new List<float>();
            for (int i = 0; i < n; ++i) t2.Add(0);
            Campy.Parallel.For(n, i => t2[i] = 0.1f * i);
            for (int i = 0; i < n; ++i) if (t2[i] != 0.1f * i) throw new Exception("unequal");

            var t3 = new List<double>();
            for (int i = 0; i < n; ++i) t3.Add(0);
            Campy.Parallel.For(n, i => t3[i] = 0.1d * i);
            for (int i = 0; i < n; ++i) if (t3[i] != 0.1d * i) throw new Exception("unequal");

            var t4 = new List<ushort>();
            for (int i = 0; i < n; ++i) t4.Add(0);
            Campy.Parallel.For(n, i => t4[i] = (ushort)(i + 1));
            for (int i = 0; i < n; ++i) if (t4[i] != (ushort)(i + 1)) throw new Exception("unequal");

            var t5 = new List<int>();
            for (int i = 0; i < n; ++i) t5.Add(0);
            Campy.Parallel.For(n, i => t5[i] = t1[i] * 2);
            for (int i = 0; i < n; ++i) if (t5[i] != t1[i] * 2) throw new Exception("unequal");

            var t6 = new List<float>();
            for (int i = 0; i < n; ++i) t6.Add(0);
            Campy.Parallel.For(n, i => t6[i] = 0.1f * i + t2[i]);
            for (int i = 0; i < n; ++i) if (t6[i] != 0.1f * i + t2[i]) throw new Exception("unequal");

            var t7 = new List<double>();
            for (int i = 0; i < n; ++i) t7.Add(0);
            Campy.Parallel.For(n, i => t7[i] = 0.1f * i - t3[i]);
            for (int i = 0; i < n; ++i) if (t7[i] != 0.1f * i - t3[i]) throw new Exception("unequal");

            var t8 = new List<ushort>();
            for (int i = 0; i < n; ++i) t8.Add(0);
            Campy.Parallel.For(n, (Index i) =>
            {
                t8[i] = (ushort)(t4[i] + i + 1);
            });
            for (int i = 0; i < n; ++i) if (t8[i] != (ushort)(t4[i] + i + 1)) throw new Exception("unequal");
        }
    }
}
