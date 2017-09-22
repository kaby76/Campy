using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject5
{
    [TestClass]
    public class ArrayTypesGetSet
    {
        [TestMethod]
        public void ArrayTypesGetSetT()
        {
            int n = 4;
            int[] t1 = new int[n];
            Campy.Parallel.For(n, i => t1[i] = i);
            for (int i = 0; i < n; ++i) if (t1[i] != i) throw new Exception("unequal");

            float[] t2 = new float[n];
            Campy.Parallel.For(n, i => t2[i] = 0.1f * i);
            for (int i = 0; i < n; ++i) if (t2[i] != 0.1f * i) throw new Exception("unequal");

            double[] t3 = new double[n];
            Campy.Parallel.For(n, i => t3[i] = 0.1d * i);
            for (int i = 0; i < n; ++i) if (t3[i] != 0.1d * i) throw new Exception("unequal");

            System.UInt16[] t4 = new ushort[n];
            Campy.Parallel.For(n, i => t4[i] = (ushort)(i + 1));
            for (int i = 0; i < n; ++i) if (t4[i] != (ushort)(i + 1)) throw new Exception("unequal");
        }
    }
}
