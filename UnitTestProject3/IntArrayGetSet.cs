using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject3
{
    [TestClass]
    public class IntArrayGetSet
    {
        [TestMethod]
        public void IntArrayGetSetT()
        {
            int n = 4;
            int[] x = new int[n];
            Campy.Parallel.For(n, i => x[i] = i);
            Campy.Parallel.For(n, i => x[i] = x[i] * 2);
            for (int i = 0; i < n; ++i) if (x[i] != 2*i)
                throw new Exception("unequal");
        }
    }
}
