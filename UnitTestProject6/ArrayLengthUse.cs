using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject6
{
    [TestClass]
    public class ArrayLengthUse
    {
        [TestMethod]
        public void ArrayLengthUseT()
        {
            int n = 4;
            int[] x = new int[n];
            Campy.Parallel.For(n, i => x[i] = x.Length);
            for (int i = 0; i < n; ++i) if (x[i] != x.Length)
                throw new Exception("unequal");
        }
    }
}
