using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject1
{
    [TestClass]
    public class IntArraySet
    {
        [TestMethod]
        public void IntArraySetT()
        {
            int n = 4;
            int[] x = new int[n];
            Campy.Parallel.For(n, i => x[i] = i);
            for (int i = 0; i < n; ++i) if (x[i] != i)
                throw new Exception();
        }
    }
}
