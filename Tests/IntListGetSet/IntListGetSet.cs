using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject4
{
    [TestClass]
    public class IntListGetSet
    {
        [TestMethod]
        public void IntListGetSetT()
        {
            List<int> x = new List<int>();
            int n = 4;
            for (int i = 0; i < n; ++i) x.Add(0);
            Campy.Parallel.For(n, i => x[i] = i);
            Campy.Parallel.For(n, i => x[i] = x[i] * 2);
            for (int i = 0; i < n; ++i) if (x[i] != 2 * i)
                throw new Exception();
        }
    }
}
