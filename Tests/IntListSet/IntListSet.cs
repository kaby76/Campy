using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject2
{
    [TestClass]
    public class IntListSet
    {
        [TestMethod]
        public void IntListSetT()
        {
            // List of ints.
            List<int> x = new List<int>();
            int n = 4;
            for (int i = 0; i < n; ++i) x.Add(0);
            Campy.Parallel.For(n, i => x[i] = i);
            for (int i = 0; i < n; ++i) if (x[i] != i)
                throw new Exception();
        }
    }
}
