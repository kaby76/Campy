using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace JaggedArrayList
{
    [TestClass]
    public class JaggedArrayList
    {
        [TestMethod]
        public void JaggedArrayListT()
        {
            // List of ints.
            List<List<int>> x = new List<List<int>>();
            int n = 4;
            for (int i = 0; i < n; ++i)
                x.Add(new List<int>());
            for (int i = 0; i < n; ++i)
                x[i].Add(0);
            Campy.Parallel.For(n, i => x[i][0] = i);
            for (int i = 0; i < n; ++i)
                if (x[i][0] != i)
                    throw new Exception("unequal");
        }
    }
}
