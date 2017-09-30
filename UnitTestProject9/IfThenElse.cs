using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace UnitTestProject9
{
    [TestClass]
    public class IfThenElse
    {
        [TestMethod]
        public void IfThenElseT()
        {
            int n = 4;

            var t1 = new List<int>();
            for (int i = 0; i < n; ++i) t1.Add(0);
            Campy.Parallel.For(n, i =>
            {
                if (i % 2 == 0)
                    t1[i] = i * 20;
                else
                    t1[i] = i * 30;
            });
            for (int i = 0; i < n; ++i)
                if (i % 2 == 0)
                {
                    if (t1[i] != i * 20) throw new Exception("unequal");
                }
                else
                {
                    if (t1[i] != i * 30) throw new Exception("unequal");
                }
        }
    }
}
