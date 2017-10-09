using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject10
{
    [TestClass]
    public class Saxpy
    {
        [TestMethod]
        public void SaxpyT()
        {
            // Saxpy (vector update).
            int n = 2;
            double[] x = new double[n];
            double[] y = new double[n];
            float a = 10f;

            Campy.Parallel.For(n, i => x[i] = i);
            Campy.Parallel.For(n, i => y[i] = i - 1);
            Campy.Parallel.For(n, i =>
            {
                y[i] = y[i] + a * x[i];
            });
            double[] answer = new double[] { -1, 10 };
            for (int i = 0; i < n; ++i)
                if (answer[i] != y[i]) throw new Exception("unequal");
        }
    }
}
