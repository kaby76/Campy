using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace OceaniasTest461NS
{
    [TestClass]
    public class OceaniasTest461C
    {
        [TestMethod]
        public void OceaniasTest461()
        {
            int n = 10000;
            var rand = new Random();
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = rand.NextDouble();
            }

            Campy.Parallel.For(n, i =>
            {
                y[i] = 1 / (1 + Math.Pow(Math.E, -x[i]));
            });

	        for (int i = 0; i < n; ++i)
		        Assert.IsTrue(0 < y[i] && y[i] < 1);
	    }
    }
}
