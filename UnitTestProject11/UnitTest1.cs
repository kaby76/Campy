using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject11
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            double[] buffer = new double[] {0.1, 0.2, 0.3};
            Campy.Parallel.For(buffer.Length, k =>
            {
                buffer[k] = Math.Sin(buffer[k]);
            });
        }
    }
}
