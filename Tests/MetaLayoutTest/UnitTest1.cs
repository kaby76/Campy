using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Campy;

namespace MetaLayoutTest
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            int a = 0;
            bool b = false;
            float c = (float)0.0;
            double d = 0.0;
            Campy.Parallel.For(1, i =>
            {
                a = 1;
                b = true;
                c = (float)2.0;
                d = 3.0;
            });
            if (a != 1) throw new Exception();
            if (b != true) throw new Exception();
            if (c != (float)2.0) throw new Exception();
            if (d != 3.0) throw new Exception();

            //        throw new Exception();
        }
    }
}
