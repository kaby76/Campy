using System;
using Campy;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace TwoDimArrayInts
{
    [TestClass]
    public class TwoDimArrayInts
    {
        [TestMethod]
        public void TwoDimArrayIntsT()
        {
            int e = 10;
            int ex0 = 3;
            int ex1 = 5;
            int[,] b = new int[ex0, ex1];
            for (int i = 0; i < ex0; ++i)
            for (int j = 0; j < ex1; ++j)
                b[i, j] = (i + 1) * (j + 1);
            Campy.Parallel.For(5, d =>
            {
                b[d % 3, d] = 33 + d;
            });
            if (b[0, 0] != 33) throw new Exception();
            if (b[1, 1] != 34) throw new Exception();
            if (b[2, 2] != 35) throw new Exception();
            if (b[0, 3] != 36) throw new Exception();
            if (b[1, 4] != 37) throw new Exception();
        }
    }
}
