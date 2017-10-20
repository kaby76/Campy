using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Campy;

namespace MultiDimensionalArraysNS
{
    [TestClass]
    public class MultiDimensionalArrays
    {
        [TestMethod]
        public void MultiDimensionalArraysT()
        {
            {
                Extent ex = new Extent(3, 5); // three rows, five columns.
                int[,] b = new int[ex[0], ex[1]];
                for (int d = 0; d < ex[0] * ex[1]; ++d)
                {
                    int i = d / ex[1];
                    int j = d % ex[1];
                    b[i, j] = d;
                }

                int[,] c = new int[ex[0], ex[1]];
                Campy.Parallel.For(15, d =>
                {
                    int i = d / ex[1];
                    int j = d % ex[1];
                    c[i, j] = d;
                });

                for (int d = 0; d < ex[0] * ex[1]; ++d)
                {
                    int i = d / ex[1];
                    int j = d % ex[1];
                    if (b[i, j] != c[i, j])
                        throw new Exception();
                }
            }
            {
                Extent ex = new Extent(3, 5, 2);
                int[,,] b = new int[ex[0], ex[1], ex[2]];
                for (int d = 0; d < ex[0] * ex[1] * ex[2]; ++d)
                {
                    // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
                    int i = d / (ex[1] * ex[2]);
                    int r = d % (ex[1] * ex[2]);
                    int j = r / ex[2];
                    int k = r % ex[2];
                    b[i, j, k] = d;
                }

                int[,,] c = new int[ex[0], ex[1], ex[2]];
                Campy.Parallel.For(ex[0] * ex[1] * ex[2], d =>
                {
                    int i = d / (ex[1] * ex[2]);
                    int r = d % (ex[1] * ex[2]);
                    int j = r / ex[2];
                    int k = r % ex[2];
                    c[i, j, k] = d;
                });

                for (int d = 0; d < ex[0] * ex[1] * ex[2]; ++d)
                {
                    // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
                    int i = d / (ex[1] * ex[2]);
                    int r = d % (ex[1] * ex[2]);
                    int j = r / ex[2];
                    int k = r % ex[2];
                    if (b[i, j, k] != c[i, j, k])
                        throw new Exception();
                }

            }
        }
    }
}
