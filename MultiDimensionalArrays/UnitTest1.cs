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
                int ex0 = 3;
                int ex1 = 5;
                // three rows, five columns.
                int[,] b = new int[ex0, ex1];
                for (int d = 0; d < ex0 * ex1; ++d)
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    b[i, j] = d;
                }

                int[,] c = new int[ex0, ex1];
                Campy.Parallel.For(15, d =>
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    c[i, j] = d;
                });

                for (int d = 0; d < ex0 * ex1; ++d)
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    if (b[i, j] != c[i, j])
                        throw new Exception();
                }
            }
            {
                int ex0 = 3;
                int ex1 = 5;
                int ex2 = 2;
                int[,,] b = new int[ex0, ex1, ex2];
                for (int d = 0; d < ex0 * ex1 * ex2; ++d)
                {
                    // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
                    int i = d / (ex1 * ex2);
                    int r = d % (ex1 * ex2);
                    int j = r / ex2;
                    int k = r % ex2;
                    b[i, j, k] = d;
                }

                int[,,] c = new int[ex0, ex1, ex2];
                Campy.Parallel.For(ex0 * ex1 * ex2, d =>
                {
                    int i = d / (ex1 * ex2);
                    int r = d % (ex1 * ex2);
                    int j = r / ex2;
                    int k = r % ex2;
                    c[i, j, k] = d;
                });

                for (int d = 0; d < ex0 * ex1 * ex2; ++d)
                {
                    // long d = i2 + i1 * ex2 + i0 * ex2 * ex1;
                    int i = d / (ex1 * ex2);
                    int r = d % (ex1 * ex2);
                    int j = r / ex2;
                    int k = r % ex2;
                    if (b[i, j, k] != c[i, j, k])
                        throw new Exception();
                }

            }
        }
    }
}
