using System;
using Xunit;

namespace TwoDimArrayGetAndSet
{
    public class UnitTest1
    {
        [Fact]
        public void Test1()
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
                    c[i, j] = b[i, j];
                });

                for (int d = 0; d < ex0 * ex1; ++d)
                {
                    int i = d / ex1;
                    int j = d % ex1;
                    if (b[i, j] != c[i, j])
                        throw new Exception();
                }
            }
        }
    }
}
