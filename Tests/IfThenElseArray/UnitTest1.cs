using System;
using System.Linq;
using Xunit;

namespace IfThenElseArray
{
    public class UnitTest1
    {
        [Fact]
        public void TestMethod1()
        {
            int n = 4;
            var t1 = new int[n];
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
                    if (t1[i] != i * 20) throw new Exception();
                }
                else
                {
                    if (t1[i] != i * 30) throw new Exception();
                }
        }
    }
}
