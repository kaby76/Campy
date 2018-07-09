using System;
using System.Linq;
using Xunit;

namespace EhFallThru
{
    public class UnitTest1
    {
        [Fact]
        public void TestMethod1()
        {
            int l = -1;
            Campy.Parallel.For(3, i =>
            {
                int j = i;
                int k = 0;
                try
                {
                    int m = 1;
                    k = -2 * i;
                    l = 99;
                }
                catch (Exception e)
                {
                    j = -i;
                }
                finally
                {
                    k = j;
                }
            });
            if (l != 99) throw new Exception();
        }
    }
}
