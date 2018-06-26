using System;
using System.Linq;
using Xunit;

namespace SimpleVirtualFunction
{
    class B
    {
        public virtual int Yo()
        {
            return 1;
        }
    }

    class A : B
    {
        public override int Yo()
        {
            return 2;
        }
    }

    public class UnitTest1
    {
        [Fact]
        public void TestMethod1()
        {
            var a = new A();
            var b = new B();
            Campy.Parallel.Compile(typeof(A));
            Campy.Parallel.Compile(typeof(B));
            B c = a;
            System.Console.WriteLine(c.Yo());
            int[] xx = new int[4];
            Campy.Parallel.For(4, i =>
            {
                xx[i] = c.Yo();
            });
            for (int i = 0; i < 4; ++i)
                if (xx[i] != 2) throw new Exception();
        }
    }
}
