using System;
using Xunit;
using Campy;

namespace ArrayOfRefs
{
    // A simple test from https://stackoverflow.com/questions/48157879/alea-i32-is-not-struct-type

    class A
    {
        public int X { get; set; }

        public int Score(A b)
        {
            return X + b.X;
        }
    }

    public class UnitTest1
    {
        [Fact]
        public void Test1()
        {
            A[] array = new A[10];
            for (int i = 0; i < 10; ++i) array[i] = new A();

            Campy.Parallel.For(10, i =>
            {
                array[i].X = i;
            });

            for (int i = 0; i < 10; i++)
            {
                if (array[i].X != i) throw new Exception();
            }
        }
    }
}
