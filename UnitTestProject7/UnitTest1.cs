using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTestProject7
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            int[][] jagged_array = new int[][]
            {
                new int[] {1, 3, 5, 7, 9},
                new int[] {0, 2, 4, 6},
                new int[] {11, 22}
            };
            Campy.Parallel.For(3, i =>
            {
                jagged_array[i][0] = jagged_array[i].Length;
            });
            for (int i = 0; i < 3; ++i)
                if (jagged_array[i][0] != jagged_array[i].Length)
                    throw new Exception("unequal");
        }
    }
}

