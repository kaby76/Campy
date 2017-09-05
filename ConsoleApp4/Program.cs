using Campy.Types;

namespace ConsoleApp4
{
    class N
    {
        public int a;
        public int b;
    }

    class Program
    {
        static void Main(string[] args)
        {
            int n = Bithacks.Power2(4);
            int[] data = new int[n];
            Campy.Parallel.For(new Extent(n), idx => data[idx] = 1);
            for (int level = 1; level <= Bithacks.Log2(n); level++)
            {
                int step = Bithacks.Power2(level);
                Campy.Parallel.For(new Extent(n / step), idx =>
                {
                    var i = step * idx;
                    data[i] = data[i] + data[i + step / 2];
                });
                System.Console.WriteLine("level " + level);
                for (int i = 0; i < data.Length; ++i)
                    System.Console.Write(data[i] + " ");
                System.Console.WriteLine();
            }

            for (int i = 0; i < data.Length; ++i)
                System.Console.WriteLine(data[i]);

            //N a = new N() { a = 1, b = 1 };
            //N b = new N() { a = 1, b = 1 };
            //N c = new N() { a = 1, b = 1 };
            //N d = new N() { a = 1, b = 1 };
            //N e = new N() { a = 1, b = 1 };
            //N f = new N() { a = 1, b = 1 };
            //N g = new N() { a = 1, b = 1 };
            //N h = new N() { a = 1, b = 1 };
            //N[] stuff = new N[] { a, b, c, d, e, f, g, h };

            //Campy.Parallel.For(new Extent(8), idx => stuff[idx].a = stuff[idx].a + stuff[idx].b);
        }
    }
}
