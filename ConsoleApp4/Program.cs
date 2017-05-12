using Campy.CIL;
using Campy.LCFG;

namespace ConsoleApp4
{
    class Program
    {
        static int Foo1()
        {
            return 1;
        }

        static int Foo2(int a)
        {
            return a + 1;
        }

        static void Main(string[] args)
        {
            Reader r = new Reader();
            r.AnalyzeMethod(() => Program.Foo2(1));
            var mg = r.Cfg;
            var lg = new LLVMCFG();
            var c2 = new Campy.LCFG.Converter(mg, lg);
            c2.ConvertToLLVM(mg.Vertices);
            c2.Call();
        }
    }
}
