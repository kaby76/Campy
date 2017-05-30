using System.Collections.Generic;
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

        static int Foo3(int b)
        {
            if (b > 10)
                return b + 1;
            else
                return b + 2;
            return 99;
        }

        static void Main(string[] args)
        {
            Reader r = new Reader();
            var mg = r.Cfg;
          //  mg.StartChangeSet(1);
           // r.AnalyzeMethod(() => Program.Foo2(1));
         //   List<CIL_CFG.Vertex> change_set = mg.EndChangeSet(1);
            var lg = new LLVMCFG();
            var c2 = new Campy.LCFG.Converter(mg, lg);
            Swigged.LLVM.Helper.Adjust.Path();
            //c2.ConvertToLLVM(change_set);
           // c2.Call();

            mg.StartChangeSet(2);
            r.AnalyzeMethod(() => Program.Foo3(2));
            List<CIL_CFG.Vertex> change_set2 = mg.EndChangeSet(2);
            c2.ConvertToLLVM(change_set2);
            c2.Call();

        }
    }
}
