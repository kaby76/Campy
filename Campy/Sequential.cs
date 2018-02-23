using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy
{
    public class Sequential
    {
        public static void For(int number_of_threads, KernelType kernel)
        {
            for (int i = 0; i < number_of_threads; ++i)
            {
                kernel(i);
            }
        }
    }
}
