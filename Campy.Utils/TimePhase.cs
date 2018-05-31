using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Campy.Utils
{
    public class TimePhase
    {
        public static void Time(string phase_name, Action action)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Reset();
            stopwatch.Start();

            action();

            stopwatch.Stop();
            var elapse = stopwatch.Elapsed;
            if (Campy.Utils.Options.IsOn("jit_trace"))
                System.Console.WriteLine(phase_name + elapse);
        }
    }
}
