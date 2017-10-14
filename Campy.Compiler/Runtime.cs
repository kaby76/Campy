using System;

namespace Campy.Compiler
{
    public class Runtime
    {
        // Arrays are implemented as a struct, with the data following the struct
        // in row major format. Note, each dimension has a length that is recorded
        // following the pointer p. The one shown here is for only one-dimensional
        // arrays.
        // Calls have to be casted to this type.
        public unsafe struct A
        {
            public void* p;
            public long d;
            public long l;
        }

        public static unsafe int get_length_multi_array(A* arr, int i0)
        {
            byte* bp = (byte*)arr;
            bp = bp + 16 + 8 * i0;
            long* lp = (long*)bp;
            return (int) *lp;
        }

        public static unsafe int get_multi_array(A* arr, int i0)
        {
            int* a = *(int**)arr;
            return *(a + i0);
        }

        public static unsafe int get_multi_array(A* arr, int i0, int i1)
        {
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            byte* bp = (byte*)arr;
            bp = bp + 24;
            long o = 0;
            long* lp = (long*)bp;
            o = (*lp) * i0 + i1;
            return *(a + o);
        }

        public static unsafe int get_multi_array(A* arr, int i0, int i1, int i2)
        {
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            byte* bp = (byte*)arr;
            bp = bp + 24;
            long o = 0;
            long* lp = (long*)bp;
            o = (*lp) * i0 + i1;
            return  *(a + o);
        }

        public static unsafe void set_multi_array(A* arr, int i0, int value)
        {
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            long o = i0;
            *(a + o) = value;
        }

        public static unsafe void set_multi_array(A* arr, int i0, int i1, int value)
        {
            int* a = (int*)(*arr).p;
            int d = (int)(*arr).d;
            byte* bp = (byte*)arr;
            bp = bp + 24;
            long o = 0;
            long* lp = (long*)bp;
            o = (*lp) * i0 + i1;
            *(a + o) = value;
        }

        public static unsafe void set_multi_array(A* arr, int i0, int i1, int i2, int value)
        {
        }

        public static double Sine(double x)
        {
            const double PI = 3.14159265358979323846264338327950288f;
            const double PI_SQR = 9.86960440108935861883449099987615114f;
            const double B = 4 / PI;
            const double C = -4 / PI_SQR;
            const double P = 0.225;

            double xp = x < 0 ? -x : x;
            double y = B * x + C * x * xp;
            double yp = y < 0 ? -y : y;
            y = P * (y * yp - y) + y;
            return y;
        }

        public double Cosine(double x)
        {
            const double PID2 = 1.57079632679489661923132169163975144f;
            const double hpi = PID2;
            x += hpi; //shift for cosine
            return Sine(x);
        }

        public double Abs(double x)
        {
            return x > 0 ? x : -x;
        }
    }
}
