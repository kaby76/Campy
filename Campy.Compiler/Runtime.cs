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
    }
}
