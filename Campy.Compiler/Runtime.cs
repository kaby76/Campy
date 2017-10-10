using System;

namespace Campy.Compiler
{
    public class Runtime
    {
        public static unsafe int get_multi_array(void* arr, int i0)
        {
            int* a = *(int**) arr;
            return *(a+i0);
        }

        public static unsafe int get_multi_array(void* arr, int i0, int i1)
        {
            int* a = *(int**) arr;
            byte* l = (byte*) arr;
            l += sizeof(byte*);
            int len = *(int*) l;
            return *(a + len*i0 + i1);
        }

        public static unsafe int get_multi_array(void* arr, int i0, int i1, int i2)
        {
            int* a = *(int**)arr;
            byte* l = (byte*)arr;
            l += sizeof(byte*);
            int len0 = *(int*)l;
            l += sizeof(byte*);
            int len1 = *(int*)l;
            return *(a + len0 * i0 + len1 * i1 + i2);
        }

        public static unsafe void set_multi_array(void* arr, int i0, int value)
        {
        }

        public static unsafe void set_multi_array(void* arr, int i0, int i1, int value)
        {
        }

        public static unsafe void set_multi_array(void* arr, int i0, int i1, int i2, int value)
        {
        }
    }
}
