using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace BatcherEvenOddMerge
{
    public class BatcherOddEvenMergeSort
    {
        public static void swap(ref int i, ref int j)
        {
            int t = i;
            i = j;
            j = t;
        }

        // Adapted from http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm

        /** sorts a piece of length n of the array
         * starting at position lo
         */
        public static void OddEvenMergeSort(int[] a, int lo, int n)
        {
            if (n > 1)
            {
                int m = n / 2;
                OddEvenMergeSort(a, lo, m);
                OddEvenMergeSort(a, lo + m, m);
                OddEvenMerge(a, lo, n, 1);
            }
        }

        /** lo is the starting position and
         * n is the length of the piece to be merged,
         * r is the distance of the elements to be compared
         */
        public static void OddEvenMerge(int[] a, int lo, int n, int r)
        {
            int m = r * 2;
            if (m < n)
            {
                OddEvenMerge(a, lo, n, m); // even subsequence
                OddEvenMerge(a, lo + r, n, m); // odd subsequence
                for (int i = lo + r; i + r < lo + n; i += m)
                    if (a[i] > a[i + r])
                        swap(ref a[i], ref a[i + r]);
            }
            else if (a[lo] > a[lo + r])
                swap(ref a[lo], ref a[lo + r]);
        }

        public static void show_pairs(int n)
        {
            for (int p = 1; p < n; p += p)
            {
                System.Console.WriteLine("p " + p);
                for (int k = p; k > 0; k /= 2)
                {
                    System.Console.WriteLine("k " + k);
                    for (int j = 0; j < k; j++)
                    {
                        System.Console.WriteLine("j " + j);
                        for (int i = k % p; i + k < n; i += k + k)
                        {
                            System.Console.WriteLine("i " + i);
                            if ((i + j) / (p + p) == (i + j + k) / (p + p))
                                System.Console.WriteLine((i + j).ToString() + " cmp " + (i + j + k).ToString());
                        }
                    }
                }
            }
        }
    }


    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            Random rnd = new Random();
            int N = 8;
            BatcherOddEvenMergeSort.show_pairs(N);
            {
                int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
                BatcherOddEvenMergeSort.OddEvenMergeSort(a, 0, N);
                for (int i = 0; i < N; ++i)
                    if (a[i] != i)
                        throw new Exception();
            }
        }
    }

    // Support
    public class Bithacks
    {
        static bool preped;

        static int[] LogTable256 = new int[256];

        static void prep()
        {
            LogTable256[0] = LogTable256[1] = 0;
            for (int i = 2; i < 256; i++)
            {
                LogTable256[i] = 1 + LogTable256[i / 2];
            }
            LogTable256[0] = -1; // if you want log(0) to return -1

            // Prepare the reverse bits table.
            prep_reverse_bits();
        }

        public static int FloorLog2(uint v)
        {
            if (!preped)
            {
                prep();
                preped = true;
            }
            int r; // r will be lg(v)
            uint tt; // temporaries

            if ((tt = v >> 24) != 0)
            {
                r = (24 + LogTable256[tt]);
            }
            else if ((tt = v >> 16) != 0)
            {
                r = (16 + LogTable256[tt]);
            }
            else if ((tt = v >> 8) != 0)
            {
                r = (8 + LogTable256[tt]);
            }
            else
            {
                r = LogTable256[v];
            }
            return r;
        }

        public static long FloorLog2(ulong v)
        {
            if (!preped)
            {
                prep();
                preped = true;
            }
            long r; // r will be lg(v)
            ulong tt; // temporaries

            if ((tt = v >> 56) != 0)
            {
                r = (56 + LogTable256[tt]);
            }
            else if ((tt = v >> 48) != 0)
            {
                r = (48 + LogTable256[tt]);
            }
            else if ((tt = v >> 40) != 0)
            {
                r = (40 + LogTable256[tt]);
            }
            else if ((tt = v >> 32) != 0)
            {
                r = (32 + LogTable256[tt]);
            }
            else if ((tt = v >> 24) != 0)
            {
                r = (24 + LogTable256[tt]);
            }
            else if ((tt = v >> 16) != 0)
            {
                r = (16 + LogTable256[tt]);
            }
            else if ((tt = v >> 8) != 0)
            {
                r = (8 + LogTable256[tt]);
            }
            else
            {
                r = LogTable256[v];
            }
            return r;
        }

        public static int CeilingLog2(uint v)
        {
            int r = Bithacks.FloorLog2(v);
            if (r < 0)
                return r;
            if (v != (uint)Bithacks.Power2((uint)r))
                return r + 1;
            else
                return r;
        }

        public static int Power2(uint v)
        {
            if (v == 0)
                return 1;
            else
                return (int)(2 << (int)(v - 1));
        }

        public static int Power2(int v)
        {
            if (v == 0)
                return 1;
            else
                return (int)(2 << (int)(v - 1));
        }

        static byte[] BitReverseTable256 = new byte[256];

        static void R2(ref int i, byte v)
        {
            BitReverseTable256[i++] = v;
            BitReverseTable256[i++] = (byte)(v + 2 * 64);
            BitReverseTable256[i++] = (byte)(v + 1 * 64);
            BitReverseTable256[i++] = (byte)(v + 3 * 64);
        }

        static void R4(ref int i, byte v)
        {
            R2(ref i, v);
            R2(ref i, (byte)(v + 2 * 16));
            R2(ref i, (byte)(v + 1 * 16));
            R2(ref i, (byte)(v + 3 * 16));
        }

        static void R6(ref int i, byte v)
        {
            R4(ref i, v);
            R4(ref i, (byte)(v + 2 * 4));
            R4(ref i, (byte)(v + 1 * 4));
            R4(ref i, (byte)(v + 3 * 4));
        }

        static void prep_reverse_bits()
        {
            int i = 0;
            R6(ref i, 0);
            R6(ref i, 2);
            R6(ref i, 1);
            R6(ref i, 3);
        }

        public static byte ReverseBits(byte from)
        {
            if (!preped)
            {
                prep();
                preped = true;
            }
            return BitReverseTable256[from];
        }

        public static Int32 ReverseBits(Int32 from)
        {
            if (!preped)
            {
                prep();
                preped = true;
            }
            Int32 result = 0;
            for (int i = 0; i < sizeof(Int32); ++i)
            {
                result = result << 8;
                result |= BitReverseTable256[(byte)(from & 0xff)];
                from = from >> 8;
            }
            return result;
        }

        public static UInt32 ReverseBits(UInt32 from)
        {
            if (!preped)
            {
                prep();
                preped = true;
            }
            UInt32 result = 0;
            for (int i = 0; i < sizeof(UInt32); ++i)
            {
                result = result << 8;
                result |= BitReverseTable256[(byte)(from & 0xff)];
                from = from >> 8;
            }
            return result;
        }

        static int Ones(uint x)
        {
            // 32-bit recursive reduction using SWAR...  but first step is mapping 2-bit values
            // into sum of 2 1-bit values in sneaky way
            x -= ((x >> 1) & 0x55555555);
            x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
            x = (((x >> 4) + x) & 0x0f0f0f0f);
            x += (x >> 8);
            x += (x >> 16);
            return (int)(x & 0x0000003f);
        }

        public static int xFloorLog2(uint x)
        {
            x |= (x >> 1);
            x |= (x >> 2);
            x |= (x >> 4);
            x |= (x >> 8);
            x |= (x >> 16);
            return (Bithacks.Ones(x) - 1);
        }

        public static int Log2(uint x)
        {
            return FloorLog2(x);
        }

        public static int Log2(int x)
        {
            return FloorLog2((uint)x);
        }
    }
}
