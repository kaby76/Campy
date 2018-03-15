using System;
using System.Linq;
using Xunit;
using Campy;

namespace CombSort
{
    public class CombSort
    {
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

        class CombSorter
        {
            public static void swap(ref int i, ref int j)
            {
                int t = i;
                i = j;
                j = t;
            }

            public static void Sort(int[] a)
            {
                // Delay and Synch should really work for specific data, not all.
                // For now, copy everything back to CPU.
                //Campy.Parallel.Delay();
                System.Console.WriteLine(String.Join(" ", a));
                int N = a.Length;
                int gap = N;
                bool swaps = true;
                float gap_factor = (float)1.25;
                while (gap > 1 || swaps)
                {
                    int local_gap = (int)(gap / gap_factor);
                    if (local_gap < 1) local_gap = 1;
                    gap = local_gap;
                    swaps = false;
                    Campy.KernelType de = i =>
                    {
                        if (a[i] > a[i + local_gap])
                        {
                            swap(ref a[i], ref a[i + local_gap]);
                            swaps = true;
                        }
                    };
                    System.Console.WriteLine("bswaps " + swaps);
                    System.Console.WriteLine(String.Join(" ", a));
                    System.Console.WriteLine("blocal_gap " + local_gap);
                    if (local_gap != 1) Campy.Parallel.For(N - local_gap, de);
                    else Campy.Sequential.For(N - local_gap, de);
                    System.Console.WriteLine("aswaps " + swaps);
                    System.Console.WriteLine("alocal_gap " + local_gap);
                    System.Console.WriteLine(String.Join(" ", a));
                }
                //Campy.Parallel.Synch();
            }
        }

        [Fact]
        public void CombSortT()
        {
            Random rnd = new Random();
            int N = 8;
            //int[] a = Enumerable.Range(0, N).ToArray().OrderBy(x => rnd.Next()).ToArray();
            int[] a = new int[]{ 1, 7, 3, 0, 6, 5, 4, 2 };
            CombSorter.Sort(a);
            for (int i = 0; i < N; ++i)
                if (a[i] != i)
                    throw new Exception();

        }
    }
}
