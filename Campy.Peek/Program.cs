using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Campy.Compiler;

namespace Peek
{
    class Program
    {
        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitTheBcl")]
        public static extern void InitTheBcl(System.IntPtr a1, long a2, long a25, int a3, System.IntPtr a4);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitFileSystem")]
        public static extern void InitFileSystem();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "GfsAddFile")]
        public static extern void GfsAddFile(System.IntPtr name, System.IntPtr file, long length, System.IntPtr result);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitializeBCL1")]
        public static extern void InitializeBCL1();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "InitializeBCL2")]
        public static extern void InitializeBCL2();

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclGetMeta")]
        public static extern System.IntPtr BclGetMeta([MarshalAs(UnmanagedType.LPStr)] string file_name);

        [global::System.Runtime.InteropServices.DllImport(@"campy-runtime-wrapper", EntryPoint = "BclPrintMeta")]
        public static extern void BclPrintMeta(System.IntPtr meta);


        static void Main(string[] args)
        {
            unsafe
            {
                JITER.InitCuda();
                BUFFERS buffers = new BUFFERS();
                int the_size = 536870912;
                IntPtr b = buffers.New(the_size);
                RUNTIME.BclPtr = b;
                RUNTIME.BclPtrSize = (ulong)the_size;
                int max_threads = 1;
                IntPtr b2 = buffers.New(sizeof(int*));
                InitTheBcl(b, the_size, 2 * 16777216, max_threads, b2);
            }

            unsafe
            {
                InitFileSystem();
                // Set up corlib.dll in file system.
                string full_path_assem = RUNTIME.FindCoreLib();
                string assem = Path.GetFileName(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                var corlib_bytes_handle_len = stream.Length;
                var corlib_bytes = new byte[corlib_bytes_handle_len];
                stream.Read(corlib_bytes, 0, (int) corlib_bytes_handle_len);
                var corlib_bytes_handle = GCHandle.Alloc(corlib_bytes, GCHandleType.Pinned);
                var corlib_bytes_intptr = corlib_bytes_handle.AddrOfPinnedObject();
                stream.Close();
                stream.Dispose();
                var ptrx = Marshal.StringToHGlobalAnsi(assem);
                BUFFERS buffers = new BUFFERS();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
                var pointer4 = buffers.New(sizeof(int));
                GfsAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);
                InitializeBCL1();
                InitializeBCL2();
            }

            // Open assembly and print contents.
            {
                string full_path_assem = args[0];
                string assem = Path.GetFileName(full_path_assem);
                Stream stream = new FileStream(full_path_assem, FileMode.Open, FileAccess.Read, FileShare.Read);
                var corlib_bytes_handle_len = stream.Length;
                var corlib_bytes = new byte[corlib_bytes_handle_len];
                stream.Read(corlib_bytes, 0, (int)corlib_bytes_handle_len);
                var corlib_bytes_handle = GCHandle.Alloc(corlib_bytes, GCHandleType.Pinned);
                var corlib_bytes_intptr = corlib_bytes_handle.AddrOfPinnedObject();
                stream.Close();
                stream.Dispose();
                var ptrx = Marshal.StringToHGlobalAnsi(assem);
                BUFFERS buffers = new BUFFERS();
                IntPtr pointer1 = buffers.New(assem.Length + 1);
                BUFFERS.Cp(pointer1, ptrx, assem.Length + 1);
                var pointer4 = buffers.New(sizeof(int));
                GfsAddFile(pointer1, corlib_bytes_intptr, corlib_bytes_handle_len, pointer4);
                var meta = BclGetMeta(assem);
                BclPrintMeta(meta);
            }
        }
    }
}
