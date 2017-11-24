using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Swigged.Cuda;
using Swigged.LLVM;
using Campy.Utils;
using Mono.Cecil;
using MethodImplAttributes = Mono.Cecil.MethodImplAttributes;

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

            public long l; // Width of dimension 0.
            // Additional widths for dimension 1, dimension 2, ...
            // Value data after all dimensional sizes.
        }

        public static unsafe int get_length_multi_array(A* arr, int i0)
        {
            byte* bp = (byte*) arr;
            bp = bp + 16 + 8 * i0;
            long* lp = (long*) bp;
            return (int) *lp;
        }

        public static unsafe int get_multi_array(A* arr, int i0)
        {
            int* a = *(int**) arr;
            return *(a + i0);
        }

        public static unsafe int get_multi_array(A* arr, int i0, int i1)
        {
            // (y * xMax) + x
            int* a = (int*) (*arr).p;
            int d = (int) (*arr).d;
            byte* d0_ptr = (byte*) arr;
            d0_ptr = d0_ptr + 24;
            long o = 0;
            long d0 = *(long*) d0_ptr;
            o = i0 * d0 + i1;
            return *(a + o);
        }

        public static unsafe int get_multi_array(A* arr, int i0, int i1, int i2)
        {
            // (z * xMax * yMax) + (y * xMax) + x;
            int* a = (int*) (*arr).p;
            int d = (int) (*arr).d;
            byte* bp_d0 = (byte*) arr;
            byte* bp_d1 = (byte*) arr;
            bp_d1 = bp_d1 + 24;
            long o = 0;
            long* lp_d1 = (long*) bp_d1;
            byte* bp_d2 = bp_d1 + 8;
            long* lp_d2 = (long*) bp_d2;
            o = (*lp_d1) * i0 + i1;
            return *(a + o);
        }

        public static unsafe void set_multi_array(A* arr, int i0, int value)
        {
            int* a = (int*) (*arr).p;
            int d = (int) (*arr).d;
            long o = i0;
            *(a + o) = value;
        }

        public static unsafe void set_multi_array(A* arr, int i0, int i1, int value)
        {
            //  b[i, j] = j  + i * ex[1];

            int* a = (int*) (*arr).p;
            long ex1 = *(long*) (24 + (byte*) arr);
            long o = i1 + ex1 * i0;
            *(a + o) = value;
        }

        public static unsafe void set_multi_array(A* arr, int i0, int i1, int i2, int value)
        {
            //  b[i, j, k] = k + j * ex[2] + i * ex[2] * ex[1];

            int* a = (int*) (*arr).p;
            long ex1 = *(long*) (24 + (byte*) arr);
            long ex2 = *(long*) (32 + (byte*) arr);
            long o = i2 + i1 * ex2 + i0 * ex2 * ex1;
            *(a + o) = value;
        }

        public static void ThrowArgumentOutOfRangeException()
        {
        }


        public static void ParseBCL(Converter converter)
        {
            var assembly = Assembly.GetAssembly(typeof(Campy.Compiler.Runtime));
            var resource_names = assembly.GetManifestResourceNames();
            foreach (var resource_name in resource_names)
            {
                using (Stream stream = assembly.GetManifestResourceStream(resource_name))
                using (StreamReader reader = new StreamReader(stream))
                {
                    string gpu_bcl_ptx = reader.ReadToEnd();
                    // Parse the PTX for ".visible" functions, and enter each in
                    // the runtime table.
                    string[] lines = gpu_bcl_ptx.Split(new char[] {'\r', '\n'}, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        Regex regex = new Regex(@"\.visible.*[ ](?<name>\w+)\($");
                        Match m = regex.Match(line);
                        if (m.Success)
                        {
                            var mangled_name = m.Groups["name"].Value;
                            var demangled_name = mangled_name;
                            Converter.built_in_functions.Add(demangled_name,
                                LLVM.AddFunction(
                                    Converter.global_llvm_module,
                                    mangled_name,
                                    LLVM.FunctionType(LLVM.Int64Type(),
                                        new TypeRef[]
                                        {
                                            LLVM.PointerType(LLVM.VoidType(), 0), // "this"
                                            LLVM.PointerType(LLVM.VoidType(), 0), // params in a block.
                                            LLVM.PointerType(LLVM.VoidType(), 0) // return value block.
                                        }, false)));
                        }
                    }
                }
            }

            CallInfo.Initialize();
        }
    }

    public class tInternalCall
    {
        public string _nameSpace;
        public string _type;
        public string _method;
        public string _fn;
        public Mono.Cecil.TypeReference _returnType;
        public List<Mono.Cecil.ParameterDefinition> _parameterTypes;

        public tInternalCall(string ns, string type, string method, string fn,
            Mono.Cecil.TypeReference returntype, List<ParameterDefinition> parametertypes)
        {
            _nameSpace = ns;
            _type = type;
            _method = method;
            _fn = fn;
            _returnType = returntype;
            _parameterTypes = parametertypes;
        }
    }

    // This table encodes runtime type information for internal calls in the native portion of
    // the BCL for the GPU. It was originally encoded in dna/internal.c. However, it's easier and
    // safer to derive the information from the C# portion of the BCL using System.Reflection.
    //
    // Why is this information needed? In Inst.c, I need to make a call of a function to the runtime.
    // I only have PTX files, which removes the type information from the signature of
    // the original call (it is all three parameters of void*).
    public class CallInfo : IEnumerable<tInternalCall>
    {
        public static void Add(string ns, string type, string method, string fn,
            TypeReference returntype, List<Mono.Cecil.ParameterDefinition> parametertypes)
        {
            internalCalls.Add(new tInternalCall(ns, type, method, fn, returntype, parametertypes));
        }

        public static List<tInternalCall> internalCalls = new List<tInternalCall>();

        public IEnumerator<tInternalCall> GetEnumerator()
        {
            return internalCalls.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public static void Initialize()
        {
            // Load C# library for BCL.
            string yopath = @"C:\Users\kenne\Documents\Campy\Campy.Runtime\Corlib\bin\Debug\netstandard1.3\corlib.dll";
            Mono.Cecil.ModuleDefinition md = Mono.Cecil.ModuleDefinition.ReadModule(yopath);
            foreach (var t in md.GetTypes())
            {
                System.Console.WriteLine(t.FullName);
                foreach (var m in t.Methods)
                {
                    var x = m.ImplAttributes;
                    if ((x & MethodImplAttributes.InternalCall) != 0)
                    {
                        Add(t.Namespace, t.FullName, m.Name, m.FullName, m.ReturnType, m.Parameters.ToList());
                    }
                }
            }
        }
    }
}
