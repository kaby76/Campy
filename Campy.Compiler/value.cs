using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Campy.Utils;
using Swigged.LLVM;

namespace Campy.Compiler
{
    public class VALUE : IComparable
    {
        private ValueRef _value_ref;

        public ValueRef V
        {
            get { return _value_ref; }
            set { _value_ref = value; }
        }

        private TYPE _type;

        public TYPE T
        {
            get { return _type; }
            set { _type = value; }
        }

        public VALUE(ValueRef v)
        {
            _value_ref = v;
            TypeRef t = LLVM.TypeOf(v);
            _type = new TYPE(t);
        }

        public VALUE(ValueRef v, TYPE t)
        {
            _value_ref = v;
            TypeRef tt = LLVM.TypeOf(v);
            // Fails. Debug.Assert(t.IntermediateType == tt);
            _type = t;
        }
        private VALUE(ValueRef v, TypeRef t)
        {
            _value_ref = v;
            _type = new TYPE(t);
        }


        public override string ToString()
        {
            try
            {
                string a = LLVM.PrintValueToString(_value_ref);
                string b = LLVM.PrintTypeToString(LLVM.TypeOf(_value_ref));
                string c = LLVM.PrintTypeToString(_type.IntermediateTypeLLVM);
                return a + ":" + b + " " + c;
            }
            catch
            {
                return "crash!";
            }
        }

        public virtual int CompareTo(object obj)
        {
            throw new ArgumentException("Unimplemented in derived type.");
        }

        public override int GetHashCode()
        {
            throw new ArgumentException("Unimplemented in derived type.");
        }

        public override bool Equals(object obj)
        {
            throw new ArgumentException("Unimplemented in derived type.");
        }

        public static new bool Equals(object obj1, object obj2) /* override */
        {
            throw new ArgumentException("Unimplemented in derived type.");
        }
    }


}
