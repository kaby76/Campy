using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Swigged.LLVM;

namespace Campy.LCFG
{
    class StackValue
    {
        public StackValue(StackValueType stackType, Type type, ValueRef value)
        {
            StackType = stackType;
            Type = type;
            Value = value;
        }

        public StackValueType StackType { get; private set; }

        public ValueRef Value { get; private set; }

        public Type Type { get; private set; }
    }
}
