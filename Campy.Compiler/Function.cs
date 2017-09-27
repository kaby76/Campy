using System.Linq;
using Mono.Cecil;
using Swigged.LLVM;

namespace Campy.Compiler
{
    public class Function
    {
        private ValueRef _value_ref;

        public ValueRef V
        {
            get { return _value_ref; }
        }

        public Function(ValueRef func)
        {
            _value_ref = func;
        }

        public int VirtualSlot { get; set; }

        public ValueRef InterfaceSlot { get; set; }
    }
}
