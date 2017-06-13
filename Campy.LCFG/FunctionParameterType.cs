using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.ControlFlowGraph
{
    public struct FunctionParameterType
    {
        public readonly Type Type;
        public readonly ABIParameterInfo ABIParameterInfo;

        public FunctionParameterType(IABI abi, Type type)
        {
            Type = type;
            ABIParameterInfo = abi.GetParameterInfo(type);
        }
    }
}
