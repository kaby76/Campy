using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.Compiler
{
    public enum ABIParameterInfoKind
    {
        Direct = 0,
        Indirect = 1,
        Coerced = 2,
    }
}
