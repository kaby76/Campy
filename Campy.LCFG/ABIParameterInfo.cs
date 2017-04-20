using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Swigged.LLVM;

namespace Campy.LCFG
{
    public struct ABIParameterInfo
    {
        public readonly ABIParameterInfoKind Kind;
        public readonly TypeRef CoerceType;

        public ABIParameterInfo(ABIParameterInfoKind kind)
        {
            Kind = kind;
            CoerceType = default(TypeRef);
        }

        public ABIParameterInfo(ABIParameterInfoKind kind, TypeRef coerceType)
        {
            if (kind != ABIParameterInfoKind.Coerced)
                throw new ArgumentException("kind");

            Kind = kind;
            CoerceType = coerceType;
        }
    }
}
