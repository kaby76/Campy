using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.Compiler
{
    /// <summary>
    /// Defines an ABI (how to pass parameters to function calls).
    /// </summary>
    public interface IABI
    {
        /// <summary>
        /// Describe how a function parameter of given type is to be passed.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns></returns>
        ABIParameterInfo GetParameterInfo(Type type);
    }
}
