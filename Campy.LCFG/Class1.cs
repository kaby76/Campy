using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.LCFG
{
    public enum ReaderAlignType
    {
        Reader_AlignNatural = ~0, ///< Default natural alignment
        Reader_AlignUnknown = 0,           ///< Unknown alignment
        Reader_Align1 = 1,                 ///< Byte alignment
        Reader_Align2 = 2,                 ///< Word alignment
        Reader_Align4 = 4,                 ///< DWord alignment
        Reader_Align8 = 8                  ///< QWord alignment
    };

}
