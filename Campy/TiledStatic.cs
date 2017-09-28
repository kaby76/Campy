using System;

namespace Campy.Types
{

    public class TileStatic<_Value_type>
    {
        private int _Rank = 1;
        private int _length;
        private Extent _extent;

        public int Length
        {
            get
            {
                return this._length;
            }
        }

        public _Value_type this[int i]
        {
            get
            {
                Type type = (Type)null;
                type = typeof(_Value_type);
                return default(_Value_type);
            }
            set
            {
            }
        }

        public TileStatic(int length)
        {
            this._length = length;
        }
    }
}
