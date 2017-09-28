namespace Campy.Types
{
    public class Extent
    {
        internal static Extent default_value = new Extent();
        internal int[] _M_base;
        public int _Rank;

        public static Extent Default_Value
        {
            get
            {
                return Extent.default_value;
            }
        }

        public Extent(int[] _Array)
        {
        }

        public Extent(int _I0, int _I1, int _I2)
        {
            _Rank = 3;
            _M_base = new int[] { _I0, _I1, _I2 };
        }

        public Extent(int _I0, int _I1)
        {
            _Rank = 2;
            _M_base = new int[] { _I0, _I1 };
        }

        public Extent(int _I0)
        {
            _Rank = 1;
            _M_base = new int[] {_I0};
        }

        public Extent(Extent e)
        {
        }

        public Extent()
        {
        }

        public static Extent operator +(Extent _Lhs, Index _Rhs)
        {
            Extent extent = new Extent();
            extent._Rank = _Rhs._Rank;
            for (int index = 0; index < _Rhs._Rank; ++index)
                extent._M_base[index] = _Lhs._M_base[index] + _Rhs._M_base[index];
            return extent;
        }

        public static Extent operator ++(Extent _Lhs)
        {
            for (int index = 0; index < _Lhs._Rank; ++index)
                _Lhs._M_base[index] = _Lhs._M_base[index] + 1;
            return _Lhs;
        }

        public static Extent operator -(Extent _Lhs, Index _Rhs)
        {
            Extent extent = new Extent();
            extent._Rank = _Rhs._Rank;
            for (int index = 0; index < _Rhs._Rank; ++index)
                extent._M_base[index] = _Lhs._M_base[index] - _Rhs._M_base[index];
            return extent;
        }

        public static Extent operator --(Extent _Lhs)
        {
            for (int index = 0; index < _Lhs._Rank; ++index)
                _Lhs._M_base[index] = _Lhs._M_base[index] - 1;
            return _Lhs;
        }

        public TiledExtent Tile(int _I0, int _I1, int _I2)
        {
            return new TiledExtent(_I0, _I1, _I2, this);
        }

        public TiledExtent Tile(int _I0, int _I1)
        {
            return new TiledExtent(_I0, _I1, this);
        }

        public TiledExtent Tile(int _I0)
        {
            return new TiledExtent(_I0, this);
        }

        public int Size()
        {
            int num = 1;
            for (int index = 0; index < this._Rank; ++index)
                num *= this._M_base[index];
            return num;
        }
    }
}
