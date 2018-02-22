namespace Campy
{
    //public class Index
    //{
    //    internal int _Rank;
    //    internal int[] _M_base;
    //    internal static Index default_value = new Index();

    //    public static Index Default_Value
    //    {
    //        get
    //        {
    //            return Index.default_value;
    //        }
    //    }

    //    public int this[int i]
    //    {
    //        get
    //        {
    //            return this._M_base[i];
    //        }
    //    }

    //    public int Rank
    //    {
    //        get
    //        {
    //            return this._Rank;
    //        }
    //    }

    //    public Index(int _I0, int _I1, int _I2)
    //    {
    //        this._Rank = 3;
    //        Index index = this;
    //        int[] numArray = new int[index._Rank];
    //        index._M_base = numArray;
    //        this._M_base[0] = _I0;
    //        this._M_base[1] = _I1;
    //        this._M_base[2] = _I2;
    //    }

    //    public Index(int _I0, int _I1)
    //    {
    //        this._Rank = 2;
    //        Index index = this;
    //        int[] numArray = new int[index._Rank];
    //        index._M_base = numArray;
    //        this._M_base[0] = _I0;
    //        this._M_base[1] = _I1;
    //    }

    //    public Index(int _I)
    //    {
    //        this._Rank = 1;
    //        Index index = this;
    //        int[] numArray = new int[index._Rank];
    //        index._M_base = numArray;
    //        this._M_base[0] = _I;
    //    }

    //    public Index(Extent extent)
    //    {
    //        this._Rank = extent._Rank;
    //        Index index = this;
    //        int[] numArray = new int[index._Rank];
    //        index._M_base = numArray;
    //        for (int i = 0; i < _Rank; ++i)
    //            this._M_base[i] = extent[i];
    //    }

    //    public Index()
    //    {
    //        this._Rank = 1;
    //        Index index = this;
    //        int[] numArray = new int[index._Rank];
    //        index._M_base = numArray;
    //    }

    //    public static implicit operator int(Index idx)
    //    {
    //        // compiler must generate special code for GPU.
    //        return idx._M_base[0];
    //    }
    //}
}
