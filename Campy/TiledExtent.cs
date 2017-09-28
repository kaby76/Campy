namespace Campy.Types
{
    public class TiledExtent : Extent
    {
        internal int _Tile_Rank;
        internal int[] _Tile_Dim;

        public int[] Tile_Dims
        {
            get
            {
                return this._Tile_Dim;
            }
        }

        public int tile_Rank
        {
            get
            {
                return this._Tile_Rank;
            }
        }

        public TiledExtent(int[] _Array, Extent e)
            : base(e)
        {
        }

        public TiledExtent(int _I0, int _I1, int _I2, Extent e)
            : base(e)
        {
            this._Tile_Rank = 3;
        }

        public TiledExtent(int _I0, int _I1, Extent e)
            : base(e)
        {
            this._Tile_Rank = 2;
        }

        public TiledExtent(int _I0, Extent e)
            : base(e)
        {
            this._Tile_Rank = 1;
        }

        public TiledExtent(Extent e)
            : base(e)
        {
            this._Tile_Rank = 1;
        }
    }
}
