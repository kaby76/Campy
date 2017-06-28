using System;
using System.Collections.Generic;
using System.Text;

namespace Campy.Types
{
    public class TiledIndex
    {
        internal int _Rank;
        public Index Local;
        public Index Global;
        public Index Tile;
        public Index Tile_Origin;
        public TileBarrier Barrier;

        public TiledIndex()
        {
            this._Rank = 1;
            this.Local = new Index();
            this.Global = new Index();
            this.Tile = new Index();
            this.Tile_Origin = new Index();
            this.Barrier = new TileBarrier();
        }
    }
}
