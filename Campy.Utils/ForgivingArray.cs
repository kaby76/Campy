using System;

namespace Campy.Utils
{
    /// <summary>
    /// ForgivingArray is a class that automatically expands in size if you try to index an "out of bounds" element.
    /// So, no exception is thrown.
    /// </summary>
    /// <typeparam name="V"></typeparam>
    public class ForgivingArray<V>
    {
        int _size;
        V[] _array;

        public ForgivingArray()
        {
            _size = 4;
            _array = new V[_size];
        }

        public V this[int i]
        {
            get
            {
                if (i >= this._size)
                {
                    this._size *= 2;
                    Array.Resize(ref this._array, this._size);
                }
                return this._array[i];
            }
            set
            {
                if (i >= this._size)
                {
                    this._size *= 2;
                    Array.Resize(ref this._array, this._size);
                }
                this._array[i] = value;
            }
        }
    }
}
