using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.Utils
{
    /// <summary>
    /// A data type to encapsulate sections of an array, acting as an array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ArraySection<T>
    {
        List<T> _arr;
        int _base;
        int _len;

        public List<T> Arr
        {
            get
            {
                return _arr;
            }
        }

        public int Base
        {
            get
            {
                return _base;
            }
        }

        public int Len
        {
            get
            {
                return _len;
            }
        }

        public ArraySection(List<T> arr, int b, int l)
        {
            _arr = arr;
            _base = b;
            _len = l;
        }

        public T this[int i]
        {
            get
            {
                return _arr[_base + i];
            }
            set
            {
                _arr[_base + i] = value;
            }
        }

        static void Resize(ref ArraySection<T> arr, int new_length)
        {
        }
    }
}
