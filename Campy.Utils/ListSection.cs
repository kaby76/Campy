using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Campy.Utils
{
    /// <summary>
    /// A data type to encapsulate sections of a list, acting itself as a list. The reason is so you do not need
    /// to keep track of offsets into lists.
    /// NB: At the moment, DO NOT MODIFY THE LIST! There is no adjustment of offsets!
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ListSection<T>
    {
        List<T> _list;
        int _base;
        int _len;

        public List<T> List
        {
            get
            {
                return _list;
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

        public ListSection(List<T> list, int b, int l)
        {
            _list = list;
            _base = b;
            _len = l;
        }

        public T this[int i]
        {
            get
            {
                return _list[_base + i];
            }
            set
            {
                _list[_base + i] = value;
            }
        }

        static void Resize(ref ListSection<T> arr, int new_length)
        {
        }
    }
}
