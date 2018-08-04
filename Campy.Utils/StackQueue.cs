using System.Collections.Generic;

namespace Campy.Utils
{
    /// <summary>
    /// StackQueue - a data structure that has both Stack and Queue interfaces.
    /// </summary>
    /// <typeparam name="NAME"></typeparam>
    public class StackQueue<T>
    {
        private int _size;
        private int _top;

        // Underlying the datatype, we need an array that resizes without changing it's
        // reference. A straight array cannot work.
        private List<T> _items;

        public StackQueue()
        {
            _size = 10;
            _top = 0;
            _items = new List<T>(_size);
            for (int i = 0; i < _size; ++i) _items.Add(default(T));
        }

        public StackQueue(T value)
        {
            _size = 10;
            _top = 0;
            _items = new List<T>(_size);
            for (int i = 0; i < _size; ++i) _items.Add(default(T));
            _items[_top++] = value;
        }

        public StackQueue(StackQueue<T> other)
        {
            _size = other._size;
            _top = other._top;
            _items = new List<T>(_size);
            for (int i = 0; i < _size; ++i) _items.Add(default(T));
            _items.AddRange(other._items);
        }

        public virtual int Size()
        {
            return _top;
        }

        public virtual int Count
        {
            get { return _top; }
        }

        public virtual T Pop()
        {
            if (_top >= _size)
            {
                int old = _size;
                _size *= 2;
                _items.Capacity = _size;
                for (int i = old; i < _size; ++i) _items.Add(default(T));
            }
            if (_top > 0)
            {
                int index = _top - 1;
                T cur = _items[index];
                _items[index] = default(T);
                _top = _top - 1;
                return cur;
            }
            else
            {
                return default(T);
            }
        }

        public virtual T this[int n]
        {
            get
            {
                return PeekBottom(n);
            }
            set
            {
                _items[n] = value;
            }
        }

        public virtual T PeekTop(int n = 0)
        {
            if (_top >= _size)
            {
                int old = _size;
                _size *= 2;
                _items.Capacity = _size;
                for (int i = old; i < _size; ++i) _items.Add(default(T));
            }
            if (_top > 0)
            {
                int index = _top - 1;
                T cur = _items[index - n];
                return cur;
            }
            else
            {
                return default(T);
            }
        }

        public virtual T PeekBottom(int n)
        {
            if (_top >= _size)
            {
                int old = _size;
                _size *= 2;
                _items.Capacity = _size;
                for (int i = old; i < _size; ++i) _items.Add(default(T));
            }
            if (n >= _top)
                return default(T);
            T cur = _items[n];
            return cur;
        }

        public virtual void Push(T value)
        {
            if (_top >= _size)
            {
                int old = _size;
                _size *= 2;
                _items.Capacity = _size;
                for (int i = old; i < _size; ++i) _items.Add(default(T));
            }
            _items[_top++] = value;
        }

        public virtual void Push(IEnumerable<T> collection)
        {
            foreach (T t in collection)
            {
                if (_top >= _size)
                {
                    int old = _size;
                    _size *= 2;
                    _items.Capacity = _size;
                    for (int i = old; i < _size; ++i) _items.Add(default(T));
                }
                _items[_top++] = t;
            }
        }

        public virtual void PushMultiple(params T[] values)
        {
            int count = values.Length;
            for (int i = 0; i < count; i++)
            {
                Push(values[i]);
            }
        }

        public virtual void EnqueueTop(T value)
        {
            // Same as "Push(value)".
            Push(value);
        }

        public virtual void EnqueueBottom(T value)
        {
            if (_top >= _size)
            {
                _size *= 2;
                _items.Capacity = _size;
            }
            // "Push" a value on the bottom of the stack.
            for (int i = _top - 1; i >= 0; --i)
                _items[i + 1] = _items[i];
            _items[0] = value;
            ++_top;
        }

        public virtual T DequeueTop()
        {
            // Same as "Pop()".
            return Pop();
        }

        public virtual T DequeueBottom()
        {
            if (_top >= _size)
            {
                _size *= 2;
                _items.Capacity = _size;
            }
            // Remove item from bottom of stack.
            if (_top > 0)
            {
                T cur = _items[0];
                for (int i = 1; i <= _top; ++i)
                    _items[i - 1] = _items[i];
                _top--;
                return cur;
            }
            else
            {
                return default(T);
            }
        }

        public virtual bool Contains(T item)
        {
            return _items.Contains(item);
        }

        public virtual System.Collections.Generic.IEnumerator<T> GetEnumerator()
        {
            for (int i = _top - 1; i >= 0; i--)
            {
                yield return _items[i];
            }
        }

        public virtual ListSection<T> Section(int start, int length)
        {
            ListSection<T> result = new ListSection<T>(_items, start, length);
            return result;
        }

        public virtual ListSection<T> Section(int length)
        {
            ListSection<T> result = new ListSection<T>(_items, this._top, length);
            return result;
        }
    }
}
