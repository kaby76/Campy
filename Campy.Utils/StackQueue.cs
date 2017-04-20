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
        private T[] _items;

        public StackQueue()
        {
            _size = 10;
            _top = 0;
            _items = new T[_size];
        }

        public StackQueue(T value)
        {
            _size = 10;
            _top = 0;
            _items = new T[_size];
            _items[_top++] = value;
        }

        public StackQueue(StackQueue<T> other)
        {
            _size = other._size;
            _top = other._top;
            _items = new T[_size];
            System.Array.Copy(other._items, _items, _size);
        }

        public int Size()
        {
            return _top;
        }

        public int Count
        {
            get { return _top; }
        }

        public T Pop()
        {
            if (_top >= _size)
            {
                _size *= 2;
                System.Array.Resize(ref _items, _size);
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

        public T this[int n]
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

        public T PeekTop(int n = 0)
        {
            if (_top >= _size)
            {
                _size *= 2;
                System.Array.Resize(ref _items, _size);
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

        public T PeekBottom(int n)
        {
            if (_top >= _size)
            {
                _size *= 2;
                System.Array.Resize(ref _items, _size);
            }
            if (n >= _top)
                return default(T);
            T cur = _items[n];
            return cur;
        }

        public void Push(T value)
        {
            if (_top >= _size)
            {
                _size *= 2;
                System.Array.Resize(ref _items, _size);
            }
            _items[_top++] = value;
        }

        public void Push(IEnumerable<T> collection)
        {
            foreach (T t in collection)
            {
                if (_top >= _size)
                {
                    _size *= 2;
                    System.Array.Resize(ref _items, _size);
                }
                _items[_top++] = t;
            }
        }

        public void PushMultiple(params T[] values)
        {
            int count = values.Length;
            for (int i = 0; i < count; i++)
            {
                Push(values[i]);
            }
        }

        public void EnqueueTop(T value)
        {
            // Same as "Push(value)".
            Push(value);
        }

        public void EnqueueBottom(T value)
        {
            if (_top >= _size)
            {
                _size *= 2;
                System.Array.Resize(ref _items, _size);
            }
            // "Push" a value on the bottom of the stack.
            for (int i = _top - 1; i >= 0; --i)
                _items[i + 1] = _items[i];
            _items[0] = value;
            ++_top;
        }

        public T DequeueTop()
        {
            // Same as "Pop()".
            return Pop();
        }

        public T DequeueBottom()
        {
            if (_top >= _size)
            {
                _size *= 2;
                System.Array.Resize(ref _items, _size);
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

        public bool Contains(T item)
        {
            return System.Array.FindIndex(_items, 0, _top, (T t) => t.Equals(item)) >= 0;
        }

        public System.Collections.Generic.IEnumerator<T> GetEnumerator()
        {
            for (int i = _top - 1; i >= 0; i--)
            {
                yield return _items[i];
            }
        }

        public ArraySection<T> Section(int start, int length)
        {
            ArraySection<T> result = new ArraySection<T>(_items, start, length);
            return result;
        }

        public ArraySection<T> Section(int length)
        {
            ArraySection<T> result = new ArraySection<T>(_items, this._top, length);
            return result;
        }
    }
}
