using System;
using System.Runtime.InteropServices;

namespace Campy.Types
{
    public class ArrayView<_Value_type>
    {
        public int Rank = 1;
        private Type ValueType;
        public ArrayView(Array src) { }
        public ArrayView(ArrayView<_Value_type> other) { }
        public ArrayView(Extent extent) { }
        public ArrayView(int l0)
        {
            Rank = 1;
        }
        public ArrayView(int l0, int l1)
        {
            Rank = 2;
        }
        public ArrayView(int l0, int l1, int l2)
        {
            Rank = 3;
        }
        public ArrayView(ref _Value_type[] data)
        {
            Rank = 1;
        }
        public ArrayView(ref _Value_type[][] data)
        {
            Rank = 2;
        }
        public ArrayView(ref _Value_type[][][] data)
        {
            Rank = 3;
        }
        public _Value_type this[int i]
        {
            get { return default(_Value_type); }
            set { }
        }
        public _Value_type this[int i, int j]
        {
            get { return default(_Value_type); }
            set { }
        }
        public _Value_type this[int i, int j, int k]
        {
            get { return default(_Value_type); }
            set { }
        }
        public void Refresh() { }
        public void Synchronize() { }
        public async void SynchronizeAsync() { }
        public void Discard_Data() { }
        public AcceleratorView GetSourceAcceleratorView()
        {
            return SourceAcceleratorView;
        }
        public AcceleratorView SourceAcceleratorView { get; private set; }
        public static ArrayView<_Value_type> Default_Value { get; internal set; }
        public Extent Extent { get; internal set; }
        public ArrayView(IntPtr data, int length) { }
        public ArrayView<_Value_type> Section(int _I0, int _E0)
        {
            return null;
        }
    }
}
