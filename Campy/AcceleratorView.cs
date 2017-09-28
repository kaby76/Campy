using System;

namespace Campy.Types
{
    public class AcceleratorView
    {
        public AcceleratorView(AcceleratorView other)
        {
        }
        public Accelerator GetAccelerator()
        {
            return Accelerator;
        }
        internal AcceleratorView(Accelerator acc)
        {
            Accelerator = acc;
        }
        public Accelerator Accelerator { get; internal set; }
        public bool GetIsDebug()
        {
            return IsDebug;
        }
        public bool IsDebug { get; internal set; }
        public uint GetVersion()
        {
            return Version;
        }
        public uint Version { get; internal set; }
        public QueuingMode GetQueuingMode()
        {
            return QueuingMode;
        }
        public QueuingMode QueuingMode { get; internal set; }
        public bool GetIsAutoSelection()
        {
            return IsAutoSelection;
        }
        public bool IsAutoSelection { get; internal set; }
        public void Wait()
        {
        }
        public void Flush()
        {
        }

        private AcceleratorView() { }
    }
}
