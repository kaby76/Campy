namespace Campy
{
    using System.Collections.Generic;

    internal class Accelerator
    {
        public static string DefaultAccelerator { get; internal set; }
        public static string CpuAccelerator { get; internal set; }

        public Accelerator()
        {
        }

        public Accelerator(string path)
        {
            Init(path);
        }

        public Accelerator(Accelerator other)
        {
        }

        public static List<Accelerator> GetAll()
        {
            var acceleratorList = AcceleratorCuda.AllCudaDevices();
            return acceleratorList;
        }

        public static bool SetDefault(string path)
        {
            var acc = new Accelerator(path);
            return SetDefaultAccelerator(acc);
        }

        public string GetDevicePath { get; internal set; }
        public string DevicePath { get; internal set; }

        public uint GetVersion()
        {
            return Version;
        }

        public uint Version { get; internal set; }

        public string GetDescription()
        {
            return Description;
        }

        public string Description { get; internal set; }

        public bool GetIsDebug()
        {
            return IsDebug;
        }

        public bool IsDebug { get; internal set; }

        public bool GetIsEmulated()
        {
            return IsEmulated;
        }

        public bool IsEmulated { get; internal set; }

        public bool GetHasDisplay()
        {
            return HasDisplay;
        }

        public bool HasDisplay { get; internal set; }

        public bool GetSupportsDoublePrecision()
        {
            return SupportsDoublePrecision;
        }

        public bool SupportsDoublePrecision { get; internal set; }

        public bool GetSupportsLimitedDoublePrecision()
        {
            return SupportsLimitedDoublePrecision;
        }

        public bool SupportsLimitedDoublePrecision { get; internal set; }

        public bool GetSupportsCpuSharedMemory()
        {
            return SupportsCpuSharedMemory;
        }

        public bool SupportsCpuSharedMemory { get; internal set; }

        public ulong GetDedicatedMemory()
        {
            return DedicatedMemory;
        }

        public ulong DedicatedMemory { get; internal set; }

        public AccessType GetDefaultCpuAccessType()
        {
            return DefaultCpuAccessType;
        }

        public AccessType DefaultCpuAccessType { get; internal set; }

        public bool SetDefaultCpuAccessType(Accelerator the_default)
        {
            return false;
        }

        private void Init(string path)
        {
        }

        private static bool SetDefaultAccelerator(Accelerator acc)
        {
            return false;
        }
}

    public class AccessType
    {
    }
}
