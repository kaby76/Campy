#pragma once
using namespace System;
using namespace System::Runtime::InteropServices;
namespace Campy {
	public ref class Array_Base
	{
	public:
		Array_Base();

		void *operator new(size_t len)
		{
			void *ptr;
			//cudaMallocManaged(&ptr, len);
			//cudaDeviceSynchronize();
			return ptr;
		}

		void operator delete(void *ptr) {
			//cudaDeviceSynchronize();
			//cudaFree(ptr);
		}
	};
}
