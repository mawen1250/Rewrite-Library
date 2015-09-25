#include "core/core.h"
#include "cuda/host_helper.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemoryCUDA

DataMemoryCUDA::DataMemoryCUDA()
: DataMemory(CUDA,
    [](int row_size, int height, int &stride, size_t &alignment)
{
    void *devPtr;
    size_t pitch;

    checkCudaErrors(cudaMallocPitch(&devPtr, &pitch, row_size, height));

    stride = stride < 0 ? -static_cast<int>(pitch) : static_cast<int>(pitch);
    alignment = ValidAlignment(alignment, stride);
    return reinterpret_cast<DataType *>(devPtr);
},
[](DataType *memory)
{
    if (memory)
    {
        cudaFree(reinterpret_cast<void *>(memory));
    }
})
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemoryCUDA_Continuous

DataMemoryCUDA_Continuous::DataMemoryCUDA_Continuous()
    : DataMemory(CUDA_Continuous,
    [](int row_size, int height, int &stride, size_t &alignment)
{
    stride = stride < 0 ? -row_size : row_size;
    alignment = ValidAlignment(alignment, stride);
    size_t size = height * static_cast<size_t>(Abs(stride));
    void *devPtr;

    checkCudaErrors(cudaMalloc(&devPtr, size));

    return reinterpret_cast<DataType *>(devPtr);
},
[](DataType *memory)
{
    if (memory)
    {
        cudaFree(reinterpret_cast<void *>(memory));
    }
})
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Preset smart pointers to DataMemory

const DataMemoryPtr dmCUDA = MakeDataMemoryPtr<DataMemoryCUDA>();
const DataMemoryPtr dmCUDA_Continuous = MakeDataMemoryPtr<DataMemoryCUDA_Continuous>();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
