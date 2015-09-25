#include "core/core.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemory

DataMemory::DataMemory(int type, Allocator alloc, Deallocator dealloc)
: _type(type), alloc(std::move(alloc)), dealloc(std::move(dealloc))
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemoryNone

DataMemoryNone::DataMemoryNone()
    : DataMemory(None,
    [](int row_size, int height, int &stride, size_t &alignment)
{
    return DataNullptr;
},
[](DataType *memory)
{})
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemoryCPU

DataMemoryCPU::DataMemoryCPU()
    : DataMemory(CPU,
    [](int row_size, int height, int &stride, size_t &alignment)
{
    size_t size = height * static_cast<size_t>(Abs(stride));

    if (size > 0)
    {
        return reinterpret_cast<DataType *>(AlignedMalloc(size, Max(MEMORY_ALIGNMENT, alignment)));
    }
    else
    {
        return DataNullptr;
    }
},
[](DataType *memory)
{
    if (memory)
    {
        AlignedFree(memory);
    }
})
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemoryCPU_Continuous

DataMemoryCPU_Continuous::DataMemoryCPU_Continuous()
    : DataMemory(CPU_Continuous,
    [](int row_size, int height, int &stride, size_t &alignment)
{
    stride = stride < 0 ? -row_size : row_size;
    alignment = ValidAlignment(alignment, stride);
    size_t size = height * static_cast<size_t>(Abs(stride));

    if (size > 0)
    {
        return reinterpret_cast<DataType *>(AlignedMalloc(size, Max(MEMORY_ALIGNMENT, alignment)));
    }
    else
    {
        return DataNullptr;
    }
},
[](DataType *memory)
{
    if (memory)
    {
        AlignedFree(memory);
    }
})
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Preset smart pointers to DataMemory

const DataMemoryPtr dmNone = MakeDataMemoryPtr<DataMemoryNone>();
const DataMemoryPtr dmCPU = MakeDataMemoryPtr<DataMemoryCPU>();
const DataMemoryPtr dmCPU_Continuous = MakeDataMemoryPtr<DataMemoryCPU_Continuous>();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
