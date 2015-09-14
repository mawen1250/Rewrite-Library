#ifndef RWL_CORE_MEMORY_H_
#define RWL_CORE_MEMORY_H_

#include "core/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Memory allocation

const size_t MEMORY_ALIGNMENT = 64;

inline void *AlignedMalloc(size_t Size, size_t Alignment = MEMORY_ALIGNMENT)
{
    void *Memory = NULLPTR;
#ifdef _WIN32
    Memory = _aligned_malloc(Size, Alignment);
#else
    if (posix_memalign(&Memory, Alignment, Size))
    {
        Memory = NULLPTR;
    }
#endif
    if (Memory == NULLPTR)
    {
        throw std::bad_alloc();
    }
    return Memory;
}

template < typename _Ty >
void AlignedMalloc(_Ty *&Memory, size_t Count, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(AlignedMalloc(Count * sizeof(_Ty), Alignment));
}


inline void AlignedFree(void **Memory)
{
#ifdef _WIN32
    _aligned_free(*Memory);
#else
    free(*Memory);
#endif
    *Memory = NULLPTR;
}

template < typename _Ty >
void AlignedFree(_Ty *&Memory)
{
    void *temp = reinterpret_cast<void *>(Memory);
    AlignedFree(&temp);
    Memory = reinterpret_cast<_Ty *>(temp);
}


inline void *AlignedRealloc(void *Memory, size_t NewSize, size_t Alignment = MEMORY_ALIGNMENT)
{
#ifdef _WIN32
    Memory = _aligned_realloc(Memory, NewSize, Alignment);
    if (Memory == NULLPTR)
    {
        throw std::bad_array_new_length();
    }
#else
    AlignedFree(&Memory);
    Memory = AlignedMalloc(NewSize, Alignment);
#endif
    return Memory;
}

template < typename _Ty >
void AlignedRealloc(_Ty *&Memory, size_t NewCount, size_t Alignment = MEMORY_ALIGNMENT)
{
    Memory = reinterpret_cast<_Ty *>(AlignedRealloc(Memory, NewCount * sizeof(_Ty), Alignment));
}


template < typename _Ty >
ptrdiff_t CalStride(int width, size_t Alignment = MEMORY_ALIGNMENT)
{
    size_t line_size = static_cast<size_t>(width) * sizeof(_Ty);
    return line_size % Alignment == 0 ? line_size : (line_size / Alignment + 1) * Alignment;
}

inline size_t ValidAlignment(size_t Alignment, ptrdiff_t stride = 0)
{
    if (Alignment <= 1) return 1;

    size_t rshift = 0;
    Alignment = (Alignment << 1) - 1;
    while (Alignment >>= 1)
    {
        ++rshift;
    }
    Alignment = size_t(1) << (rshift - 1);

    if (stride != 0)
    {
        while (Abs(stride) % Alignment)
        {
            Alignment >>= 1;
        }
    }

    return Alignment;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// row-strided data copy

template < typename _Dt1, typename _St1 >
void BitBlt(void *dstp, const void *srcp, int height, int width, ptrdiff_t dst_stride, ptrdiff_t src_stride)
{
    for (int j = 0; j < height; ++j)
    {
        auto dst = reinterpret_cast<_Dt1 *>(reinterpret_cast<uint8_t *>(dstp) + dst_stride);
        auto src = reinterpret_cast<const _St1 *>(reinterpret_cast<const uint8_t *>(srcp) + src_stride);

        for (int i = 0; i < width; ++i)
        {
            dst[i] = static_cast<_Dt1>(src[i]);
        }
    }
}

template < typename _Dt1, typename _St1 >
void BitBlt(_Dt1 *dstp, const _St1 *srcp, int height, int width, ptrdiff_t dst_stride, ptrdiff_t src_stride)
{
    STATIC_ASSERT(!std::is_same<_Dt1, void>::value && !std::is_same<_St1, void>::value,
        "BitBlt: instantiating with void pointer is not allowed here in this template function.");
    BitBlt<_Dt1, _St1>(reinterpret_cast<void *>(dstp), reinterpret_cast<const void *>(srcp),
        height, width, dst_stride * sizeof(_Dt1), src_stride * sizeof(_St1));
}


inline void BitBlt(void *dstp, const void *srcp, int height, size_t row_size, ptrdiff_t dst_stride, ptrdiff_t src_stride)
{
    if (dstp == srcp)
    {
        return;
    }

    if (height > 0)
    {
        if (src_stride == dst_stride && src_stride == row_size)
        {
            memcpy(dstp, srcp, height * row_size);
        }
        else
        {
            for (int j = 0; j < height; ++j)
            {
                memcpy(dstp, srcp, row_size);
                dstp = reinterpret_cast<uint8_t *>(dstp) + dst_stride;
                srcp = reinterpret_cast<const uint8_t *>(srcp) + src_stride;
            }
        }
    }
}

template < typename _Ty >
void BitBlt(void *dstp, const void *srcp, int height, int width, ptrdiff_t dst_stride, ptrdiff_t src_stride)
{
    BitBlt(dstp, srcp, height, width * sizeof(_Ty), dst_stride * sizeof(_Ty), src_stride * sizeof(_Ty));
}

template < typename _Ty >
void BitBlt(_Ty *dstp, const _Ty *srcp, int height, int width, ptrdiff_t dst_stride, ptrdiff_t src_stride)
{
    BitBlt(reinterpret_cast<void *>(dstp), reinterpret_cast<const void *>(srcp),
        height, width * sizeof(_Ty), dst_stride * sizeof(_Ty), src_stride * sizeof(_Ty));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
