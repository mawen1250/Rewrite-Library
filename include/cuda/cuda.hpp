#ifndef RWL_CUDA_CUDA_HPP_
#define RWL_CUDA_CUDA_HPP_

#include "core/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PtrCUDA

template <typename _Ty>
PtrCUDA<_Ty>::PtrCUDA(const PlaneData &pdata)
    : width(pdata.width), height(pdata.height), stride(pdata.stride), Bps(pdata.Bps), _data(pdata.image.get())
{}

template <typename _Ty>
PtrCUDA<_Ty>::PtrCUDA(const Frame &frame, int plane)
    : PtrCUDA(*frame.GetPlanePtr(plane))
{}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty &PtrCUDA<_Ty>::operator[](const ptrdiff_t &offset_bytes) const
{
    return *reinterpret_cast<const _Ty *>(_data + offset_bytes);
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty &PtrCUDA<_Ty>::operator[](const ptrdiff_t &offset_bytes)
{
    return *reinterpret_cast<_Ty *>(_data + offset_bytes);
}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty &PtrCUDA<_Ty>::operator()(const int &y, const int &x) const
{
    return *Ptr(y, x);
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty &PtrCUDA<_Ty>::operator()(const int &y, const int &x)
{
    return *Ptr(y, x);
}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty &PtrCUDA<_Ty>::operator()(const int &y, const int &x, const int &c) const
{
    return *Ptr(y, x, c);
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty &PtrCUDA<_Ty>::operator()(const int &y, const int &x, const int &c)
{
    return *Ptr(y, x, c);
}

template <typename _Ty>
__CUDA_DEVICE__ const DataType *PtrCUDA<_Ty>::Data() const
{
    return _data;
}

template <typename _Ty>
__CUDA_DEVICE__ DataType *PtrCUDA<_Ty>::Data()
{
    return _data;
}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty *PtrCUDA<_Ty>::Ptr() const
{
    return reinterpret_cast<const _Ty *>(_data);
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty *PtrCUDA<_Ty>::Ptr()
{
    return reinterpret_cast<_Ty *>(_data);
}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty *PtrCUDA<_Ty>::Ptr(const int &y) const
{
    return reinterpret_cast<const _Ty *>(_data + y * stride);
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty *PtrCUDA<_Ty>::Ptr(const int &y)
{
    return reinterpret_cast<_Ty *>(_data + y * stride);
}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty *PtrCUDA<_Ty>::Ptr(const int &y, const int &x) const
{
    return reinterpret_cast<const _Ty *>(_data + y * stride + x * Bps);
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty *PtrCUDA<_Ty>::Ptr(const int &y, const int &x)
{
    return reinterpret_cast<_Ty *>(_data + y * stride + x * Bps);
}

template <typename _Ty>
__CUDA_DEVICE__ const _Ty *PtrCUDA<_Ty>::Ptr(const int &y, const int &x, const int &c) const
{
    return reinterpret_cast<const _Ty *>(_data + y * stride + x * Bps + c * sizeof(_Ty));
}

template <typename _Ty>
__CUDA_DEVICE__ _Ty *PtrCUDA<_Ty>::Ptr(const int &y, const int &x, const int &c)
{
    return reinterpret_cast<_Ty *>(_data + y * stride + x * Bps + c * sizeof(_Ty));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
