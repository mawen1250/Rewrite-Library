#ifndef RWL_CUDA_CUDA_H_
#define RWL_CUDA_CUDA_H_

#include "core/core.h"
#include "cuda/host_helper.h"
#include "cuda/device_helper.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Instantiate and export template classes
// Type definitions

template class RW_EXPORTS std::shared_ptr<struct CUstream_st>;
typedef std::shared_ptr<struct CUstream_st> CUStreamPtr;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CudaStream

class RW_EXPORTS CudaStream
{
public:
    typedef CudaStream _Myt;

    static const _Myt null;

    static void PriorityRange(int &leastPriority, int &greatestPriority);

public:
    explicit CudaStream(cuIdx flags = cudaStreamDefault, int priority = 0);
    CudaStream(::cudaStream_t stream);
    CudaStream(const std::nullptr_t &);

    operator ::cudaStream_t() const;

public:
    ::cudaStream_t Stream() const;
    cuIdx Flags() const;
    int Priority() const;
    void WaitEvent(::cudaEvent_t event, cuIdx flags = 0) const;
    void AddCallback(::cudaStreamCallback_t callback, void *userData, cuIdx flags = 0) const;
    void Synchronize() const;
    bool Query() const;

    template < typename _Ty >
    void AttachMemAsync(_Ty *devPtr, size_t length = 0, cuIdx flags = cudaMemAttachSingle)
    {
        checkCudaErrors(::cudaStreamAttachMemAsync(Stream(), reinterpret_cast<void *>(devPtr), length, flags));
    }

private:
    void create(cuIdx flags, int priority);

private:
    CUStreamPtr _stream;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PtrCUDA

template <typename _Ty>
class PtrCUDA
{
public:
    typedef PtrCUDA<_Ty> _Myt;

    int width; // image width in pixels
    int height; // image height in pixels
    int stride; // ptrdiff to the next row in Bytes (can be negative)
    int Bps; // Bytes per sample

    explicit PtrCUDA(const PlaneData &pdata);
    explicit PtrCUDA(const Frame &frame, int plane);

    __CUDA_DEVICE__ const _Ty &operator[](const ptrdiff_t &offset_bytes) const;
    __CUDA_DEVICE__ _Ty &operator[](const ptrdiff_t &offset_bytes);
    __CUDA_DEVICE__ const _Ty &operator()(const int &y, const int &x) const;
    __CUDA_DEVICE__ _Ty &operator()(const int &y, const int &x);
    __CUDA_DEVICE__ const _Ty &operator()(const int &y, const int &x, const int &c) const;
    __CUDA_DEVICE__ _Ty &operator()(const int &y, const int &x, const int &c);

    __CUDA_DEVICE__ const DataType *Data() const;
    __CUDA_DEVICE__ DataType *Data();
    __CUDA_DEVICE__ const _Ty *Ptr() const;
    __CUDA_DEVICE__ _Ty *Ptr();
    __CUDA_DEVICE__ const _Ty *Ptr(const int &y) const;
    __CUDA_DEVICE__ _Ty *Ptr(const int &y);
    __CUDA_DEVICE__ const _Ty *Ptr(const int &y, const int &x) const;
    __CUDA_DEVICE__ _Ty *Ptr(const int &y, const int &x);
    __CUDA_DEVICE__ const _Ty *Ptr(const int &y, const int &x, const int &c) const;
    __CUDA_DEVICE__ _Ty *Ptr(const int &y, const int &x, const int &c);

private:
    DataType *_data; // pointer to the beginning of the image (aliasing constructed from _data)
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Include inline/template definitions

#include "cuda/cuda.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
