#include "cuda/cuda.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CudaStream

void CudaStream::PriorityRange(int &leastPriority, int &greatestPriority)
{
    checkCudaErrors(::cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
}

CudaStream::CudaStream(cuIdx flags, int priority)
: _stream(nullptr)
{
    create(flags, priority);
}

CudaStream::CudaStream(::cudaStream_t stream)
    : _stream(stream, [](::cudaStream_t) {})
{}

CudaStream::CudaStream(const std::nullptr_t &)
    : _stream(nullptr)
{}

CudaStream::operator ::cudaStream_t() const
{
    return Stream();
}

::cudaStream_t CudaStream::Stream() const
{
    return _stream.get();
}

cuIdx CudaStream::Flags() const
{
    cuIdx flags;
    checkCudaErrors(::cudaStreamGetFlags(Stream(), &flags));
    return flags;
}

int CudaStream::Priority() const
{
    int priority;
    checkCudaErrors(::cudaStreamGetPriority(Stream(), &priority));
    return priority;
}

void CudaStream::WaitEvent(::cudaEvent_t event, cuIdx flags) const
{
    checkCudaErrors(::cudaStreamWaitEvent(Stream(), event, flags));
}

void CudaStream::AddCallback(::cudaStreamCallback_t callback, void *userData, cuIdx flags) const
{
    checkCudaErrors(::cudaStreamAddCallback(Stream(), callback, userData, flags));
}

void CudaStream::Synchronize() const
{
    checkCudaErrors(::cudaStreamSynchronize(Stream()));
}

bool CudaStream::Query() const
{
    return ::cudaStreamQuery(Stream()) == ::cudaSuccess;
}

void CudaStream::create(cuIdx flags, int priority)
{
    cudaStream_t stream_t;
    checkCudaErrors(::cudaStreamCreateWithPriority(&stream_t, flags, priority));

    _stream.reset(stream_t, [](cudaStream_t stream)
    {
        checkCudaErrors(::cudaStreamDestroy(stream));
    });
}

const CudaStream CudaStream::null(nullptr);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
