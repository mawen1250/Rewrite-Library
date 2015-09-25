#ifndef RWL_CUDA_DEVICE_HELPER_H_
#define RWL_CUDA_DEVICE_HELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__
#define __CUDA_DEVICE__ __device__ __forceinline__
#else
#define __CUDA_DEVICE__
#endif

#ifdef __CUDACC__
#define __CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
#define __CUDA_HOST_DEVICE__
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename _Ty1, typename _Ty2 >
struct CUDA_KeyPair
    : public std::pair<_Ty1, _Ty2>
{
    typedef CUDA_KeyPair<_Ty1, _Ty2> _Myt;
    typedef std::pair<_Ty1, _Ty2> _Mybase;

    typedef _Ty1 KeyType;
    typedef _Ty2 ValType;

    CUDA_KeyPair()
        : _Mybase()
    {}

    CUDA_KeyPair(const _Ty1& _Val1, const _Ty2& _Val2)
        : _Mybase(_Val1, _Val2)
    {}

    CUDA_KeyPair(const _Myt &_Right)
        : _Mybase(_Right)
    {}

    CUDA_KeyPair(_Myt &&_Right)
        : _Mybase(_Right)
    {}

    __CUDA_DEVICE__ _Myt &operator=(const _Myt &_Right)
    {
        _Mybase::operator=(_Right);
        return *this;
    }

    __CUDA_DEVICE__ _Myt &operator=(_Myt &&_Right)
    {
        _Mybase::operator=(_Right);
        return *this;
    }

    __CUDA_DEVICE__ bool operator==(const _Myt &_Right)
    {
        return this->first == _Right.first;
    }

    __CUDA_DEVICE__ bool operator!=(const _Myt &_Right)
    {
        return this->first != _Right.first;
    }

    __CUDA_DEVICE__ bool operator<(const _Myt &_Right)
    {
        return this->first < _Right.first;
    }

    __CUDA_DEVICE__ bool operator>(const _Myt &_Right)
    {
        return this->first > _Right.first;
    }

    __CUDA_DEVICE__ bool operator<=(const _Myt &_Right)
    {
        return this->first <= _Right.first;
    }

    __CUDA_DEVICE__ bool operator>=(const _Myt &_Right)
    {
        return this->first >= _Right.first;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Max Min
template < typename T >
__CUDA_DEVICE__ T cuMax(T a, T b)
{
    return a < b ? b : a;
}

template < typename T >
__CUDA_DEVICE__ T cuMin(T a, T b)
{
    return a > b ? b : a;
}

template < typename T >
__CUDA_DEVICE__ T cuClip(T input, T lower, T upper)
{
    return input >= upper ? upper : input <= lower ? lower : input;
}

// Abs
template < typename T >
__CUDA_DEVICE__ T cuAbs(T input)
{
    return input < 0 ? -input : input;
}

template < typename T >
__CUDA_DEVICE__ T cuAbsSub(T a, T b)
{
    return a >= b ? a - b : b - a;
}

// Initialization
template < typename _Ty >
__global__ void CUDA_Set_Kernel(_Ty *dst, const cuIdx count, _Ty value)
{
    const cuIdx idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count)
    {
        dst[idx] = value;
    }
}

template < typename _Ty >
inline void CUDA_Set(_Ty *dst, const cuIdx count, _Ty value = 0, cuIdx _block_dim = 256)
{
    if (count < _block_dim) _block_dim = count;
    CudaGlobalCall(CUDA_Set_Kernel, CudaGridDim(count, _block_dim), _block_dim)(dst, count, value);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__CUDA_DEVICE__ float atomicMinFloat(float *address, float val)
{
    unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_ul, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ul, assumed,
            __float_as_int(cuMin(val, __int_as_float(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

__CUDA_DEVICE__ float atomicMaxFloat(float *address, float val)
{
    unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_ul, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ul, assumed,
            __float_as_int(cuMax(val, __int_as_float(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

#ifdef _CUDA_1_2_
__CUDA_DEVICE__ float atomicAddFloat(float *address, float val)
{
    unsigned int *address_as_ul = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_ul, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ul, assumed,
            __float_as_int(val + __int_as_float(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}
#else
__CUDA_DEVICE__ float atomicAddFloat(float *address, float val)
{
    return atomicAdd(address, val);
}

__CUDA_DEVICE__ double atomicAddFloat(double *address, double val)
{
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__CUDA_DEVICE__ double atomicMinFloat(double *address, double val)
{
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(cuMin(val, __longlong_as_double(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__CUDA_DEVICE__ double atomicMaxFloat(double *address, double val)
{
    unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(cuMax(val, __longlong_as_double(assumed))));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // __CUDACC__

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
