#include "cuda/cuda.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const dim3 MF2D_blockDim(32, 8);

static __global__ void MedianFilter2D_kernel_r1(PtrCUDA<uint8_t> dst, const PtrCUDA<uint8_t> src)
{

}

static __global__ void MedianFilter2D_kernel_r2(PtrCUDA<uint8_t> dst, const PtrCUDA<uint8_t> src)
{

}

static __global__ void MedianFilter2D_kernel_r3(PtrCUDA<uint8_t> dst, const PtrCUDA<uint8_t> src)
{

}

void MedianFilter2D(Frame &dst, const Frame &src, int radius, CudaStream _stream = CudaStream::null)
{
    static const std::string funcName = "MedianFilter2D";

    if (src.MemoryTypeMain() != DataMemory::CUDA)
    {
        throw std::invalid_argument(funcName + ": only DataMemory::CUDA is supported!");
    }

    for (int p = 0; p < src.Planes(); ++p)
    {
        int height = src.Height(p);
        int width = src.Width(p);

        PtrCUDA<uint8_t> dstp(dst, p), srcp(src, p);
        dim3 block_dim = MF2D_blockDim;
        dim3 grid_dim(CudaGridDim(width, block_dim.x), CudaGridDim(height, block_dim.y));

        switch (radius)
        {
        case 1:
            CudaStreamCall(MedianFilter2D_kernel_r1, grid_dim, block_dim, _stream)(dstp, srcp);
            break;
        case 2:
            CudaStreamCall(MedianFilter2D_kernel_r2, grid_dim, block_dim, _stream)(dstp, srcp);
            break;
        case 3:
            CudaStreamCall(MedianFilter2D_kernel_r3, grid_dim, block_dim, _stream)(dstp, srcp);
            break;
        default:
            throw std::invalid_argument(funcName + ": unsupported radius(=" + std::to_string(radius) + ")!");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
