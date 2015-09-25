#ifndef RWL_CORE_CORE_H_
#define RWL_CORE_CORE_H_

#include <memory>
#include <functional>
#include <vector>
#include <map>

#include "core/common.h"
#include "core/memory.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward declarations

struct RW_EXPORTS Format;
class RW_EXPORTS FormatManager;
struct RW_EXPORTS DataMemory;
struct RW_EXPORTS PlaneData;
class RW_EXPORTS Frame;

typedef uint8_t DataType;
typedef std::shared_ptr<const Format> FormatPtr;
typedef std::shared_ptr<DataType> DataPtr;
typedef std::shared_ptr<const DataMemory> DataMemoryPtr;
typedef std::shared_ptr<PlaneData> PlaneDataPtr;
typedef std::shared_ptr<Frame> FramePtr;

#define DataNullptr ((DataType *)(NULLPTR))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Point

template <typename _Ty>
struct _Point
{
    typedef _Point<_Ty> _Myt;

    _Ty x;
    _Ty y;

    _Point();
    _Point(int x, int y);

    void Scale(double scale_x, double scale_y, _Ty offset_x = 0, _Ty offset_y = 0);

    friend _Myt operator+(const _Myt &left, const _Myt &right);
    friend _Myt operator-(const _Myt &left, const _Myt &right);
    friend _Myt operator*(const _Myt &left, const _Ty &right);
    friend _Myt operator/(const _Myt &left, const _Ty &right);
    friend _Myt operator*(const _Ty &left, const _Myt &right);
    friend _Myt operator/(const _Ty &left, const _Myt &right);
    friend _Myt &operator+=(_Myt &left, const _Myt &right);
    friend _Myt &operator-=(_Myt &left, const _Myt &right);
    friend _Myt &operator*=(_Myt &left, const _Ty &right);
    friend _Myt &operator/=(_Myt &left, const _Ty &right);
};

typedef _Point<int> Point2I;
typedef _Point<float> Point2S;
typedef _Point<double> Point2D;
typedef Point2I Point;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rect

template <typename _Ty>
struct _Rect
{
    typedef _Rect<_Ty> _Myt;

    _Ty x;
    _Ty y;
    _Ty width;
    _Ty height;

    _Rect();
    _Rect(int x, int y, int width, int height);
    _Rect(const Point &p1, const Point &p2);

    bool Check(int image_width, int image_height, bool throw_ = false);
    void Normalize(int image_width, int image_height);
    void Scale(double scale_x, double scale_y, _Ty offset_x = 0, _Ty offset_y = 0);

    friend _Myt operator+(const _Myt &left, const _Myt &right);
    friend _Myt operator-(const _Myt &left, const _Myt &right);
    friend _Myt operator*(const _Myt &left, const _Ty &right);
    friend _Myt operator/(const _Myt &left, const _Ty &right);
    friend _Myt operator*(const _Ty &left, const _Myt &right);
    friend _Myt operator/(const _Ty &left, const _Myt &right);
    friend _Myt &operator+=(_Myt &left, const _Myt &right);
    friend _Myt &operator-=(_Myt &left, const _Myt &right);
    friend _Myt &operator*=(_Myt &left, const _Ty &right);
    friend _Myt &operator/=(_Myt &left, const _Ty &right);
};

typedef _Rect<int> Rect2I;
typedef _Rect<float> Rect2S;
typedef _Rect<double> Rect2D;
typedef Rect2I Rect;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Format

struct RW_EXPORTS Format
{
    enum Type
    {
        // Planar formats
        Gray = 0x00,
        RGB = 0x01,
        RGBA = 0x02,
        YUV = 0x03,
        // Packed(Interleaved) formats
        RGB24 = 0x80,
        RGB32 = 0x81,
        RGBA32 = 0x82,
        GBR24 = 0x88,
        GBR32 = 0x89,
        GBRA32 = 0x8a,
        BGR24 = 0x90,
        BGR32 = 0x91,
        BGRA32 = 0x92,
        YUYV = 0xc0, // YUY2
        YVYU = 0xc1, // YUY2
        UYVY = 0xc2, // YUY2
        VYUY = 0xc3, // YUY2
        AYUV = 0xc8,
        // Undefined
        Undef = 0xff
    };

    enum Sample
    {
        UInteger = 0x0,
        Integer = 0x1,
        Float = 0x2,
        Complex = 0x3
    };

    enum IDMask
    {
        typeMask = 0xff000000,
        sswMask = 0x00e00000,
        sshMask = 0x001c0000,
        BpsMask = 0x0003f000,
        bpsMask = 0x00000ff0,
        sampleMask = 0x0000000f
    };

    enum IDShift
    {
        typeShift = 24,
        sswShift = 21,
        sshShift = 18,
        BpsShift = 12,
        bpsShift = 4,
        sampleShift = 0
    };

    int id; // unique identifier for specific format
    int type; // type of the color space | id[31:24] | [0,255]
    int channels; // channel number of the color space
    int subsample_w; // chroma sub-sampling of width | id[23:21] | [0,7]
    int subsample_h; // chroma sub-sampling of height | id[20:18] | [0,7]
    int Bps; // Bytes per sample in a *plane* | id[17:12] | [0,63]
    int bps; // bits per sample in a *channel* | id[11:4] | [0,255]
    int sample; // sample type | id[3:0] | [0,15]

    Format();

    bool IsPlanar() const;
    bool IsPacked() const;
    bool IsYUV() const;
    bool IsRGB() const;
    bool IsRGBA() const;

    static void TypeRestrict(Format &format, int type = -1);
    static int bps2Bps(int bps);
    static int Format2ID(const Format &format);
    static Format ID2Format(int id);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FormatManager

class RW_EXPORTS FormatManager
{
public:
    typedef std::map<int, FormatPtr> FormatMap;

public:
    FormatPtr operator()(FormatPtr format);
    FormatPtr operator()(int type, int bps = -1, int sample = -1,
        int subsample_w = -1, int subsample_h = -1);
    FormatPtr RegisterFormat(int type, int bps = -1, int sample = -1,
        int subsample_w = -1, int subsample_h = -1);

    bool IsRegistered(const FormatPtr &format) const;

private:
    FormatMap _formats;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Preset formats

extern RW_EXPORTS FormatManager formatManager;

extern RW_EXPORTS const FormatPtr fGray8U;
extern RW_EXPORTS const FormatPtr fGray16U;
extern RW_EXPORTS const FormatPtr fGray16I;
extern RW_EXPORTS const FormatPtr fGray32I;
extern RW_EXPORTS const FormatPtr fGray32F;
extern RW_EXPORTS const FormatPtr fGray64F;
extern RW_EXPORTS const FormatPtr fGray64C;
extern RW_EXPORTS const FormatPtr fGray128C;

extern RW_EXPORTS const FormatPtr fRGB8U;
extern RW_EXPORTS const FormatPtr fRGB16U;
extern RW_EXPORTS const FormatPtr fRGB16I;
extern RW_EXPORTS const FormatPtr fRGB32I;
extern RW_EXPORTS const FormatPtr fRGB32F;
extern RW_EXPORTS const FormatPtr fRGB64F;
extern RW_EXPORTS const FormatPtr fRGB64C;
extern RW_EXPORTS const FormatPtr fRGB128C;

extern RW_EXPORTS const FormatPtr fYUV444P8U;
extern RW_EXPORTS const FormatPtr fYUV444P16U;
extern RW_EXPORTS const FormatPtr fYUV444P16I;
extern RW_EXPORTS const FormatPtr fYUV444P32I;
extern RW_EXPORTS const FormatPtr fYUV444P32F;
extern RW_EXPORTS const FormatPtr fYUV444P64F;
extern RW_EXPORTS const FormatPtr fYUV444P64C;
extern RW_EXPORTS const FormatPtr fYUV444P128C;

extern RW_EXPORTS const FormatPtr fYUV422P8U;
extern RW_EXPORTS const FormatPtr fYUV422P16U;
extern RW_EXPORTS const FormatPtr fYUV422P16I;
extern RW_EXPORTS const FormatPtr fYUV422P32I;
extern RW_EXPORTS const FormatPtr fYUV422P32F;
extern RW_EXPORTS const FormatPtr fYUV422P64F;
extern RW_EXPORTS const FormatPtr fYUV422P64C;
extern RW_EXPORTS const FormatPtr fYUV422P128C;

extern RW_EXPORTS const FormatPtr fYUV420P8U;
extern RW_EXPORTS const FormatPtr fYUV420P16U;
extern RW_EXPORTS const FormatPtr fYUV420P16I;
extern RW_EXPORTS const FormatPtr fYUV420P32I;
extern RW_EXPORTS const FormatPtr fYUV420P32F;
extern RW_EXPORTS const FormatPtr fYUV420P64F;
extern RW_EXPORTS const FormatPtr fYUV420P64C;
extern RW_EXPORTS const FormatPtr fYUV420P128C;

extern RW_EXPORTS const FormatPtr fYUV411P8U;
extern RW_EXPORTS const FormatPtr fYUV411P16U;
extern RW_EXPORTS const FormatPtr fYUV411P16I;
extern RW_EXPORTS const FormatPtr fYUV411P32I;
extern RW_EXPORTS const FormatPtr fYUV411P32F;
extern RW_EXPORTS const FormatPtr fYUV411P64F;
extern RW_EXPORTS const FormatPtr fYUV411P64C;
extern RW_EXPORTS const FormatPtr fYUV411P128C;

extern RW_EXPORTS const FormatPtr fYUV410P8U;
extern RW_EXPORTS const FormatPtr fYUV410P16U;
extern RW_EXPORTS const FormatPtr fYUV410P16I;
extern RW_EXPORTS const FormatPtr fYUV410P32I;
extern RW_EXPORTS const FormatPtr fYUV410P32F;
extern RW_EXPORTS const FormatPtr fYUV410P64F;
extern RW_EXPORTS const FormatPtr fYUV410P64C;
extern RW_EXPORTS const FormatPtr fYUV410P128C;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemory

struct RW_EXPORTS DataMemory
{
    typedef std::function<DataType *(int row_size, int height, int &stride, size_t &alignment)> Allocator;
    typedef std::function<void(DataType *memory)> Deallocator;

    enum
    {
        mtStride = 0x100,
        mtSubStep = 0x10
    };

    enum MemoryType
    {
        None = 0,
        CPU = None + mtStride,
        CPU_Continuous = CPU + mtSubStep,
        OpenCL = CPU + mtStride,
        OpenCL_Continuous = OpenCL + mtSubStep,
        CUDA = OpenCL + mtStride,
        CUDA_Continuous = CUDA + mtSubStep,
        Custom = mtStride * 0x400000
    };

    Allocator alloc;
    Deallocator dealloc;

    int Type() const { return _type; }
    int TypeMain() const { return _type / mtStride * mtStride; }
    int TypeSub() const { return _type % mtStride / mtSubStep * mtSubStep; }

protected:
    DataMemory(int type, Allocator alloc, Deallocator dealloc);

private:
    int _type;
};

struct RW_EXPORTS DataMemoryNone
    : public DataMemory
{
    DataMemoryNone();
};

struct RW_EXPORTS DataMemoryCPU
    : public DataMemory
{
    DataMemoryCPU();
};

struct RW_EXPORTS DataMemoryCPU_Continuous
    : public DataMemory
{
    DataMemoryCPU_Continuous();
};

struct RW_EXPORTS DataMemoryOpenCL
    : public DataMemory
{
    DataMemoryOpenCL(); // not implemented yet
};

struct RW_EXPORTS DataMemoryOpenCL_Continuous
    : public DataMemory
{
    DataMemoryOpenCL_Continuous(); // not implemented yet
};

struct RW_EXPORTS DataMemoryCUDA
    : public DataMemory
{
    DataMemoryCUDA();
};

struct RW_EXPORTS DataMemoryCUDA_Continuous
    : public DataMemory
{
    DataMemoryCUDA_Continuous();
};

extern RW_EXPORTS const DataMemoryPtr dmNone;
extern RW_EXPORTS const DataMemoryPtr dmCPU;
extern RW_EXPORTS const DataMemoryPtr dmCPU_Continuous;
extern RW_EXPORTS const DataMemoryPtr dmOpenCL; // not implemented yet
extern RW_EXPORTS const DataMemoryPtr dmOpenCL_Continuous; // not implemented yet
extern RW_EXPORTS const DataMemoryPtr dmCUDA;
extern RW_EXPORTS const DataMemoryPtr dmCUDA_Continuous;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PlaneData

struct RW_EXPORTS PlaneData
{
private:
    DataMemoryPtr _memory; // DataMemory pointer containing memory type, allocator and deallocator
    DataPtr _data; // pointer to the beginning of allocated memory

public:
    int width; // image width in pixels
    int height; // image height in pixels
    int stride; // ptrdiff to the next row in Bytes (can be negative)
    int Bps; // Bytes per sample
    size_t alignment; // row alignment in Bytes
    DataPtr image; // pointer to the beginning of the image (aliasing constructed from _data)

    PlaneData(int width, int height, int stride, int Bps, DataMemoryPtr memory = dmCPU, ptrdiff_t offset = 0, size_t alignment = MEMORY_ALIGNMENT);
    PlaneData(void *data, int width, int height, int stride, int Bps, DataMemoryPtr memory = dmNone, ptrdiff_t offset = 0, size_t alignment = MEMORY_ALIGNMENT);
    PlaneData(const PlaneData &ref, Rect roi);
    PlaneData(PlaneDataPtr ref);
    PlaneData(PlaneDataPtr ref, Rect roi);

    void Create(int width, int height, int stride, int Bps, DataMemoryPtr memory = dmCPU, ptrdiff_t offset = 0, size_t alignment = MEMORY_ALIGNMENT);
    void Release();
    PlaneDataPtr GetPtr() const;
    PlaneData Clone() const;
    PlaneDataPtr ClonePtr() const;

public: // inline/template functions
    PlaneData(PlaneData &&src);
    PlaneData &operator=(PlaneData &&src);
    bool operator==(const PlaneData &right) const;
    bool operator!=(const PlaneData &right) const;

    int MemoryType() const;
    bool Continuous() const;

    bool Empty() const;
    bool Unique() const;
    int UseCount() const;

    const DataType *Ptr() const;
    DataType *Ptr();
    template <typename _Ty> const _Ty *Ptr() const;
    template <typename _Ty> _Ty *Ptr();
    template <typename _Ty> const _Ty *Ptr(int y) const;
    template <typename _Ty> _Ty *Ptr(int y);
    template <typename _Ty> const _Ty *Ptr(int y, int x) const;
    template <typename _Ty> _Ty *Ptr(int y, int x);
    template <typename _Ty> const _Ty *Ptr(int y, int x, int c) const;
    template <typename _Ty> _Ty *Ptr(int y, int x, int c);

private:
    void createData(ptrdiff_t offset);
    void releaseData();
    void image2align();
    ptrdiff_t getOffset() const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Frame

class RW_EXPORTS Frame
{
public:
    Frame();
    Frame(int width, int height, FormatPtr format = fGray8U, DataMemoryPtr memory = dmCPU, size_t alignment = MEMORY_ALIGNMENT);
    Frame(const Frame &ref, Rect roi);
    Frame(FramePtr ref);
    Frame(FramePtr ref, Rect roi);

public:
    void Create(int width, int height, FormatPtr format = fGray8U, DataMemoryPtr memory = dmCPU, size_t alignment = MEMORY_ALIGNMENT);
    void Release();
    FramePtr GetPtr() const;
    Frame Clone() const;
    FramePtr ClonePtr() const;

public: // inline/template functions
    Frame(Frame &&src);
    Frame &operator=(Frame &&src);
    bool operator==(const Frame &right) const;
    bool operator!=(const Frame &right) const;

    FormatPtr Format() const;
    int MemoryType() const;
    bool Continuous(int plane) const;
    int Bps() const;
    size_t Alignment() const;

    bool IsPlanar() const;
    bool IsPacked() const;
    bool IsYUV() const;
    bool IsRGB() const;
    bool IsRGBA() const;
    bool IsChroma(int plane) const;

    int Channels() const;
    int Planes() const;
    int Width(int plane) const;
    int Width() const;
    int Height(int plane) const;
    int Height() const;
    int Stride(int plane = 0) const;
    int SubSampleWRatio(int channel = 1) const;
    int SubSampleHRatio(int channel = 1) const;

    bool Empty(int plane) const;
    bool Empty() const;
    bool Unique(int plane) const;
    bool Unique() const;
    int UseCount(int plane = 0) const;

    const DataType *Ptr(int plane = 0) const;
    DataType *Ptr(int plane = 0);
    template <typename _Ty> const _Ty *Ptr(int plane = 0) const;
    template <typename _Ty> _Ty *Ptr(int plane = 0);
    template <typename _Ty> const _Ty *Ptr(int plane, int y) const;
    template <typename _Ty> _Ty *Ptr(int plane, int y);
    template <typename _Ty> const _Ty *Ptr(int plane, int y, int x) const;
    template <typename _Ty> _Ty *Ptr(int plane, int y, int x);
    template <typename _Ty> const _Ty *Ptr(int plane, int y, int x, int c) const;
    template <typename _Ty> _Ty *Ptr(int plane, int y, int x, int c);

private:
    void createData();
    void releaseData();

private:
    int _width; // width of the image
    int _height; // height of the image
    FormatPtr _format; // Format pointer containing all the image format information
    DataMemoryPtr _memory; // DataMemory pointer for creating PlaneData
    size_t _alignment; // expected row alignment, for creating PlaneData(the actual alignment used can be different)
    std::vector<PlaneDataPtr> _data; // vector of PlaneDataPtr, each one points to a plane
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Include template definitions

#include "core/core.hpp"
#include "core/frame.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
