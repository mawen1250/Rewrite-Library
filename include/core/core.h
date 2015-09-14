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

struct Format;
class FormatManager;
struct DataMemory;
struct PlaneData;
class Frame;

typedef uint8_t DataType;
typedef std::shared_ptr<const Format> FormatPtr;
typedef std::shared_ptr<DataType> DataPtr;
typedef std::shared_ptr<const DataMemory> DataMemoryPtr;
typedef std::shared_ptr<PlaneData> PlaneDataPtr;
typedef std::shared_ptr<Frame> FramePtr;

#define DataNullptr ((DataType *)(NULLPTR))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Format

struct Format
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

    static void TypeRestrict(Format &format, int type = -1);
    static int bps2Bps(int bps);
    static int Format2ID(const Format &format);
    static Format ID2Format(int id);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FormatManager

class FormatManager
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

extern FormatManager formatManager;

extern const FormatPtr fGray8U;
extern const FormatPtr fGray16U;
extern const FormatPtr fGray16I;
extern const FormatPtr fGray32I;
extern const FormatPtr fGray32F;
extern const FormatPtr fGray64F;
extern const FormatPtr fGray64C;
extern const FormatPtr fGray128C;

extern const FormatPtr fRGB8U;
extern const FormatPtr fRGB16U;
extern const FormatPtr fRGB16I;
extern const FormatPtr fRGB32I;
extern const FormatPtr fRGB32F;
extern const FormatPtr fRGB64F;
extern const FormatPtr fRGB64C;
extern const FormatPtr fRGB128C;

extern const FormatPtr fYUV444P8U;
extern const FormatPtr fYUV444P16U;
extern const FormatPtr fYUV444P16I;
extern const FormatPtr fYUV444P32I;
extern const FormatPtr fYUV444P32F;
extern const FormatPtr fYUV444P64F;
extern const FormatPtr fYUV444P64C;
extern const FormatPtr fYUV444P128C;

extern const FormatPtr fYUV422P8U;
extern const FormatPtr fYUV422P16U;
extern const FormatPtr fYUV422P16I;
extern const FormatPtr fYUV422P32I;
extern const FormatPtr fYUV422P32F;
extern const FormatPtr fYUV422P64F;
extern const FormatPtr fYUV422P64C;
extern const FormatPtr fYUV422P128C;

extern const FormatPtr fYUV420P8U;
extern const FormatPtr fYUV420P16U;
extern const FormatPtr fYUV420P16I;
extern const FormatPtr fYUV420P32I;
extern const FormatPtr fYUV420P32F;
extern const FormatPtr fYUV420P64F;
extern const FormatPtr fYUV420P64C;
extern const FormatPtr fYUV420P128C;

extern const FormatPtr fYUV411P8U;
extern const FormatPtr fYUV411P16U;
extern const FormatPtr fYUV411P16I;
extern const FormatPtr fYUV411P32I;
extern const FormatPtr fYUV411P32F;
extern const FormatPtr fYUV411P64F;
extern const FormatPtr fYUV411P64C;
extern const FormatPtr fYUV411P128C;

extern const FormatPtr fYUV410P8U;
extern const FormatPtr fYUV410P16U;
extern const FormatPtr fYUV410P16I;
extern const FormatPtr fYUV410P32I;
extern const FormatPtr fYUV410P32F;
extern const FormatPtr fYUV410P64F;
extern const FormatPtr fYUV410P64C;
extern const FormatPtr fYUV410P128C;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Point

struct Point
{
    int x;
    int y;

    Point();
    Point(int x, int y);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rect

struct Rect
{
    int x;
    int y;
    int width;
    int height;

    Rect();
    Rect(int x, int y, int width, int height);
    Rect(const Point &p1, const Point &p2);

    bool Check(int width_, int height_, bool throw_ = false);
    void Normalize(int width_, int height_);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemory

struct DataMemory
{
    typedef std::function<DataType *(int, int, int &, size_t &)> Allocator;
    typedef std::function<void(DataType *)> Deallocator;

    enum MemoryType
    {
        Custom = -1,
        None = 0,
        CPU = 1,
        OpenCL = 2,
        CUDA = 3
    };

    Allocator alloc;
    Deallocator dealloc;

    int Type() const { return _type; }

protected:
    DataMemory(int type, Allocator alloc, Deallocator dealloc);

private:
    int _type;
};

struct DataMemoryNone
    : public DataMemory
{
    DataMemoryNone();
};

struct DataMemoryCPU
    : public DataMemory
{
    DataMemoryCPU();
};

extern const DataMemoryPtr dmNone;
extern const DataMemoryPtr dmCPU;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PlaneData

struct PlaneData
{
private:
    DataMemoryPtr _memory;
    DataPtr _data; // pointer to the beginning of allocated memory

public:
    int width; // image width in pixels
    int height; // image height in pixels
    int stride; // ptrdiff to the next row in Bytes (can be negative)
    size_t alignment; // row alignment in Bytes
    DataPtr image; // pointer to the beginning of the image (aliasing constructed from _data)

    PlaneData(int width, int height, int stride, DataMemoryPtr memory = dmCPU, ptrdiff_t offset = 0, size_t alignment = MEMORY_ALIGNMENT);
    PlaneData(void *data, int width, int height, int stride, DataMemoryPtr memory = dmNone, ptrdiff_t offset = 0, size_t alignment = MEMORY_ALIGNMENT);
    PlaneData(PlaneDataPtr ref);
    PlaneData(PlaneDataPtr ref, Rect roi, int Bps);

    PlaneDataPtr Clone() const;
    int MemoryType() const;

private:
    void createData(ptrdiff_t offset);
    void image2align();
    ptrdiff_t getOffset() const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Frame

class Frame
{
public:
    Frame();
    Frame(int width, int height, FormatPtr format, DataMemoryPtr memory, size_t alignment = MEMORY_ALIGNMENT);

public:
    void Create(int width, int height, FormatPtr format, DataMemoryPtr memory, size_t alignment = MEMORY_ALIGNMENT);
    void Release();

    bool empty() const;

private:
    void createData();
    void releaseData();

private:
    int _width;
    int _height;
    FormatPtr _format;
    DataMemoryPtr _memory;
    size_t _alignment;
    std::vector<PlaneDataPtr> _data;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
