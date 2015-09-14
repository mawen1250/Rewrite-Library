#include "core/core.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Point

Point::Point()
    : x(0), y(0)
{}

Point::Point(int x, int y)
    : x(x), y(y)
{}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rect

Rect::Rect()
    : x(0), y(0), width(0), height(0)
{}

Rect::Rect(int x, int y, int width, int height)
    : x(x), y(y), width(width), height(height)
{}

Rect::Rect(const Point &p1, const Point &p2)
    : x(p1.x), y(p1.y), width(p2.x - p1.x), height(p2.y - p1.y)
{}

bool Rect::Check(int width_, int height_, bool throw_)
{
    static const std::string funcName = "Rect::Check: ";

    if (width_ <= 0 || height_ <= 0)
    {
        throw std::invalid_argument(funcName + "\"width_\" and \"height_\" must be positive!");
    }

    if (x < 0 || x >= width_)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->x\" out of range!");
        else return false;
    }
    if (y < 0 || y >= height_)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->y\" out of range!");
        else return false;
    }

    if (width <= 0) width += width_ - x;
    if (height <= 0) height += height_ - y;

    if (width <= 0 || x + width > width_)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->width\" out of range!");
        else return false;
    }
    if (height <= 0 || x + height > height_)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->height\" out of range!");
        else return false;
    }

    return true;
}

void Rect::Normalize(int width_, int height_)
{
    static const std::string funcName = "Rect::Normalize: ";

    if (width_ <= 0 || height_ <= 0)
    {
        throw std::invalid_argument(funcName + "\"width_\" and \"height_\" must be positive!");
    }

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= width_) x = width_ - 1;
    if (y >= height_) y = height_ - 1;

    if (width <= 0) width += width_ - x;
    if (height <= 0) height += height_ - y;
    if (x + width > width_) width = width_ - x;
    if (y + height > height_) height = height_ - y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DataMemory

DataMemory::DataMemory(int type, Allocator alloc, Deallocator dealloc)
    : _type(type), alloc(std::move(alloc)), dealloc(std::move(dealloc))
{}

DataMemoryNone::DataMemoryNone()
    : DataMemory(None,
    [](int width, int height, int &stride, size_t &alignment)
    {
        return DataNullptr;
    },
    [](DataType *memory)
    {})
{}

DataMemoryCPU::DataMemoryCPU()
    : DataMemory(CPU,
    [](int width, int height, int &stride, size_t &alignment)
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
// Preset smart pointers to DataMemory

const DataMemoryPtr dmNone = std::make_shared<const DataMemoryNone>();
const DataMemoryPtr dmCPU = std::make_shared<const DataMemoryCPU>();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PlaneData

PlaneData::PlaneData(int width, int height, int stride, DataMemoryPtr memory, ptrdiff_t offset, size_t alignment)
    : width(width), height(height), stride(stride), alignment(ValidAlignment(alignment, stride)),
    _memory(memory), _data(NULLPTR), image(NULLPTR)
{
    createData(offset);
}

PlaneData::PlaneData(void *data, int width, int height, int stride, DataMemoryPtr memory, ptrdiff_t offset, size_t alignment)
    : width(width), height(height), stride(stride), alignment(ValidAlignment(alignment, stride)),
    _memory(memory), _data(reinterpret_cast<DataType *>(data), memory->dealloc),
    image(_data, reinterpret_cast<DataType *>(data) + offset)
{
    image2align();
}

PlaneData::PlaneData(PlaneDataPtr ref)
    : width(ref->width), height(ref->height), stride(ref->stride), alignment(ref->alignment),
    _memory(ref->_memory), _data(ref->_data), image(ref->image)
{}

PlaneData::PlaneData(PlaneDataPtr ref, Rect roi, int Bps)
    : width(roi.width), height(roi.height), stride(ref->stride), alignment(ref->alignment),
    _memory(ref->_memory), _data(ref->_data), image(NULLPTR)
{
    roi.Check(ref->width, ref->height, true);
    ptrdiff_t offset = roi.y * static_cast<size_t>(stride) + roi.x * Bps;
    image = DataPtr(ref->_data, ref->image.get() + offset); // aliasing constructor
    image2align();
}

PlaneDataPtr PlaneData::Clone() const
{
    PlaneDataPtr dst = std::make_shared<PlaneData>(width, height, stride, _memory, alignment, getOffset());
    BitBlt(dst->image.get(), image.get(), dst->height, dst->width, dst->stride, stride);
    return dst;
}

int PlaneData::MemoryType() const
{
    return _memory->Type();
}

void PlaneData::createData(ptrdiff_t offset)
{
    static const std::string funcName = "PlaneData::create: ";

    if (width < 0 || height < 0)
    {
        throw std::invalid_argument(funcName + "\"width\" and \"height\" cannot be negative!");
    }

    DataType *alloc_mem = _memory->alloc(width, height, stride, alignment);
    if (alloc_mem)
    {
        _data = DataPtr(alloc_mem, _memory->dealloc);
        image = DataPtr(_data, alloc_mem + offset);
    }
    else
    {
        _data = NULLPTR;
        image = NULLPTR;
    }

    if (offset != 0)
    {
        image2align();
    }
}

void PlaneData::image2align()
{
    alignment = ValidAlignment(alignment, image.get() - DataNullptr);
}

ptrdiff_t PlaneData::getOffset() const
{
    return _data && image ? image.get() - _data.get() : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Frame

Frame::Frame()
    : _width(0), _height(0), _format(fGray8U), _memory(dmNone), _alignment(MEMORY_ALIGNMENT)
{}

Frame::Frame(int width, int height, FormatPtr format, DataMemoryPtr memory, size_t alignment)
{
    Create(width, height, format, memory, alignment);
}

void Frame::Create(int width, int height, FormatPtr format, DataMemoryPtr memory, size_t alignment)
{
    static const std::string funcName = "Frame::Create: ";

    if (width < 0 || height < 0)
    {
        throw std::invalid_argument(funcName + "\"width\" and \"height\" cannot be negative!");
    }
    if (!(format = formatManager(format)))
    {
        throw std::invalid_argument(funcName + "\"format\" cannot be nullptr!");
    }
    if (width > 0 && width % (1 << format->subsample_w))
    {
        throw std::invalid_argument(funcName + "\"width\" must be MOD" + std::to_string(1 << format->subsample_w)
            + " for \"subsample_w\"=" + std::to_string(format->subsample_w) + " !");
    }
    if (height > 0 && height % (1 << format->subsample_w))
    {
        throw std::invalid_argument(funcName + "\"height\" must be MOD" + std::to_string(1 << format->subsample_h)
            + " for \"subsample_h\"=" + std::to_string(format->subsample_h) + " !");
    }
    if (!memory)
    {
        throw std::invalid_argument(funcName + "\"memory\" cannot be nullptr!");
    }

    _width = width;
    _height = height;
    _format = format;
    _memory = memory;
    _alignment = ValidAlignment(alignment);

    createData();
}

void Frame::Release()
{
    _width = 0;
    _height = 0;
    _format = NULLPTR;
    _alignment = 0;
    releaseData();
}

bool Frame::empty() const
{
    return _data.empty();
}

void Frame::createData()
{
    releaseData();

    if (_width > 0 && _height > 0)
    {
        if (_format->IsPlanar())
        {
            for (int c = 0; c < _format->channels; ++c)
            {
                int width = _format->type == Format::YUV && c > 0 ? _width >> _format->subsample_w : _width;
                int height = _format->type == Format::YUV && c > 0 ? _height >> _format->subsample_h : _height;
                int stride = static_cast<int>(CalStride<uint8_t>(width * _format->Bps, _alignment));
                _data.push_back(std::make_shared<PlaneData>(width, height, stride, _memory, 0, _alignment));
            }
        }
        else if (_format->IsPacked())
        {
            int width = _width;
            int height = _height;
            int stride = static_cast<int>(CalStride<uint8_t>(width * _format->Bps, _alignment));
            _data.push_back(std::make_shared<PlaneData>(width, height, stride, _memory, 0, _alignment));
        }
        else
        {
            throw std::invalid_argument("Frame::createData(): Unsupported format(id: " + std::to_string(_format->id) + ")!");
        }
    }
}

void Frame::releaseData()
{
    _data.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
