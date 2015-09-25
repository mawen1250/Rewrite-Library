#include "core/core.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PlaneData

PlaneData::PlaneData(int width_, int height_, int stride_, int Bps_, DataMemoryPtr memory_, ptrdiff_t offset_, size_t alignment_)
{
    Create(width_, height_, stride_, Bps_, std::move(memory_), offset_, alignment);
}

PlaneData::PlaneData(void *data, int width, int height, int stride, int Bps, DataMemoryPtr memory, ptrdiff_t offset, size_t alignment)
    : width(width), height(height), stride(stride), Bps(Bps), alignment(ValidAlignment(alignment, stride)),
    _memory(memory), _data(reinterpret_cast<DataType *>(data), memory->dealloc),
    image(_data, reinterpret_cast<DataType *>(data) + offset)
{
    image2align();
}

PlaneData::PlaneData(const PlaneData &ref, Rect roi)
    : width(roi.width), height(roi.height), stride(ref.stride), Bps(ref.Bps), alignment(ref.alignment),
    _memory(ref._memory), _data(ref._data), image()
{
    roi.Check(ref.width, ref.height, true);
    ptrdiff_t offset = roi.y * stride + roi.x * Bps;
    image = DataPtr(_data, ref.image.get() + offset); // aliasing constructor
    image2align();
}

PlaneData::PlaneData(PlaneDataPtr ref)
    : PlaneData(*ref)
{}

PlaneData::PlaneData(PlaneDataPtr ref, Rect roi)
    : PlaneData(*ref, roi)
{}

void PlaneData::Create(int width_, int height_, int stride_, int Bps_, DataMemoryPtr memory_, ptrdiff_t offset_, size_t alignment_)
{
    width = width_;
    height = height_;
    stride = stride_;
    Bps = Bps_;
    _memory = std::move(memory_);
    alignment = alignment_;

    createData(offset_);
}

void PlaneData::Release()
{
    _memory = NULLPTR;
    width = 0;
    height = 0;
    stride = 0;
    Bps = 0;
    alignment = 0;

    releaseData();
}

PlaneDataPtr PlaneData::GetPtr() const
{
    return MakePlaneDataPtr(*this);
}

PlaneData PlaneData::Clone() const
{
    PlaneData dst(width, height, stride, Bps, _memory, alignment, getOffset());
    BitBlt(dst.image.get(), image.get(), dst.height, dst.width * dst.Bps, dst.stride, stride);
    return dst;
}

PlaneDataPtr PlaneData::ClonePtr() const
{
    return MakePlaneDataPtr(Clone());
}

void PlaneData::createData(ptrdiff_t offset)
{
    static const std::string funcName = "PlaneData::createData: ";

    if (width < 0 || height < 0)
    {
        throw std::invalid_argument(funcName + "\"width\" and \"height\" cannot be negative!");
    }
    if (Bps < 1)
    {
        throw std::invalid_argument(funcName + "\"Bps\" cannot be less than 1!");
    }
    if (stride < width * Bps)
    {
        throw std::invalid_argument(funcName + "\"stride\" cannot be less than \"width * Bps\"!");
    }

    alignment = ValidAlignment(alignment, stride);
    releaseData();

    DataType *alloc_mem = _memory->alloc(width * Bps, height, stride, alignment);
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

void PlaneData::releaseData()
{
    image = NULLPTR;
    _data = NULLPTR;
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
    : _width(0), _height(0), _format(fGray8U), _memory(dmNone), _alignment(MEMORY_ALIGNMENT), _data()
{}

Frame::Frame(int width, int height, FormatPtr format, DataMemoryPtr memory, size_t alignment)
{
    Create(width, height, std::move(format), std::move(memory), alignment);
}

Frame::Frame(const Frame &ref, Rect roi)
    : _width(ref._width), _height(ref._height), _format(ref._format),
    _memory(ref._memory), _alignment(ref._alignment), _data()
{
    for (int p = 0; p < Planes(); ++p)
    {
        Rect roi2(roi.x / SubSampleWRatio(p), roi.y / SubSampleHRatio(p),
            roi.width / SubSampleWRatio(p), roi.height / SubSampleHRatio(p));
        _data.push_back(MakePlaneDataPtr(ref._data.at(p), roi2));
    }
}

Frame::Frame(FramePtr ref)
    : Frame(*ref)
{}

Frame::Frame(FramePtr ref, Rect roi)
    : Frame(*ref, roi)
{}

void Frame::Create(int width, int height, FormatPtr format, DataMemoryPtr memory, size_t alignment)
{
    _width = width;
    _height = height;
    _format = std::move(format);
    _memory = std::move(memory);
    _alignment = alignment;

    createData();
}

void Frame::Release()
{
    _width = 0;
    _height = 0;
    _format = NULLPTR;
    _memory = NULLPTR;
    _alignment = 0;

    releaseData();
}

FramePtr Frame::GetPtr() const
{
    return MakeFramePtr(*this);
}

Frame Frame::Clone() const
{
    Frame dst(*this);
    for (int p = 0; p < Planes(); ++p)
    {
        dst._data.at(p) = _data.at(p)->ClonePtr();
    }
    return dst;
}

FramePtr Frame::ClonePtr() const
{
    return MakeFramePtr(Clone());
}

void Frame::createData()
{
    static const std::string funcName = "Frame::createData: ";

    if (_width < 0 || _height < 0)
    {
        throw std::invalid_argument(funcName + "\"width\" and \"height\" cannot be negative!");
    }
    if (!(_format = formatManager(_format)))
    {
        throw std::invalid_argument(funcName + "\"format\" cannot be nullptr!");
    }
    if (_width > 0 && _width % SubSampleWRatio())
    {
        throw std::invalid_argument(funcName + "\"width\" must be MOD" + std::to_string(SubSampleWRatio())
            + " for \"subsample_w\"=" + std::to_string(_format->subsample_w) + " !");
    }
    if (_height > 0 && _height % SubSampleHRatio())
    {
        throw std::invalid_argument(funcName + "\"height\" must be MOD" + std::to_string(SubSampleHRatio())
            + " for \"subsample_h\"=" + std::to_string(_format->subsample_h) + " !");
    }
    if (!_memory)
    {
        throw std::invalid_argument(funcName + "\"memory\" cannot be nullptr!");
    }

    _alignment = ValidAlignment(_alignment);
    releaseData();

    if (_width > 0 && _height > 0)
    {
        if (IsPlanar())
        {
            for (int p = 0; p < Planes(); ++p)
            {
                int width = IsChroma(p) ? _width >> _format->subsample_w : _width;
                int height = IsChroma(p) ? _height >> _format->subsample_h : _height;
                int stride = static_cast<int>(CalStride<uint8_t>(width * Bps(), _alignment));
                _data.push_back(MakePlaneDataPtr(width, height, stride, Bps(), _memory, 0, _alignment));
            }
        }
        else if (IsPacked())
        {
            int width = _width;
            int height = _height;
            int stride = static_cast<int>(CalStride<uint8_t>(width * Bps(), _alignment));
            _data.push_back(MakePlaneDataPtr(width, height, stride, Bps(), _memory, 0, _alignment));
        }
        else
        {
            throw std::invalid_argument(funcName + "Unsupported format(id: " + std::to_string(_format->id) + ")!");
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
