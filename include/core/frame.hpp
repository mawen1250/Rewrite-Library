#ifndef RWL_CORE_FRAME_HPP_
#define RWL_CORE_FRAME_HPP_

#include "core/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PlaneData

inline PlaneData::PlaneData(PlaneData &&src)
    : _memory(std::move(src._memory)), _data(std::move(src._data)),
    width(src.width), height(src.height), stride(src.stride), Bps(src.Bps), alignment(src.alignment),
    image(std::move(src.image))
{
    src.Release();
}

inline PlaneData &PlaneData::operator=(PlaneData &&src)
{
    _memory = std::move(src._memory);
    _data = std::move(src._data);
    width = src.width;
    height = src.height;
    stride = src.stride;
    Bps = src.Bps;
    alignment = src.alignment;
    image = std::move(src.image);

    src.Release();

    return *this;
}

inline bool PlaneData::operator==(const PlaneData &right) const
{
    if (_memory == right._memory
        && _data == right._data
        && width == right.width
        && height == right.height
        && stride == right.stride
        && Bps == right.Bps
        && alignment == right.alignment
        && image == right.image)
        return true;
    return false;
}

inline bool PlaneData::operator!=(const PlaneData &right) const
{
    return !(*this == right);
}

inline int PlaneData::MemoryType() const
{
    return _memory->Type();
}

inline int PlaneData::MemoryTypeMain() const
{
    return _memory->TypeMain();
}

inline int PlaneData::MemoryTypeSub() const
{
    return _memory->TypeSub();
}

inline bool PlaneData::Continuous() const
{
    return width * Bps == stride;
}

inline bool PlaneData::Empty() const
{
    return !image;
}

inline bool PlaneData::Unique() const
{
    return image.unique();
}

inline int PlaneData::UseCount() const
{
    return image.use_count();
}

inline const DataType *PlaneData::Data() const
{
    return image.get();
}

inline DataType *PlaneData::Data()
{
    return image.get();
}

template <typename _Ty>
const _Ty *PlaneData::Ptr() const
{
    return reinterpret_cast<const _Ty *>(image.get());
}

template <typename _Ty>
_Ty *PlaneData::Ptr()
{
    return reinterpret_cast<_Ty *>(image.get());
}

template <typename _Ty>
const _Ty *PlaneData::Ptr(const int &y) const
{
    return reinterpret_cast<const _Ty *>(image.get() + y * stride);
}

template <typename _Ty>
_Ty *PlaneData::Ptr(const int &y)
{
    return reinterpret_cast<_Ty *>(image.get() + y * stride);
}

template <typename _Ty>
const _Ty *PlaneData::Ptr(const int &y, const int &x) const
{
    return reinterpret_cast<const _Ty *>(image.get() + y * stride + x * Bps);
}

template <typename _Ty>
_Ty *PlaneData::Ptr(const int &y, const int &x)
{
    return reinterpret_cast<_Ty *>(image.get() + y * stride + x * Bps);
}

template <typename _Ty>
const _Ty *PlaneData::Ptr(const int &y, const int &x, const int &c) const
{
    return reinterpret_cast<const _Ty *>(image.get() + y * stride + x * Bps + c * sizeof(_Ty));
}

template <typename _Ty>
_Ty *PlaneData::Ptr(const int &y, const int &x, const int &c)
{
    return reinterpret_cast<_Ty *>(image.get() + y * stride + x * Bps + c * sizeof(_Ty));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Frame

inline Frame::Frame(Frame &&src)
    : _width(src._width), _height(src._height), _format(std::move(src._format)),
    _memory(std::move(src._memory)), _alignment(src._alignment), _data(std::move(src._data))
{
    src.Release();
}

inline Frame &Frame::operator=(Frame &&src)
{
    _width = src._width;
    _height = src._height;
    _format = std::move(src._format);
    _memory = std::move(src._memory);
    _alignment = src._alignment;
    _data = std::move(src._data);

    src.Release();

    return *this;
}

inline bool Frame::operator==(const Frame &right) const
{
    if (_width == right._width
        && _height == right._height
        && _format == right._format
        && _memory == right._memory
        && _alignment == right._alignment
        && _data == right._data)
        return true;
    return false;
}

inline bool Frame::operator!=(const Frame &right) const
{
    return !(*this == right);
}

inline FormatPtr Frame::Format() const
{
    return _format;
}

inline int Frame::MemoryType() const
{
    return _memory->Type();
}

inline int Frame::MemoryTypeMain() const
{
    return _memory->TypeMain();
}

inline int Frame::MemoryTypeSub() const
{
    return _memory->TypeSub();
}

inline bool Frame::Continuous(int plane) const
{
    return _data.at(plane)->Continuous();
}

inline int Frame::Bps() const
{
    return _format->Bps;
}

inline size_t Frame::Alignment() const
{
    return _alignment;
}

inline bool Frame::IsPlanar() const
{
    return _format->IsPlanar();
}

inline bool Frame::IsPacked() const
{
    return _format->IsPacked();
}

inline bool Frame::IsYUV() const
{
    return _format->IsYUV();
}

inline bool Frame::IsRGB() const
{
    return _format->IsRGB();
}

inline bool Frame::IsRGBA() const
{
    return _format->IsRGBA();
}

inline bool Frame::IsChroma(int plane) const
{
    return _format->type == Format::YUV && plane > 0 && plane < 3;
}

inline int Frame::Channels() const
{
    return _format->channels;
}

inline int Frame::Planes() const
{
    return IsPlanar() ? Channels() : IsPacked() ? 1 : 0;
}

inline int Frame::Width(const int &plane) const
{
    return _data.at(plane)->width;
}

inline int Frame::Width() const
{
    return _width;
}

inline int Frame::Height(const int &plane) const
{
    return _data.at(plane)->height;
}

inline int Frame::Height() const
{
    return _height;
}

inline int Frame::Stride(const int &plane) const
{
    return _data.at(plane)->stride;
}

inline int Frame::SubSampleWRatio(const int &channel) const
{
    return channel > 0 && channel < 3 ? 1 << _format->subsample_w : 1;
}

inline int Frame::SubSampleHRatio(const int &channel) const
{
    return channel > 0 && channel < 3 ? 1 << _format->subsample_h : 1;
}

inline bool Frame::Empty(int plane) const
{
    return _data.empty() || !_data.at(plane) || _data.at(plane)->Empty();
}

inline bool Frame::Empty() const
{
    bool empty = true;
    for (int p = 0; p < Planes(); ++p)
    {
        empty &= Empty(p);
    }
    return empty;
}

inline bool Frame::Unique(int plane) const
{
    if (!Empty(plane))
    {
        return _data.at(plane)->Unique();
    }
    return false;
}

inline bool Frame::Unique() const
{
    if (Planes() <= 0) return false;
    bool unique = true;
    for (int p = 0; p < Planes(); ++p)
    {
        unique &= Unique(p);
    }
    return unique;
}

inline int Frame::UseCount(int plane) const
{
    if (!Empty(plane))
    {
        return _data.at(plane)->UseCount();
    }
    return 0;
}

inline const DataType *Frame::Data(const int &plane) const
{
    return _data.at(plane)->Data();
}

inline DataType *Frame::Data(const int &plane)
{
    return _data.at(plane)->Data();
}

template <typename _Ty>
const _Ty *Frame::Ptr(const int &plane) const
{
    return _data.at(plane)->Ptr<_Ty>();
}

template <typename _Ty>
_Ty *Frame::Ptr(const int &plane)
{
    return _data.at(plane)->Ptr<_Ty>();
}

template <typename _Ty>
const _Ty *Frame::Ptr(const int &plane, const int &y) const
{
    return _data.at(plane)->Ptr<_Ty>(y);
}

template <typename _Ty>
_Ty *Frame::Ptr(const int &plane, const int &y)
{
    return _data.at(plane)->Ptr<_Ty>(y);
}

template <typename _Ty>
const _Ty *Frame::Ptr(const int &plane, const int &y, const int &x) const
{
    return _data.at(plane)->Ptr<_Ty>(y, x);
}

template <typename _Ty>
_Ty *Frame::Ptr(const int &plane, const int &y, const int &x)
{
    return _data.at(plane)->Ptr<_Ty>(y, x);
}

template <typename _Ty>
const _Ty *Frame::Ptr(const int &plane, const int &y, const int &x, const int &c) const
{
    return _data.at(plane)->Ptr<_Ty>(y, x, c);
}

template <typename _Ty>
_Ty *Frame::Ptr(const int &plane, const int &y, const int &x, const int &c)
{
    return _data.at(plane)->Ptr<_Ty>(y, x, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
