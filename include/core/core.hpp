#ifndef RWL_CORE_CORE_HPP_
#define RWL_CORE_CORE_HPP_

#include "core/common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make smart pointer

template <typename ..._Types>
FormatPtr MakeFormatPtr(_Types &&...args)
{
    return std::make_shared<const Format>(args...);
}

template <typename ..._Types>
DataPtr MakeDataPtr(_Types &&...args)
{
    return std::make_shared<DataType>(args...);
}

template <typename _Ty, typename ..._Types>
DataMemoryPtr MakeDataMemoryPtr(_Types &&...args)
{
    return std::make_shared<const _Ty>(args...);
}

template <typename ..._Types>
PlaneDataPtr MakePlaneDataPtr(_Types &&...args)
{
    return std::make_shared<PlaneData>(args...);
}

template <typename ..._Types>
FramePtr MakeFramePtr(_Types &&...args)
{
    return std::make_shared<Frame>(args...);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Point

template <typename _Ty>
_Point<_Ty>::_Point()
    : x(0), y(0)
{}

template <typename _Ty>
_Point<_Ty>::_Point(int x, int y)
    : x(x), y(y)
{}

template <typename _Ty>
void _Point<_Ty>::Scale(double scale_x, double scale_y, _Ty offset_x, _Ty offset_y)
{
    x = Round<_Ty>(x * scale_x) + offset_x;
    y = Round<_Ty>(y * scale_y) + offset_y;
}

template <typename _Ty>
_Point<_Ty> operator+(const _Point<_Ty> &left, const _Point<_Ty> &right)
{
    return _Point<_Ty>(left.x + right.x, left.y + right.y);
}

template <typename _Ty>
_Point<_Ty> operator-(const _Point<_Ty> &left, const _Point<_Ty> &right)
{
    return _Point<_Ty>(left.x - right.x, left.y - right.y);
}

template <typename _Ty>
_Point<_Ty> operator*(const _Point<_Ty> &left, const _Ty &right)
{
    return _Point<_Ty>(left.x * right, left.y * right);
}

template <typename _Ty>
_Point<_Ty> operator/(const _Point<_Ty> &left, const _Ty &right)
{
    return _Point<_Ty>(left.x / right, left.y / right);
}

template <typename _Ty>
_Point<_Ty> operator*(const _Ty &left, const _Point<_Ty> &right)
{
    return _Point<_Ty>(left * right.x, left * right.y);
}

template <typename _Ty>
_Point<_Ty> operator/(const _Ty &left, const _Point<_Ty> &right)
{
    return _Point<_Ty>(left / right.x, left / right.y);
}

template <typename _Ty>
_Point<_Ty> &operator+=(_Point<_Ty> &left, const _Point<_Ty> &right)
{
    left.x += right.x;
    left.y += right.y;
    return left;
}

template <typename _Ty>
_Point<_Ty> &operator-=(_Point<_Ty> &left, const _Point<_Ty> &right)
{
    left.x -= right.x;
    left.y -= right.y;
    return left;
}

template <typename _Ty>
_Point<_Ty> &operator*=(_Point<_Ty> &left, const _Ty &right)
{
    left.x *= right;
    left.y *= right;
    return left;
}

template <typename _Ty>
_Point<_Ty> &operator/=(_Point<_Ty> &left, const _Ty &right)
{
    left.x /= right;
    left.y /= right;
    return left;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rect

template <typename _Ty>
_Rect<_Ty>::_Rect()
    : x(0), y(0), width(0), height(0)
{}

template <typename _Ty>
_Rect<_Ty>::_Rect(int x, int y, int width, int height)
    : x(x), y(y), width(width), height(height)
{}

template <typename _Ty>
_Rect<_Ty>::_Rect(const Point &p1, const Point &p2)
    : x(p1.x), y(p1.y), width(p2.x - p1.x), height(p2.y - p1.y)
{}

template <typename _Ty>
bool _Rect<_Ty>::Check(int image_width, int image_height, bool throw_)
{
    static const std::string funcName = "Rect::Check: ";

    if (image_width <= 0 || image_height <= 0)
    {
        throw std::invalid_argument(funcName + "\"image_width\" and \"image_height\" must be positive!");
    }

    if (x < 0 || x >= image_width)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->x\" out of range!");
        else return false;
    }
    if (y < 0 || y >= image_height)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->y\" out of range!");
        else return false;
    }

    if (width <= 0) width += image_width - x;
    if (height <= 0) height += image_height - y;

    if (width <= 0 || x + width > image_width)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->width\" out of range!");
        else return false;
    }
    if (height <= 0 || x + height > image_height)
    {
        if (throw_) throw std::out_of_range(funcName + "\"this->height\" out of range!");
        else return false;
    }

    return true;
}

template <typename _Ty>
void _Rect<_Ty>::Normalize(int image_width, int image_height)
{
    static const std::string funcName = "Rect::Normalize: ";

    if (image_width <= 0 || image_height <= 0)
    {
        throw std::invalid_argument(funcName + "\"image_width\" and \"image_height\" must be positive!");
    }

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= image_width) x = image_width - 1;
    if (y >= image_height) y = image_height - 1;

    if (width <= 0) width += image_width - x;
    if (height <= 0) height += image_height - y;
    if (x + width > image_width) width = image_width - x;
    if (y + height > image_height) height = image_height - y;
}

template <typename _Ty>
void _Rect<_Ty>::Scale(double scale_x, double scale_y, _Ty offset_x, _Ty offset_y)
{
    x = Round<_Ty>(x * scale_x) + offset_x;
    y = Round<_Ty>(y * scale_y) + offset_y;
    width = Round<_Ty>(width * scale_x);
    height = Round<_Ty>(height * scale_y);
}

template <typename _Ty>
_Rect<_Ty> operator+(const _Rect<_Ty> &left, const _Rect<_Ty> &right)
{
    return _Rect<_Ty>(left.x + right.x, left.y + right.y, left.width + right.width, left.height + right.height);
}

template <typename _Ty>
_Rect<_Ty> operator-(const _Rect<_Ty> &left, const _Rect<_Ty> &right)
{
    return _Rect<_Ty>(left.x - right.x, left.y - right.y, left.width - right.width, left.height - right.height);
}

template <typename _Ty>
_Rect<_Ty> operator*(const _Rect<_Ty> &left, const _Ty &right)
{
    return _Rect<_Ty>(left.x * right, left.y * right, left.width * right, left.height * right);
}

template <typename _Ty>
_Rect<_Ty> operator/(const _Rect<_Ty> &left, const _Ty &right)
{
    return _Rect<_Ty>(left.x / right, left.y / right, left.width / right, left.height / right);
}

template <typename _Ty>
_Rect<_Ty> operator*(const _Ty &left, const _Rect<_Ty> &right)
{
    return _Rect<_Ty>(left * right.x, left * right.y, left * right.width, left * right.height);
}

template <typename _Ty>
_Rect<_Ty> operator/(const _Ty &left, const _Rect<_Ty> &right)
{
    return _Rect<_Ty>(left / right.x, left / right.y, left / right.width, left / right.height);
}

template <typename _Ty>
_Rect<_Ty> &operator+=(_Rect<_Ty> &left, const _Rect<_Ty> &right)
{
    left.x += right.x;
    left.y += right.y;
    left.width += right.width;
    left.height += right.height;
    return left;
}

template <typename _Ty>
_Rect<_Ty> &operator-=(_Rect<_Ty> &left, const _Rect<_Ty> &right)
{
    left.x -= right.x;
    left.y -= right.y;
    left.width -= right.width;
    left.height -= right.height;
    return left;
}

template <typename _Ty>
_Rect<_Ty> &operator*=(_Rect<_Ty> &left, const _Ty &right)
{
    left.x *= right;
    left.y *= right;
    left.width *= right;
    left.height *= right;
    return left;
}

template <typename _Ty>
_Rect<_Ty> &operator/=(_Rect<_Ty> &left, const _Ty &right)
{
    left.x /= right;
    left.y /= right;
    left.width /= right;
    left.height /= right;
    return left;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
