#ifndef RWL_CORE_TYPE_H_
#define RWL_CORE_TYPE_H_

#include <cstdint>
#include <cfloat>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// determine whether _Ty satisfies Signed Int requirements
template < typename _Ty >
struct _IsSInt
    : std::integral_constant<bool, std::is_same<_Ty, signed char>::value
    || std::is_same<_Ty, short>::value
    || std::is_same<_Ty, int>::value
    || std::is_same<_Ty, long>::value
    || std::is_same<_Ty, long long>::value>
{};

// determine whether _Ty satisfies Unsigned Int requirements
template < typename _Ty >
struct _IsUInt
    : std::integral_constant<bool, std::is_same<_Ty, unsigned char>::value
    || std::is_same<_Ty, unsigned short>::value
    || std::is_same<_Ty, unsigned int>::value
    || std::is_same<_Ty, unsigned long>::value
    || std::is_same<_Ty, unsigned long long>::value>
{};

// determine whether _Ty satisfies Int requirements
template < typename _Ty >
struct _IsInt
    : std::integral_constant<bool, _IsSInt<_Ty>::value
    || _IsUInt<_Ty>::value>
{};

// determine whether _Ty satisfies Float requirements
template < typename _Ty >
struct _IsFloat
    : std::integral_constant<bool, std::is_same<_Ty, float>::value
    || std::is_same<_Ty, double>::value
    || std::is_same<_Ty, long double>::value>
{};

#define isSInt(T) (_IsSInt<T>::value)
#define isUInt(T) (_IsUInt<T>::value)
#define isInt(T) (_IsInt<T>::value)
#define isFloat(T) (_IsFloat<T>::value)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Min value and Max value of each numeric type

#define IntMin(type) (type(sizeof(type) <= 1 ? INT8_MIN : sizeof(type) <= 2 ? INT16_MIN : sizeof(type) <= 4 ? INT32_MIN : INT64_MIN))
#define IntMax(type) (type(sizeof(type) <= 1 ? INT8_MAX : sizeof(type) <= 2 ? INT16_MAX : sizeof(type) <= 4 ? INT32_MAX : INT64_MAX))
#define UIntMin(type) (type(0))
#define UIntMax(type) (type(sizeof(type) <= 1 ? UINT8_MAX : sizeof(type) <= 2 ? UINT16_MAX : sizeof(type) <= 4 ? UINT32_MAX : UINT64_MAX))
#define FltMin(type) (type(sizeof(type) <= 4 ? FLT_MIN : sizeof(type) <= 8 ? DBL_MIN : LDBL_MIN))
#define FltMax(type) (type(sizeof(type) <= 4 ? FLT_MAX : sizeof(type) <= 8 ? DBL_MAX : LDBL_MAX))
#define FltNegMax(type) (type(sizeof(type) <= 4 ? -FLT_MAX : sizeof(type) <= 8 ? -DBL_MAX : -LDBL_MAX))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename _Ty >
_Ty _TypeMinInt(const std::false_type &)
{
    return UIntMin(_Ty);
}

template < typename _Ty >
_Ty _TypeMinInt(const std::true_type &)
{
    return IntMin(_Ty);
}

template < typename _Ty >
_Ty _TypeMin(const std::false_type &)
{
    return _TypeMinInt<_Ty>(_IsSInt<_Ty>());
}

template < typename _Ty >
_Ty _TypeMin(const std::true_type &)
{
    return FltNegMax(_Ty);
}

template < typename _Ty >
_Ty TypeMin()
{
    return _TypeMin<_Ty>(_IsFloat<_Ty>());
}

template < typename _Ty >
_Ty TypeMin(const _Ty &)
{
    return TypeMin<_Ty>();
}


template < typename _Ty >
_Ty _TypeMaxInt(const std::false_type &)
{
    return UIntMax(_Ty);
}

template < typename _Ty >
_Ty _TypeMaxInt(const std::true_type &)
{
    return IntMax(_Ty);
}

template < typename _Ty >
_Ty _TypeMax(const std::false_type &)
{
    return _TypeMaxInt<_Ty>(_IsSInt<_Ty>());
}

template < typename _Ty >
_Ty _TypeMax(const std::true_type &)
{
    return FltMax(_Ty);
}

template < typename _Ty >
_Ty TypeMax()
{
    return _TypeMax<_Ty>(_IsFloat<_Ty>());
}

template < typename _Ty >
_Ty TypeMax(const _Ty &)
{
    return TypeMax<_Ty>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename _Ty1, typename _Ty2 >
struct KeyPair
    : public std::pair<_Ty1, _Ty2>
{
    typedef KeyPair<_Ty1, _Ty2> _Myt;
    typedef std::pair<_Ty1, _Ty2> _Mybase;

    typedef _Ty1 KeyType;
    typedef _Ty2 ValType;

    KeyPair()
        : _Mybase()
    {}

    KeyPair(const _Ty1& _Val1, const _Ty2& _Val2)
        : _Mybase(_Val1, _Val2)
    {}

    KeyPair(const _Myt &_Right)
        : _Mybase(_Right)
    {}

    KeyPair(_Myt &&_Right)
        : _Mybase(_Right)
    {}

    _Myt &operator=(const _Myt &_Right)
    {
        _Mybase::operator=(_Right);
        return *this;
    }

    _Myt &operator=(_Myt &&_Right)
    {
        _Mybase::operator=(_Right);
        return *this;
    }

    bool operator==(const _Myt &_Right)
    {
        return this->first == _Right.first;
    }

    bool operator!=(const _Myt &_Right)
    {
        return this->first != _Right.first;
    }

    bool operator<(const _Myt &_Right)
    {
        return this->first < _Right.first;
    }

    bool operator>(const _Myt &_Right)
    {
        return this->first > _Right.first;
    }

    bool operator<=(const _Myt &_Right)
    {
        return this->first <= _Right.first;
    }

    bool operator>=(const _Myt &_Right)
    {
        return this->first >= _Right.first;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
