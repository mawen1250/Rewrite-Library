#ifndef RWL_CORE_COMMON_H_
#define RWL_CORE_COMMON_H_

#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <string>
#include "core/type.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace macros

#ifndef RW
#    define RW_BEGIN namespace rw {
#    define RW_END }
#    define RW ::rw::
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// API macros

#if defined(WIN32) || defined(_WIN32) || defined(WINCE)
#    define RW_CDECL __cdecl
#    define RW_STDCALL __stdcall
#else
#    define RW_CDECL
#    define RW_STDCALL
#endif

#if (defined(WIN32) || defined(_WIN32) || defined(WINCE)) && !defined(_WIN64)
#    define RW_CC RW_STDCALL
#else
#    define RW_CC
#endif

#ifndef RW_EXTERN_C
#    ifdef __cplusplus
#        define RW_EXTERN_C extern "C"
#    else
#        define RW_EXTERN_C
#    endif
#endif

#ifndef RW_EXTERN_C_FUNCPTR
#    ifdef __cplusplus
#        define RW_EXTERN_C_FUNCPTR(x) extern "C" { typedef x; }
#    else
#        define RW_EXTERN_C_FUNCPTR(x) typedef x
#    endif
#endif

#if (defined(WIN32) || defined(_WIN32) || defined(WINCE)) && defined(RWAPI_EXPORTS)
#    define RW_EXPORTS __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#    define RW_EXPORTS __attribute__((visibility("default")))
#else
#    define RW_EXPORTS
#endif

#if (defined(WIN32) || defined(_WIN32) || defined(WINCE)) && !defined(RWAPI_EXPORTS)
#    define RWAPI(rettype) RW_EXTERN_C __declspec(dllimport) rettype RW_CC
#else
#    define RWAPI(rettype) RW_EXTERN_C RW_EXPORTS rettype RW_CC
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++11 keyword macros

#ifndef STATIC_ASSERT
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 3))
#define STATIC_ASSERT(_Expression, _Message) assert(_Expression)
#else
#define STATIC_ASSERT static_assert
#endif
#endif

#ifndef DEFAULT
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 4))
#define DEFAULT
#else
#define DEFAULT = default
#endif
#endif

#ifndef DELETE
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 4))
#define DELETE
#else
#define DELETE = delete
#endif
#endif

#ifndef EXPLICIT
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 5))
#define EXPLICIT
#else
#define EXPLICIT explicit
#endif
#endif

#ifndef NULLPTR
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 6))
#define NULLPTR NULL
#else
#define NULLPTR nullptr
#endif
#endif

#ifndef OVERRIDE
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 7))
#define OVERRIDE
#else
#define OVERRIDE override
#endif
#endif

#ifndef FINAL
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ < 4 || (__GUNC__ == 4 && __GNUC_MINOR__ < 7))
#define FINAL
#else
#define FINAL final
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Min Max Clip

template < typename _Ty >
_Ty Min(const _Ty &a, const _Ty &b)
{
    return b < a ? b : a;
}

template < typename _Ty >
_Ty Max(const _Ty &a, const _Ty &b)
{
    return b > a ? b : a;
}

template < typename _Ty >
_Ty Clip(const _Ty &input, const _Ty &lower, const _Ty &upper)
{
    return input <= lower ? lower : input >= upper ? upper : input;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Abs AbsSub

template < typename _Ty >
_Ty Abs(const _Ty &input)
{
    return input < 0 ? -input : input;
}

template < typename _Ty >
_Ty AbsSub(const _Ty &a, const _Ty &b)
{
    return b < a ? a - b : b - a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Division with rounding to integer

template < typename _Ty >
_Ty _RoundDiv(const _Ty &dividend, const _Ty &divisor, const std::false_type &)
{
    return (dividend + divisor / 2) / divisor;
}

template < typename _Ty >
_Ty _RoundDiv(const _Ty &dividend, const _Ty &divisor, const std::true_type &)
{
    return dividend / divisor;
}

template < typename _Ty >
_Ty RoundDiv(const _Ty &dividend, const _Ty &divisor)
{
    return _RoundDiv(dividend, divisor, _IsFloat<_Ty>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bit shift with rounding to integer

template < typename _Ty >
_Ty RoundBitRsh(const _Ty &input, const int &shift)
{
    static_assert(_IsInt<_Ty>::value, "RoundBitRsh: Invalid arguments for template instantiation! Must be integer.");
    return (input + (1 << (shift - 1))) >> shift;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Round - numeric type conversion with up-rounding (float to int) and saturation

// UInt to UInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(const _St1 &input, const std::false_type &, const std::false_type &)
{
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input >= max ? max : input);
}

// UInt to SInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(const _St1 &input, const std::false_type &, const std::true_type &)
{
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input >= max ? max : input);
}

// SInt to UInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(const _St1 &input, const std::true_type &, const std::false_type &)
{
    _St1 min = _St1(0);
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input);
}

// SInt to SInt
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2Int(const _St1 &input, const std::true_type &, const std::true_type &)
{
    _St1 min = _St1(TypeMin<_Dt1>() > TypeMin<_St1>() ? TypeMin<_Dt1>() : TypeMin<_St1>());
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input);
}

// Int to Int
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2(const _St1 &input, const std::false_type &)
{
    return _Round_Int2Int<_Dt1, _St1>(input, _IsSInt<_St1>(), _IsSInt<_Dt1>());
}

// Int to Float
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Int2(const _St1 &input, const std::true_type &)
{
    return static_cast<_Dt1>(input);
}

// Int to Any
template < typename _Dt1, typename _St1 >
_Dt1 _Round(const _St1 &input, const std::false_type &)
{
    return _Round_Int2<_Dt1, _St1>(input, _IsFloat<_Dt1>());
}

// Float to Int
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Float2(const _St1 &input, const std::false_type &)
{
    _St1 min = _St1(TypeMin<_Dt1>());
    _St1 max = _St1(TypeMax<_Dt1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input + _St1(0.5));
}

// Float to Float
template < typename _Dt1, typename _St1 >
_Dt1 _Round_Float2(const _St1 &input, const std::true_type &)
{
    _St1 min = _St1(TypeMin<_Dt1>() > TypeMin<_St1>() ? TypeMin<_Dt1>() : TypeMin<_St1>());
    _St1 max = _St1(TypeMax<_Dt1>() < TypeMax<_St1>() ? TypeMax<_Dt1>() : TypeMax<_St1>());
    return static_cast<_Dt1>(input <= min ? min : input >= max ? max : input);
}

// Float to Any
template < typename _Dt1, typename _St1 >
_Dt1 _Round(const _St1 &input, const std::true_type &)
{
    return _Round_Float2<_Dt1, _St1>(input, _IsFloat<_Dt1>());
}

// Any to Any
template < typename _Dt1, typename _St1 >
_Dt1 Round(const _St1 &input)
{
    return _Round<_Dt1, _St1>(input, _IsFloat<_St1>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
