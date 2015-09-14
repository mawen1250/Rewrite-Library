#ifndef RWL_CORE_COMMON_H_
#define RWL_CORE_COMMON_H_

#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace macros

#ifndef RW
#    define RW_BEGIN namespace rw {
#    define RW_END }
#    define RW ::rw::
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
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
