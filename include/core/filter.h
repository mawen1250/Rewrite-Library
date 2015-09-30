#ifndef RWL_CORE_FILTER_H_
#define RWL_CORE_FILTER_H_

#include <complex>
#include "core/core.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Instantiate and export template classes
// Type definitions

template class RW_EXPORTS std::basic_string<char>;
template class RW_EXPORTS std::vector<int>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FilterBase

class RW_EXPORTS FilterBase
{
public:
    enum IOMode
    {
        D0S0 = 0x00,
        D0S1,
        D0S2,
        D0S3,
        D0S4,
        D1S0 = 0x10,
        D1S1,
        D1S2,
        D1S3,
        D1S4,
        D2S0 = 0x20,
        D2S1,
        D2S2,
        D2S3,
        D2S4,
        D3S0 = 0x30,
        D3S1,
        D3S2,
        D3S3,
        D3S4,
        D4S0 = 0x40,
        D4S1,
        D4S2,
        D4S3,
        D4S4
    };

    const std::string nameSpace;
    const std::string filterName;
    const std::string fullName;

    void operator()(Frame &dst, const Frame &src);
    void Filter(Frame &dst, const Frame &src);

protected:
    FilterBase(const std::string &nameSpace, const std::string &filterName, std::vector<int> modes);
    virtual ~FilterBase() {}

    virtual void filterCheck();
    virtual void filterPrepare();
    virtual void filterProcess();
    virtual void filterPlanar();
    virtual void filterPacked();

    virtual void filterPlanar8U() { filterPlanarT<uint8_t>(); }
    virtual void filterPlanar16U() { filterPlanarT<uint16_t>(); }
    virtual void filterPlanar32U() { filterPlanarT<uint32_t>(); }
    virtual void filterPlanar64U() { filterPlanarT<uint64_t>(); }
    virtual void filterPlanar8I() { filterPlanarT<int8_t>(); }
    virtual void filterPlanar16I() { filterPlanarT<int16_t>(); }
    virtual void filterPlanar32I() { filterPlanarT<int32_t>(); }
    virtual void filterPlanar64I() { filterPlanarT<int64_t>(); }
    virtual void filterPlanar32F() { filterPlanarT<float>(); }
    virtual void filterPlanar64F() { filterPlanarT<double>(); }
    virtual void filterPlanar64C() { filterPlanarT<std::complex<float>>(); }
    virtual void filterPlanar128C() { filterPlanarT<std::complex<double>>(); }

    bool checkMode(int mode, bool throw_ = false, std::string message = std::string()) const;

    std::vector<int> _modes;
    std::shared_ptr<Frame> _dst;
    std::shared_ptr<const Frame> _src;

private:
    template <typename _Ty> void filterPlanarT()
    {
        throw std::invalid_argument(filterName + "::filterPlanar: unsupported format \"" + _src->Format()->Name() + "\"!");
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
