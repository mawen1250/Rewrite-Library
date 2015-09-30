#include "core/filter.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FilterBase

void FilterBase::operator()(Frame &dst, const Frame &src)
{
    Filter(dst, src);
}

void FilterBase::Filter(Frame &dst, const Frame &src)
{
    const std::string funcName = fullName + "::Filter";

    checkMode(D1S1, true, funcName + ": this filter mode (I1O1) is not supported!");

    _src = std::make_shared<const Frame>(src);
    _dst = std::make_shared<Frame>(dst);

    filterCheck();
    filterPrepare();
    filterProcess();
}

FilterBase::FilterBase(const std::string &nameSpace, const std::string &filterName, std::vector<int> modes)
    : nameSpace(nameSpace), filterName(filterName), fullName(nameSpace + "::" + filterName), _modes(modes)
{}

void FilterBase::filterCheck()
{
    const std::string funcName = fullName + "::filterCheck";

    if (_src->Empty())
    {
        throw std::invalid_argument(funcName + ": \"src\" should not be empty!");
    }

}

void FilterBase::filterPrepare()
{
    const std::string funcName = fullName + "::filterPrepare";

    _dst->Create(*_src);
}

void FilterBase::filterProcess()
{
    const std::string funcName = fullName + "::filterProcess";

    if (_src->IsPlanar())
    {
        filterPlanar();
    }
    else if (_src->IsPacked())
    {
        filterPacked();
    }
    else
    {
        throw std::invalid_argument(funcName + ": unsupported format \"" + _src->Format()->Name() + "\"!");
    }
}

void FilterBase::filterPlanar()
{
    const std::string funcName = fullName + "::filterPlanar";

    if (_src->IsPlanar())
    {
        auto format = _src->Format();

        switch (format->sample)
        {
        case Format::UInteger:
            switch (format->Bps)
            {
            case 1:
                filterPlanar8U();
            case 2:
                filterPlanar16U();
            case 4:
                filterPlanar32U();
            case 8:
                filterPlanar64U();
                break;
            default:
                throw std::invalid_argument(funcName + ": unsupported format \"" + format->Name() + "\"!");
            }
            break;
        case Format::Integer:
            switch (format->Bps)
            {
            case 1:
                filterPlanar8I();
            case 2:
                filterPlanar16I();
            case 4:
                filterPlanar32I();
            case 8:
                filterPlanar64I();
                break;
            default:
                throw std::invalid_argument(funcName + ": unsupported format \"" + format->Name() + "\"!");
            }
            break;
        case Format::Float:
            switch (format->Bps)
            {
            case 4:
                filterPlanar32F();
            case 8:
                filterPlanar64F();
                break;
            default:
                throw std::invalid_argument(funcName + ": unsupported format \"" + format->Name() + "\"!");
            }
            break;
        case Format::Complex:
            switch (format->Bps)
            {
            case 8:
                filterPlanar64C();
            case 16:
                filterPlanar128C();
                break;
            default:
                throw std::invalid_argument(funcName + ": unsupported format \"" + format->Name() + "\"!");
            }
            break;
        default:
            throw std::invalid_argument(funcName + ": unsupported sample(=" + std::to_string(format->sample) + ")!");
        }
    }
}

void FilterBase::filterPacked()
{
    throw std::invalid_argument(filterName + "::filterPacked: unsupported format \"" + _src->Format()->Name() + "\"!");
}

bool FilterBase::checkMode(int mode, bool throw_, std::string message) const
{
    bool valid = false;

    for (auto &m : _modes)
    {
        if (m == mode)
        {
            valid = true;
            break;
        }
    }

    if (throw_ && !valid)
    {
        if (message.empty())
        {
            message = fullName + "::checkMode: unsupported filter mode(=" + std::to_string(mode) + ")!";
        }

        throw std::runtime_error(message);
    }

    return valid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
