#include "core/core.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw begin

RW_BEGIN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Format

Format::Format()
    : type(Gray), channels(1), subsample_w(0), subsample_h(0), Bps(1), bps(8), sample(UInteger)
{
    id = Format2ID(*this);
}

bool Format::IsPlanar() const
{
    return type >= 0x00 && type < 0x80;
}

bool Format::IsPacked() const
{
    return type >= 0x80 && type < 0xff;
}

bool Format::IsYUV() const
{
    switch (type)
    {
    case YUV:
    case YUYV:
    case YVYU:
    case UYVY:
    case VYUY:
    case AYUV:
        return true;
    default:
        return false;
    }
}

bool Format::IsRGB() const
{
    switch (type)
    {
    case RGB:
    case RGB24:
    case RGB32:
    case GBR24:
    case GBR32:
    case BGR24:
    case BGR32:
        return true;
    default:
        return false;
    }
}

bool Format::IsRGBA() const
{
    switch (type)
    {
    case RGBA:
    case RGBA32:
    case GBRA32:
    case BGRA32:
        return true;
    default:
        return false;
    }
}

std::string Format::Name() const
{
    static const std::string funcName = "Format::Name";

    if (IsPlanar())
    {
        std::string name;

        switch (type)
        {
        case Gray:
            name += "Gray";
            break;
        case RGB:
            name += "RGB";
            break;
        case RGBA:
            name += "RGBA";
            break;
        case YUV:
            name += "YUV";
            if (subsample_w == 0 && subsample_h == 0)
            {
                name += "444P";
            }
            else if (subsample_w == 1 && subsample_h == 0)
            {
                name += "422P";
            }
            else if (subsample_w == 1 && subsample_h == 1)
            {
                name += "420P";
            }
            else if (subsample_w == 2 && subsample_h == 0)
            {
                name += "411P";
            }
            else if (subsample_w == 2 && subsample_h == 1)
            {
                name += "410P";
            }
            else
            {
                name += "ssw" + std::to_string(subsample_w) + "ssh" + std::to_string(subsample_h) + "P";
            }
            break;
        default:
            throw std::invalid_argument(funcName + ": unrecognized Planar type!");
        }

        name += std::to_string(bps);

        switch (sample)
        {
        case UInteger:
            name += "U";
            break;
        case Integer:
            name += "I";
            break;
        case Float:
            name += "F";
            break;
        case Complex:
            name += "C";
            break;
        default:
            throw std::invalid_argument(funcName + ": unrecognized sample!");
        }

        return name;
    }
    else if (IsPacked())
    {
        switch (type)
        {
        case RGB24:
            return "RGB24";
        case RGB32:
            return "RGB32";
        case RGBA32:
            return "RGBA32";
        case GBR24:
            return "GBR24";
        case GBR32:
            return "GBR32";
        case GBRA32:
            return "GBRA32";
        case BGR24:
            return "BGR24";
        case BGR32:
            return "BGR32";
        case BGRA32:
            return "BGRA32";
        case YUYV:
            return "YUYV";
        case YVYU:
            return "YVYU";
        case UYVY:
            return "UYVY";
        case VYUY:
            return "VYUY";
        case AYUV:
            return "AYUV";
        default:
            throw std::invalid_argument(funcName + ": unrecognized Packed type!");
        }
    }
    else if (type == Undef)
    {
        return "Undefined";
    }
    else
    {
        throw std::invalid_argument(funcName + ": unrecognized type!");
    }

    return std::string();
}

void Format::TypeRestrict(Format &format, int type)
{
    if (type < 0) type = format.type;
    else format.type = type;

    switch (type)
    {
    // Planar formats
    case Gray:
        format.channels = 1;
        format.subsample_w = 0;
        format.subsample_h = 0;
        break;
    case RGB:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        break;
    case RGBA:
        format.channels = 4;
        format.subsample_w = 0;
        format.subsample_h = 0;
        break;
    case YUV:
        format.channels = 3;
        break;
    // Packed formats
    case RGB24:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 3;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case RGB32:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case RGBA32:
        format.channels = 4;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case GBR24:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 3;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case GBR32:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case GBRA32:
        format.channels = 4;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case BGR24:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 3;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case BGR32:
        format.channels = 3;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case BGRA32:
        format.channels = 4;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case YUYV:
        format.channels = 3;
        format.subsample_w = 1;
        format.subsample_h = 0;
        format.Bps = 2;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case YVYU:
        format.channels = 3;
        format.subsample_w = 1;
        format.subsample_h = 0;
        format.Bps = 2;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case UYVY:
        format.channels = 3;
        format.subsample_w = 1;
        format.subsample_h = 0;
        format.Bps = 2;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case VYUY:
        format.channels = 3;
        format.subsample_w = 1;
        format.subsample_h = 0;
        format.Bps = 2;
        format.bps = 8;
        format.sample = UInteger;
        break;
    case AYUV:
        format.channels = 4;
        format.subsample_w = 0;
        format.subsample_h = 0;
        format.Bps = 4;
        format.bps = 8;
        format.sample = UInteger;
        break;
    default:
        break;
    }
}

int Format::bps2Bps(int bps)
{
    int Bps = 1;

    while (Bps * 8 < bps)
    {
        Bps *= 2;
    }

    return Bps;
}

int Format::Format2ID(const Format &format)
{
    int id = 0;

    id += typeMask & (format.type << typeShift);
    id += sswMask & (format.subsample_w << sswShift);
    id += sshMask & (format.subsample_h << sshShift);
    id += BpsMask & (format.Bps << BpsShift);
    id += bpsMask & (format.bps << bpsShift);
    id += sampleMask & (format.sample << sampleShift);

    return id;
}

Format Format::ID2Format(int id)
{
    Format format;

    format.type = (typeMask & id) >> typeShift;
    format.subsample_w = (sswMask & id) >> sswShift;
    format.subsample_h = (sshMask & id) >> sshShift;
    format.Bps = (BpsMask & id) >> BpsShift;
    format.bps = (bpsMask & id) >> bpsShift;
    format.sample = (sampleMask & id) >> sampleShift;

    return format;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FormatManager

FormatPtr FormatManager::operator()(FormatPtr format)
{
    if (!format || IsRegistered(format))
    {
        return format;
    }

    return RegisterFormat(format->type, format->bps, format->sample, format->subsample_w, format->subsample_h);
}

FormatPtr FormatManager::operator()(int type, int bps, int sample,
    int subsample_w, int subsample_h)
{
    return RegisterFormat(type, bps, sample, subsample_w, subsample_h);
}

FormatPtr FormatManager::RegisterFormat(int type, int bps, int sample,
    int subsample_w, int subsample_h)
{
    static const std::string funcName = "FormatManager::RegisterFormat: ";

    if (type != Format::Gray && type != Format::RGB && type != Format::RGBA && type != Format::YUV)
    {
        throw std::invalid_argument(funcName + "Invalid \"type\" specified!");
    }
    if (bps <= 0 || bps > 64)
    {
        throw std::invalid_argument(funcName + "Invalid \"bps\" specified! Should be range in (0, 64].");
    }
    if (sample != Format::UInteger && sample != Format::Integer && sample != Format::Float && sample != Format::Complex)
    {
        throw std::invalid_argument(funcName + "Invalid \"sample\" specified!");
    }

    Format format;

    format.type = type;
    format.bps = bps < 0 ? 8 : bps;
    format.sample = sample < 0 ? bps < 32 ? Format::UInteger : bps < 128 ? Format::Float : Format::Complex : sample;
    format.subsample_w = subsample_w < 0 ? 0 : subsample_w;
    format.subsample_h = subsample_h < 0 ? 0 : subsample_h;
    format.Bps = Format::bps2Bps(format.bps);
    Format::TypeRestrict(format);
    format.id = Format::Format2ID(format);

    if (_formats.count(format.id))
    {
        return _formats.at(format.id);
    }
    else
    {
        FormatPtr p_format = MakeFormatPtr(format);
        _formats.insert(std::make_pair(format.id, p_format));
        return p_format;
    }
}

bool FormatManager::IsRegistered(const FormatPtr &format) const
{
    if (format)
    {
        if (_formats.count(format->id))
        {
            if (_formats.at(format->id).get() == format.get())
            {
                return true;
            }
        }
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Preset formats

FormatManager formatManager;

const FormatPtr fGray8U = formatManager.RegisterFormat(Format::Gray, 8, Format::UInteger, 0, 0);
const FormatPtr fGray16U = formatManager.RegisterFormat(Format::Gray, 16, Format::UInteger, 0, 0);
const FormatPtr fGray16I = formatManager.RegisterFormat(Format::Gray, 16, Format::Integer, 0, 0);
const FormatPtr fGray32I = formatManager.RegisterFormat(Format::Gray, 32, Format::Integer, 0, 0);
const FormatPtr fGray32F = formatManager.RegisterFormat(Format::Gray, 32, Format::Float, 0, 0);
const FormatPtr fGray64F = formatManager.RegisterFormat(Format::Gray, 64, Format::Float, 0, 0);
const FormatPtr fGray64C = formatManager.RegisterFormat(Format::Gray, 64, Format::Complex, 0, 0);
const FormatPtr fGray128C = formatManager.RegisterFormat(Format::Gray, 128, Format::Complex, 0, 0);

const FormatPtr fRGB8U = formatManager.RegisterFormat(Format::RGB, 8, Format::UInteger, 0, 0);
const FormatPtr fRGB16U = formatManager.RegisterFormat(Format::RGB, 16, Format::UInteger, 0, 0);
const FormatPtr fRGB16I = formatManager.RegisterFormat(Format::RGB, 16, Format::Integer, 0, 0);
const FormatPtr fRGB32I = formatManager.RegisterFormat(Format::RGB, 32, Format::Integer, 0, 0);
const FormatPtr fRGB32F = formatManager.RegisterFormat(Format::RGB, 32, Format::Float, 0, 0);
const FormatPtr fRGB64F = formatManager.RegisterFormat(Format::RGB, 64, Format::Float, 0, 0);
const FormatPtr fRGB64C = formatManager.RegisterFormat(Format::RGB, 64, Format::Complex, 0, 0);
const FormatPtr fRGB128C = formatManager.RegisterFormat(Format::RGB, 128, Format::Complex, 0, 0);

const FormatPtr fYUV444P8U = formatManager.RegisterFormat(Format::YUV, 8, Format::UInteger, 0, 0);
const FormatPtr fYUV444P16U = formatManager.RegisterFormat(Format::YUV, 16, Format::UInteger, 0, 0);
const FormatPtr fYUV444P16I = formatManager.RegisterFormat(Format::YUV, 16, Format::Integer, 0, 0);
const FormatPtr fYUV444P32I = formatManager.RegisterFormat(Format::YUV, 32, Format::Integer, 0, 0);
const FormatPtr fYUV444P32F = formatManager.RegisterFormat(Format::YUV, 32, Format::Float, 0, 0);
const FormatPtr fYUV444P64F = formatManager.RegisterFormat(Format::YUV, 64, Format::Float, 0, 0);
const FormatPtr fYUV444P64C = formatManager.RegisterFormat(Format::YUV, 64, Format::Complex, 0, 0);
const FormatPtr fYUV444P128C = formatManager.RegisterFormat(Format::YUV, 128, Format::Complex, 0, 0);

const FormatPtr fYUV422P8U = formatManager.RegisterFormat(Format::YUV, 8, Format::UInteger, 1, 0);
const FormatPtr fYUV422P16U = formatManager.RegisterFormat(Format::YUV, 16, Format::UInteger, 1, 0);
const FormatPtr fYUV422P16I = formatManager.RegisterFormat(Format::YUV, 16, Format::Integer, 1, 0);
const FormatPtr fYUV422P32I = formatManager.RegisterFormat(Format::YUV, 32, Format::Integer, 1, 0);
const FormatPtr fYUV422P32F = formatManager.RegisterFormat(Format::YUV, 32, Format::Float, 1, 0);
const FormatPtr fYUV422P64F = formatManager.RegisterFormat(Format::YUV, 64, Format::Float, 1, 0);
const FormatPtr fYUV422P64C = formatManager.RegisterFormat(Format::YUV, 64, Format::Complex, 1, 0);
const FormatPtr fYUV422P128C = formatManager.RegisterFormat(Format::YUV, 128, Format::Complex, 1, 0);

const FormatPtr fYUV420P8U = formatManager.RegisterFormat(Format::YUV, 8, Format::UInteger, 1, 1);
const FormatPtr fYUV420P16U = formatManager.RegisterFormat(Format::YUV, 16, Format::UInteger, 1, 1);
const FormatPtr fYUV420P16I = formatManager.RegisterFormat(Format::YUV, 16, Format::Integer, 1, 1);
const FormatPtr fYUV420P32I = formatManager.RegisterFormat(Format::YUV, 32, Format::Integer, 1, 1);
const FormatPtr fYUV420P32F = formatManager.RegisterFormat(Format::YUV, 32, Format::Float, 1, 1);
const FormatPtr fYUV420P64F = formatManager.RegisterFormat(Format::YUV, 64, Format::Float, 1, 1);
const FormatPtr fYUV420P64C = formatManager.RegisterFormat(Format::YUV, 64, Format::Complex, 1, 1);
const FormatPtr fYUV420P128C = formatManager.RegisterFormat(Format::YUV, 128, Format::Complex, 1, 1);

const FormatPtr fYUV411P8U = formatManager.RegisterFormat(Format::YUV, 8, Format::UInteger, 2, 0);
const FormatPtr fYUV411P16U = formatManager.RegisterFormat(Format::YUV, 16, Format::UInteger, 2, 0);
const FormatPtr fYUV411P16I = formatManager.RegisterFormat(Format::YUV, 16, Format::Integer, 2, 0);
const FormatPtr fYUV411P32I = formatManager.RegisterFormat(Format::YUV, 32, Format::Integer, 2, 0);
const FormatPtr fYUV411P32F = formatManager.RegisterFormat(Format::YUV, 32, Format::Float, 2, 0);
const FormatPtr fYUV411P64F = formatManager.RegisterFormat(Format::YUV, 64, Format::Float, 2, 0);
const FormatPtr fYUV411P64C = formatManager.RegisterFormat(Format::YUV, 64, Format::Complex, 2, 0);
const FormatPtr fYUV411P128C = formatManager.RegisterFormat(Format::YUV, 128, Format::Complex, 2, 0);

const FormatPtr fYUV410P8U = formatManager.RegisterFormat(Format::YUV, 8, Format::UInteger, 2, 1);
const FormatPtr fYUV410P16U = formatManager.RegisterFormat(Format::YUV, 16, Format::UInteger, 2, 1);
const FormatPtr fYUV410P16I = formatManager.RegisterFormat(Format::YUV, 16, Format::Integer, 2, 1);
const FormatPtr fYUV410P32I = formatManager.RegisterFormat(Format::YUV, 32, Format::Integer, 2, 1);
const FormatPtr fYUV410P32F = formatManager.RegisterFormat(Format::YUV, 32, Format::Float, 2, 1);
const FormatPtr fYUV410P64F = formatManager.RegisterFormat(Format::YUV, 64, Format::Float, 2, 1);
const FormatPtr fYUV410P64C = formatManager.RegisterFormat(Format::YUV, 64, Format::Complex, 2, 1);
const FormatPtr fYUV410P128C = formatManager.RegisterFormat(Format::YUV, 128, Format::Complex, 2, 1);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Namespace rw end

RW_END

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
