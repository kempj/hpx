//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/guid_initialization.hpp>
#include <hpx/util/void_cast.hpp>

#include <hpx/plugins/compression/snappy_serialization_filter.hpp>

#include <boost/format.hpp>

#include <snappy.h>

///////////////////////////////////////////////////////////////////////////////
HPX_SERIALIZATION_REGISTER_TYPE_DEFINITION(
    hpx::plugins::compression::snappy_serialization_filter);
HPX_REGISTER_BASE_HELPER(
    hpx::plugins::compression::snappy_serialization_filter,
    snappy_serialization_filter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace compression
{
    snappy_serialization_filter::~snappy_serialization_filter()
    {
        hpx::actions::detail::guid_initialization<snappy_serialization_filter>();
    }

    void snappy_serialization_filter::register_base()
    {
        util::void_cast_register_nonvirt<
            snappy_serialization_filter, util::binary_filter>();
    }

    void snappy_serialization_filter::set_max_compression_length(std::size_t size)
    {
        buffer_.reserve(size);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t snappy_serialization_filter::init_decompression_data(
        char const* buffer, std::size_t size, std::size_t decompressed_size)
    {
        buffer_.resize(decompressed_size);
        snappy::RawUncompress(buffer, size, buffer_.data());
        current_ = 0;
        return buffer_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    void snappy_serialization_filter::load(void* dst, std::size_t dst_count)
    {
        if (current_+dst_count > buffer_.size())
        {
            BOOST_THROW_EXCEPTION(
                boost::archive::archive_exception(
                    boost::archive::archive_exception::input_stream_error,
                    "archive data bstream is too short"));
            return;
        }

        std::memcpy(dst, &buffer_[current_], dst_count);
        current_ += dst_count;
    }

    ///////////////////////////////////////////////////////////////////////////
    void snappy_serialization_filter::save(void const* src,
        std::size_t src_count)
    {
        char const* src_begin = static_cast<char const*>(src);
        std::copy(src_begin, src_begin+src_count, std::back_inserter(buffer_));
    }

    ///////////////////////////////////////////////////////////////////////////
    bool snappy_serialization_filter::flush(void* dst, std::size_t dst_count,
        std::size_t& written)
    {
        // make sure we have enough memory
        std::size_t needed = snappy::MaxCompressedLength(buffer_.size());
        if (needed > dst_count)
        {
            written = 0;
            return false;
        }

        // compress everything in one go
        char* dst_begin = static_cast<char*>(dst);
        char const* src_begin = buffer_.data();
        size_t compressed_length = 0;
        snappy::RawCompress(src_begin, buffer_.size(), dst_begin,
            &compressed_length);

        if (compressed_length > dst_count)
        {
            HPX_THROW_EXCEPTION(serialization_error,
                "snappy_serialization_filter::flush",
                "compression failure, flushing did not reach end of data");
            return false;
        }

        written = compressed_length;
        return true;
    }
}}}
