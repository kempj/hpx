//  Copyright (C) 2012 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <utility>
#include <memory>
#include <string>

#include <boost/move/move.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return 42;
}

void test_wait_for_two_out_of_five_futures()
{
    unsigned const count = 2;

    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2();
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    pt4();
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();

    typedef hpx::when_some_result<hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > > result_type;
    hpx::lcos::future<result_type> r = hpx::when_some(count, f1, f2, f3, f4, f5);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST_EQ(result.indices.size(), count);
    HPX_TEST(!hpx::util::get<0>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<1>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<2>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<3>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<4>(result.futures).is_ready());
}

void test_wait_for_three_out_of_five_futures()
{
    unsigned const count = 3;

    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1();
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3();
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();
    pt5();

    typedef hpx::when_some_result<hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > > result_type;
    hpx::lcos::future<result_type> r = hpx::when_some(count, f1, f2, f3, f4, f5);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST_EQ(result.indices.size(), count);
    HPX_TEST(hpx::util::get<0>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<1>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<2>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<3>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<4>(result.futures).is_ready());
}

void test_wait_for_two_out_of_five_late_futures()
{
    unsigned const count = 2;

    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();

    typedef hpx::when_some_result<hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > > result_type;
    hpx::lcos::future<result_type> r = hpx::when_some(count, f1, f2, f3, f4, f5);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    pt2();
    pt4();

    result_type result = r.get();

    HPX_TEST_EQ(result.indices.size(), count);
    HPX_TEST(!hpx::util::get<0>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<1>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<2>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<3>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<4>(result.futures).is_ready());
}

void test_wait_for_two_out_of_five_deferred_futures()
{
    unsigned const count = 2;

    hpx::lcos::future<int> f1 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::lcos::future<int> f2 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::lcos::future<int> f3 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::lcos::future<int> f4 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::lcos::future<int> f5 = hpx::async(hpx::launch::deferred, &make_int_slowly);

    typedef hpx::when_some_result<hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > > result_type;
    hpx::lcos::future<result_type> r = hpx::when_some(count, f1, f2, f3, f4, f5);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    result_type result = r.get();

    HPX_TEST_EQ(result.indices.size(), count);
    HPX_TEST(hpx::util::get<0>(result.futures).is_ready());
    HPX_TEST(hpx::util::get<1>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<2>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<3>(result.futures).is_ready());
    HPX_TEST(!hpx::util::get<4>(result.futures).is_ready());
}

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map&)
{
    {
        test_wait_for_two_out_of_five_futures();
        test_wait_for_three_out_of_five_futures();
        test_wait_for_two_out_of_five_late_futures();
        test_wait_for_two_out_of_five_deferred_futures();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

