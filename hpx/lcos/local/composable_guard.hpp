//  (C) Copyright 2013-2015 Steven R. Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef COMPOSABLE_GUARD_HPP
#define COMPOSABLE_GUARD_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/actions.hpp>
#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <iostream>

const int DEBUG_MAGIC = 0x2cab;
struct DebugObject {
	int magic;
	DebugObject() : magic(DEBUG_MAGIC) {}
	~DebugObject() {
		check();
		magic = ~DEBUG_MAGIC;
	}
	void check() {
		if(magic == ~DEBUG_MAGIC) {
			abort();
		}
		if(magic != DEBUG_MAGIC) {
			abort();
		}
	}
};

namespace hpx { namespace lcos { namespace local {
struct guard_task;

typedef std::atomic<guard_task *> guard_atomic;
typedef boost::atomic<signed char> char_atomic;

struct guard : DebugObject {
    guard_atomic task;

    guard() : task((guard_task *)0) {}
    ~guard() {
        free(task.load());
    }
};

class guard_set : DebugObject {
    std::vector<boost::shared_ptr<guard> > guards;
    // the guards need to be sorted, but we don't
    // want to sort them more often than necessary
    bool sorted;
    void sort();
public:
    guard_set() : guards(), sorted(true) {}
    ~guard_set() {}

    void add(boost::shared_ptr<guard> guard_ptr) {
        guards.push_back(guard_ptr);
        sorted = false;
    }

    friend HPX_API_EXPORT void run_guarded(guard_set& guards,boost::function<void()> task);
    boost::shared_ptr<guard> get(std::size_t i) { return guards[i]; }
};

/// Conceptually, a guard acts like a mutex on an asyncrhonous task. The
/// mutex is locked before the task runs, and unlocked afterwards.
HPX_API_EXPORT void run_guarded(guard& guard,boost::function<void()> task);

/// Conceptually, a guard_set acts like a set of mutexes on an asyncrhonous task. The
/// mutexes are locked before the task runs, and unlocked afterwards.
HPX_API_EXPORT void run_guarded(guard_set& guards,boost::function<void()> task);
}}};
#endif
