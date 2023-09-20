/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_ssim.h"

#include <gtest/gtest.h>

#include <memory>

TEST(ssimTest, memory) {
  enum {
    ITER = 1000,
    MAX_LOG2_ALIGN = 12,
    MAX_SIZE = 16 * 1024,
    MIN_LOG2_ALIGN = 1
  };

  for (uint32_t i = 0; i < ITER; ++i) {
    const size_t size = std::rand() % MAX_SIZE;
    const uint32_t log2Align =
        std::rand() % (MAX_LOG2_ALIGN - MIN_LOG2_ALIGN + 1) + MIN_LOG2_ALIGN;
    const size_t align = 1u << log2Align;

    std::unique_ptr<void, ssim::aligned_memory_deleter_t> p(
        ssim_alloc_aligned(size, log2Align));
    if (!p) {
      GTEST_FAIL() << "Allocation failed";
    }
    ASSERT_EQ(0, ((ptrdiff_t)p.get() & (align - 1)));
    memset(p.get(), 0, size);
  }

} // TEST(ssimTest, GetCpuType)
