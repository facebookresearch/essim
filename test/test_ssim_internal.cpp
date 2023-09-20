/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_ssim.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

namespace {

enum { LOG2_ALIGN = 6, ALIGN = 1u << LOG2_ALIGN, MAX_INC = 64 };

template <typename ptr_t>
inline void Allocate(ptr_t &ptr, size_t &allocatedSize,
                     const size_t requiredSize) {
  if (allocatedSize < requiredSize) {
    ptr.reset(ssim_alloc_aligned(requiredSize, LOG2_ALIGN));
    ASSERT_NE(ptr.get(), nullptr) << "CAN'T ALLOCATE MEMORY";
    allocatedSize = requiredSize;
  }
}

template <typename data_t, typename ptr_t>
inline std::tuple<data_t *, ptrdiff_t>
AllocateAndFill(ptr_t &ptr, size_t &allocatedSize, const uint32_t width,
                const data_t maxValue) {
  const ptrdiff_t stride =
      sizeof(data_t) * width + ((std::rand() % MAX_INC) & -sizeof(data_t));
  const size_t requiredSize = 4 * stride;
  Allocate(ptr, allocatedSize, requiredSize + MAX_INC);
  data_t *p = (data_t *)ptr.get() + (std::rand() % (MAX_INC / sizeof(data_t)));
  for (size_t s = 0; s < requiredSize / sizeof(data_t); ++s) {
    p[s] = std::rand() & maxValue;
  }

  return {p, stride};
}

template <typename sum_t, typename sigma_t, typename src_t>
void load_4x4_windows(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  const uint8_t *ref = (const uint8_t *)pSrc->ref;
  const ptrdiff_t refStride = pSrc->refStride;
  const uint8_t *cmp = (const uint8_t *)pSrc->cmp;
  const ptrdiff_t cmpStride = pSrc->cmpStride;

  sum_t *sum = (sum_t *)pBuf->p;
  sigma_t *s0 = (sigma_t *)((uint8_t *)pBuf->p + 1 * pBuf->stride);
  sigma_t *s1 = (sigma_t *)((uint8_t *)pBuf->p + 2 * pBuf->stride);
  sigma_t *s2 = (sigma_t *)((uint8_t *)pBuf->p + 3 * pBuf->stride);

  for (size_t i = 0; i < num4x4Windows; ++i) {
    sum_t ref_sum = 0;
    sum_t cmp_sum = 0;
    sigma_t ref_sigma_sqd = 0;
    sigma_t cmp_sigma_sqd = 0;
    sigma_t sigma_both = 0;

    for (uint32_t y = 0; y < 4; ++y) {
      const src_t *pR = (const src_t *)(ref + y * refStride);
      const src_t *pC = (const src_t *)(cmp + y * cmpStride);

      for (size_t x = 0; x < 4; ++x) {
        ref_sum += pR[x];
        cmp_sum += pC[x];
        ref_sigma_sqd += pR[x] * pR[x];
        cmp_sigma_sqd += pC[x] * pC[x];
        sigma_both += pR[x] * pC[x];
      }
    }

    ref += 4 * sizeof(src_t);
    cmp += 4 * sizeof(src_t);

    *sum++ = ref_sum;
    *sum++ = cmp_sum;
    *s0++ = ref_sigma_sqd;
    *s1++ = cmp_sigma_sqd;
    *s2++ = sigma_both;
  }

} // void load_4x4_windows(LOAD_4x4_WINDOWS_FORMAL_ARGS)

template <typename sum_t, typename sigma_t, typename data_t>
void TestLoad4x4Windows(load_4x4_windows_proc_t testProc,
                        const uint32_t bitDepthMinus8) {
  enum { ITER = 1000, MAX_4X4_NUM = 64 };
  const data_t maxValue = (data_t)((1u << (bitDepthMinus8 + 8)) - 1);

  ASSERT_EQ(sizeof(sum_t) * 2, sizeof(sigma_t)) << "WRONG WORKING TYPES";

  std::unique_ptr<void, ssim::aligned_memory_deleter_t> pSrcAllocated,
      pCmpAllocated;
  size_t srcAllocatedSize = 0;
  size_t cmpAllocatedSize = 0;
  std::unique_ptr<void, ssim::aligned_memory_deleter_t> pRef, pTst;
  size_t refAllocatedSize = 0;
  size_t tstAllocatedSize = 0;

  for (uint32_t i = 0; i < ITER; ++i) {
    const size_t num4x4Windows = std::rand() % (MAX_4X4_NUM + 1);
    const uint32_t width = num4x4Windows * 4;

    // allocate the source data
    auto [pSrc, srcStride] = AllocateAndFill<data_t>(
        pSrcAllocated, srcAllocatedSize, width, maxValue);

    auto [pCmp, cmpStride] = AllocateAndFill<data_t>(
        pCmpAllocated, cmpAllocatedSize, width, maxValue);

    // allocate the destination buffer
    const ptrdiff_t dstStride =
        ((sizeof(sigma_t) * num4x4Windows + (ALIGN - 1)) & -ALIGN);
    const size_t dstSize = dstStride * 4;
    Allocate(pRef, refAllocatedSize, dstSize);
    memset(pRef.get(), 0, dstSize);
    Allocate(pTst, tstAllocatedSize, dstSize);
    memset(pTst.get(), 0, dstSize);

    SSIM_SRC src = {pSrc, srcStride, pCmp, cmpStride};
    SSIM_4X4_WINDOW_BUFFER refBuf = {pRef.get(), dstStride};
    SSIM_4X4_WINDOW_BUFFER tstBuf = {pTst.get(), dstStride};

    // call functions
    load_4x4_windows<sum_t, sigma_t, data_t>(&refBuf, num4x4Windows, &src);
    testProc(&tstBuf, num4x4Windows, &src);

    // compare results
    auto res = memcmp(pRef.get(), pTst.get(), dstSize);
    ASSERT_EQ(res, 0);
  }

} // void TestLoad4x4Windows(load_4x4_windows_proc_t testProc)

} // namespace

TEST(ssimTest, Load4x4Windows8u) {
  TestLoad4x4Windows<uint16_t, uint32_t, uint8_t>(load_4x4_windows_8u_c, 0);
#if defined(_X86) || defined(_X64)
  if (ssim::CheckSIMD(cpu_ssse3)) {
    TestLoad4x4Windows<uint16_t, uint32_t, uint8_t>(load_4x4_windows_8u_ssse3,
                                                    0);
  }
  if (ssim::CheckSIMD(cpu_avx2)) {
    TestLoad4x4Windows<uint16_t, uint32_t, uint8_t>(load_4x4_windows_8u_avx2,
                                                    0);
  }
#elif defined(_ARM) || defined(_ARM64)
  if (ssim::CheckSIMD(cpu_neon)) {
    TestLoad4x4Windows<uint16_t, uint32_t, uint8_t>(load_4x4_windows_8u_neon,
                                                    0);
  }
#endif // defined(_X86) || defined(_X64)
  TestLoad4x4Windows<uint16_t, uint32_t, uint8_t>(load_4x4_windows_8u, 0);

} // TEST(ssimTest, Read4x4Windows8u)

TEST(ssimTest, Load4x4Windows16u) {
  for (uint32_t bitDepthMinus8 = 0; bitDepthMinus8 + 8 <= 14;
       bitDepthMinus8 += 2) {
    TestLoad4x4Windows<uint32_t, uint64_t, uint16_t>(load_4x4_windows_16u_c,
                                                     bitDepthMinus8);
#if defined(_X86) || defined(_X64)
    if (ssim::CheckSIMD(cpu_ssse3)) {
      TestLoad4x4Windows<uint32_t, uint64_t, uint16_t>(
          load_4x4_windows_16u_ssse3, bitDepthMinus8);
    }
    if (ssim::CheckSIMD(cpu_avx2)) {
      TestLoad4x4Windows<uint32_t, uint64_t, uint16_t>(
          load_4x4_windows_16u_avx2, bitDepthMinus8);
    }
#elif defined(_ARM) || defined(_ARM64)
    if (ssim::CheckSIMD(cpu_neon)) {
      TestLoad4x4Windows<uint32_t, uint64_t, uint16_t>(
          load_4x4_windows_16u_neon, bitDepthMinus8);
    }
#endif // defined(_X86) || defined(_X64)
    TestLoad4x4Windows<uint32_t, uint64_t, uint16_t>(load_4x4_windows_16u,
                                                     bitDepthMinus8);
  }

} // TEST(ssimTest, Read4x4Windows16u)
