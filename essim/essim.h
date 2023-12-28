/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(__SSIM_SSIM_H)
#define __SSIM_SSIM_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

#define PROFILING_PRINTS 0
#define DEBUG_PRINTS 0

typedef enum eSSIMResult {
  SSIM_OK = 0,
  SSIM_ERR_NULL_PTR = 1,
  SSIM_ERR_BAD_PARAM = 2,
  SSIM_ERR_BAD_SIZE = 3,
  SSIM_ERR_UNSUPPORTED = 4,
  SSIM_ERR_ALLOC = 5,
  SSIM_ERR_FAILED = 6
} eSSIMResult;

typedef enum eSSIMDataType {
  SSIM_DATA_8BIT = 1,
  SSIM_DATA_16BIT = 2
} eSSIMDataType;

typedef enum eSSIMMode {
  /* reference integer implementation. the function has few optimizations and
     may be used for experimentation */
  SSIM_MODE_REF = 0,

  /* performance integer implementation. the function is bit-exact as one in
     SSIM_MODE_REF, but it has a lot of optimizations. */
  SSIM_MODE_INT = 1,

  /* performance float implementation. the functions has more optimization.
     results on different platforms may vary. */
  SSIM_MODE_PERF = 2
} eSSIMMode;

typedef enum eSSIMFlags {
  /* return SSIM value using arithmetic mean */
  SSIM_SPATIAL_POOLING_AVERAGE = 0,

  /* return eSSIM value using minkowski pooling */
  SSIM_SPATIAL_POOLING_MINK = 1,

  /* return both SSIM and eSSIM values */
  SSIM_SPATIAL_POOLING_BOTH = 2

} eSSIMFlags;

enum {
  /* scaling of SSIM score */
  SSIM_LOG2_SCALE = 10
};

/*
    basic API
*/
eSSIMResult ssim_compute_8u(
    float* const pSsimScore,
    float* const pEssimScore,
    const uint8_t* ref,
    const ptrdiff_t refStride,
    const uint8_t* cmp,
    const ptrdiff_t cmpStride,
    const uint32_t width,
    const uint32_t height,
    const uint32_t windowSize,
    const uint32_t windowStride,
    const uint32_t d2h,
    const eSSIMMode mode,
    const eSSIMFlags flags,
    const uint32_t essim_mink_value);

eSSIMResult ssim_compute_16u(float *const pSsimScore, float *const pEssimScore,
                             const uint16_t *ref, const ptrdiff_t refStride,
                             const uint16_t *cmp, const ptrdiff_t cmpStride,
                             const uint32_t width, const uint32_t height,
                             const uint32_t bitDepthMinus8,
                             const uint32_t windowSize,
                             const uint32_t windowStride, const uint32_t d2h,
                             const eSSIMMode mode, const eSSIMFlags flags,
                             const uint32_t essim_mink_value);
/*
    advanced API
*/

typedef struct SSIM_CTX SSIM_CTX;
typedef struct SSIM_CTX_ARRAY SSIM_CTX_ARRAY;

/* allocate SSIM calculation contexts. these contexts have long life cycle
   and maybe reused in SSIM calls. typically contexts are allocated one per
   thread. */

SSIM_CTX_ARRAY *
ssim_allocate_ctx_array(const size_t numCtx, const uint32_t width,
                        const uint32_t height, const uint32_t bitDepthMinus8,
                        const eSSIMDataType dataType, const uint32_t windowSize,
                        const uint32_t windowStride, const uint32_t d2h,
                        const eSSIMMode mode, const eSSIMFlags flags,
                        const uint32_t essim_mink_value);

/* access a single context from allocated array of contexts */
SSIM_CTX* ssim_access_ctx(
    const SSIM_CTX_ARRAY* const ctxa,
    const size_t ctxIdx);

/* reset a ctx with partial SSIM score */
void ssim_reset_ctx(SSIM_CTX* const ctx);

/* compute partial SSIM score of an image region */
eSSIMResult ssim_compute_ctx(
    SSIM_CTX* const ctx,
    const void* ref,
    const ptrdiff_t refStride,
    const void* cmp,
    const ptrdiff_t cmpStride,
    const uint32_t roiY,
    const uint32_t roiHeight,
    const uint32_t essim_mink_value);

/* aggregate partial SSIM scores from contexts and provide the final SSIM score
 */
eSSIMResult ssim_aggregate_score(float *const pSsimScore,
                                 float *const pEssimScore,
                                 const SSIM_CTX_ARRAY *ctxa,
                                 const uint32_t essim_mink_value);

/* free SSIM contexts */
void ssim_free_ctx_array(SSIM_CTX_ARRAY* const ctxa);

#if defined(__cplusplus)
} // extern "C"

#include <memory>

// C++ callers

namespace ssim {

struct context_array_deleter_t {
  void operator()(SSIM_CTX_ARRAY* const ctxa) {
    ssim_free_ctx_array(ctxa);
  }
};

using context_array_t =
    std::unique_ptr<SSIM_CTX_ARRAY, context_array_deleter_t>;

inline context_array_t AllocateCtxArray(
    const size_t numCtx,
    const uint32_t width,
    const uint32_t height,
    const uint32_t bitDepthMinus8,
    const eSSIMDataType dataType,
    const uint32_t windowSize,
    const uint32_t windowStride,
    const uint32_t d2h,
    const eSSIMMode mode,
    const eSSIMFlags flags,
    const uint32_t essim_mink_value) {
  return context_array_t(ssim_allocate_ctx_array(
      numCtx,
      width,
      height,
      bitDepthMinus8,
      dataType,
      windowSize,
      windowStride,
      d2h,
      mode,
      flags,
      essim_mink_value));
}
} // namespace ssim

#endif /* defined(__cplusplus) */

#endif /* __SSIM_SSIM_H */
