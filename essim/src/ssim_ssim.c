/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/essim.h>

#include <essim/inc/internal.h>
#include <essim/inc/memory.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define max(a, b) (((a) > (b)) ? (a) : (b))

uint32_t div_lookup[65536];

uint32_t* div_lookup_generator(void) {
    div_lookup[0] = div_Q_factor;
    for (int i = 1; i <= 65535; ++i) {
        div_lookup[i] = div_Q_factor / i;
    }
    return div_lookup;
}/*void div_lookup_generator()*/

eSSIMResult ssim_compute_8u(float *const pSsimScore, float *const pEssimScore,
                            const uint8_t *ref, const ptrdiff_t refStride,
                            const uint8_t *cmp, const ptrdiff_t cmpStride,
                            const uint32_t width, const uint32_t height,
                            const uint32_t windowSize,
                            const uint32_t windowStride, const uint32_t d2h,
                            const eSSIMMode mode, const eSSIMFlags flags,
                            const uint32_t essim_mink_value) {
    /* check error(s) */
    if ((NULL == pSsimScore) || (NULL == pEssimScore) || (NULL == ref) ||
        (NULL == cmp)) {
        return SSIM_ERR_NULL_PTR;
    }
    if ((0 == width) || (0 == height)) {
        return SSIM_ERR_BAD_SIZE;
    }
    if ((width < windowSize) || (height < windowSize) || (0 == windowSize)) {
        return SSIM_ERR_BAD_PARAM;
    }
    if ((8 > windowSize) || (16 < windowSize)) {
        return SSIM_ERR_BAD_PARAM;
    }
    if ((SSIM_MODE_REF != mode) && ((windowSize & 3) || (windowStride & 3))) {
        /* optimized versions support product of 4 dimensions only */
        return SSIM_ERR_BAD_PARAM;
    }

    SSIM_CTX_ARRAY *ctx_array = ssim_allocate_ctx_array(
        1, width, height, 0, SSIM_DATA_8BIT, windowSize, windowStride, d2h,
        mode, flags, essim_mink_value);

    if (NULL == ctx_array) {
        return SSIM_ERR_ALLOC;
    }

    SSIM_CTX *ctx = ssim_access_ctx(ctx_array, 0);
    eSSIMResult res = SSIM_OK;
    if (ctx) {
        ssim_reset_ctx(ctx);

        res = ssim_compute_ctx(ctx, ref, refStride, cmp, cmpStride, 0, height,
                               essim_mink_value);

        if (SSIM_OK == res) {
          res = ssim_aggregate_score(pSsimScore, pEssimScore, ctx_array,
                                     essim_mink_value);
        }

    } else {
        res = SSIM_ERR_FAILED;
    }

    ssim_free_ctx_array(ctx_array);

    return res;

} /* int ssim_compute_8u(uint32_t * const pScore, */

eSSIMResult ssim_compute_16u(float *const pSsimScore, float *const pEssimScore,
                             const uint16_t *ref, const ptrdiff_t refStride,
                             const uint16_t *cmp, const ptrdiff_t cmpStride,
                             const uint32_t width, const uint32_t height,
                             const uint32_t bitDepthMinus8,
                             const uint32_t windowSize,
                             const uint32_t windowStride, const uint32_t d2h,
                             const eSSIMMode mode, const eSSIMFlags flags,
                             const uint32_t essim_mink_value) {
    /* check error(s) */
    if ((NULL == pSsimScore) || (NULL == pEssimScore) || (NULL == ref) ||
        (NULL == cmp)) {
        return SSIM_ERR_NULL_PTR;
    }
    if ((0 == width) || (0 == height)) {
        return SSIM_ERR_BAD_SIZE;
    }
    if ((width < windowSize) || (height < windowSize) || (0 == windowSize)) {
        return SSIM_ERR_BAD_PARAM;
    }
    if ((8 > windowSize) || (16 < windowSize)) {
        return SSIM_ERR_BAD_PARAM;
    }
    if ((SSIM_MODE_REF != mode) && ((windowSize & 3) || (windowStride & 3))) {
        /* optimized versions support product of 4 dimensions only */
        return SSIM_ERR_BAD_PARAM;
    }

    SSIM_CTX_ARRAY *ctx_array = ssim_allocate_ctx_array(
        1, width, height, bitDepthMinus8, SSIM_DATA_16BIT, windowSize,
        windowStride, d2h, mode, flags, essim_mink_value);
    if (NULL == ctx_array) {
        return SSIM_ERR_ALLOC;
    }

    SSIM_CTX *ctx = ssim_access_ctx(ctx_array, 0);
    eSSIMResult res = SSIM_OK;
    if (ctx) {
        ssim_reset_ctx(ctx);

        res = ssim_compute_ctx(ctx, ref, refStride, cmp, cmpStride, 0, height,
                               essim_mink_value);

        if (SSIM_OK == res) {
          res = ssim_aggregate_score(pSsimScore, pEssimScore, ctx_array,
                                     essim_mink_value);
        }

    } else {
        res = SSIM_ERR_FAILED;
    }

    ssim_free_ctx_array(ctx_array);

    return res;

} /* eSSIMResult ssim_compute_16u(uint32_t * const pScore, */

static void ssim_free_ctx(SSIM_CTX *const ctx) {
  ssim_free_aligned(ctx->buffer);
  free(ctx->windowRows);
  free(ctx);
}

static SSIM_CTX *ssim_allocate_ctx(const uint32_t width,
                                   const eSSIMDataType dataType,
                                   const uint32_t windowSize,
                                   const eSSIMMode mode) {
  // extra | ALIGN is required to make sure that the first bit after alignment
  // is 1
  const ptrdiff_t bufferStride =
      ((width * dataType + (ALIGN - 1)) & (-ALIGN)) | ALIGN;
  const uint32_t numWindowRows = windowSize / 4;
  const size_t bufferSize = windowSize * bufferStride;

  SSIM_CTX *ctx = NULL;

  const size_t ctxSize = sizeof(SSIM_CTX);
  ctx = malloc(ctxSize);
  if (NULL == ctx) {
    return NULL;
  }
  memset(ctx, 0, ctxSize);

  if (SSIM_MODE_REF != mode) {
    /* allocate buffer */
    ctx->buffer = ssim_alloc_aligned(bufferSize, LOG2_ALIGN);
    if (NULL == ctx->buffer) {
      ssim_free_ctx(ctx);
      return NULL;
    }
    ctx->bufferSize = bufferSize;
    ctx->bufferStride = bufferStride;

    /* reset row buffers */
    const size_t rowBuffersArraySize =
        sizeof(SSIM_4X4_WINDOW_ROW) * numWindowRows;
    uint8_t *pBuf = ctx->buffer;
    ctx->windowRows = malloc(rowBuffersArraySize);
    if (NULL == ctx->windowRows) {
      ssim_free_ctx(ctx);
      return NULL;
    }
    memset(ctx->windowRows, 0, rowBuffersArraySize);
    ctx->numWindowRows = numWindowRows;

    for (size_t r = 0; r < numWindowRows; ++r) {
      ctx->windowRows[r].ptrs.p = pBuf;
      ctx->windowRows[r].ptrs.stride = bufferStride;
      pBuf += 4 * bufferStride;
    }
  }

  return ctx;

} /* SSIM_CTX* ssim_allocate_ctx(const uint32_t width, */

SSIM_CTX_ARRAY *
ssim_allocate_ctx_array(const size_t numCtx, const uint32_t width,
                        const uint32_t height, const uint32_t bitDepthMinus8,
                        const eSSIMDataType dataType, const uint32_t windowSize,
                        const uint32_t windowStride, const uint32_t d2h,
                        const eSSIMMode eSSIMmode, const eSSIMFlags flags,
                        const uint32_t essim_mink_value) {
  /* check error(s) */
  if ((0 == width) || (0 == height)) {
    return NULL;
  }
  if (14 < bitDepthMinus8 + 8) {
    return NULL;
  }
  if ((SSIM_DATA_8BIT != dataType) && (SSIM_DATA_16BIT != dataType)) {
    return NULL;
  }
  if ((SSIM_DATA_8BIT == dataType) && (0 != bitDepthMinus8)) {
    return NULL;
  }
  if ((width < windowSize) || (height < windowSize) || (0 == windowSize)) {
    return NULL;
  }
  if ((8 > windowSize) || (16 < windowSize)) {
    return NULL;
  }

  if ((width >= MAX_FRAME_WIDTH) && (height >= MAX_FRAME_HEIGHT)) {
    if(windowStride==4)
      printf("\nWARNING : eSSIM precision will be very low\n");
  }
  eSSIMMode mode = SSIM_MODE_INT;
  if(bitDepthMinus8 > 2) {
    mode = SSIM_MODE_PERF;
    printf("\nWARNING : Bit-depth > 10 only eSSIM Float is supported\n");
  } else {
    mode = eSSIMmode;
  }

  if ((SSIM_MODE_REF != mode) && ((windowSize & 3) || (windowStride & 3))) {
    /* optimized versions support dimensions that product of 4 only */
    return NULL;
  }

  /* allocate & initialize the ctx array */
  const size_t ctxArraySize = sizeof(SSIM_CTX_ARRAY);
  SSIM_CTX_ARRAY *p = malloc(ctxArraySize);
  if (NULL == p) {
    return NULL;
  }
  memset(p, 0, ctxArraySize);
  p->params.width = width;
  p->params.height = height;
  p->params.bitDepthMinus8 = bitDepthMinus8;
  p->params.dataType = dataType;
  p->params.windowSize = windowSize;
  p->params.windowStride = windowStride;
  p->params.mode = mode;
  p->params.flags = flags;
  if (SSIM_MODE_REF != mode) {
    if (SSIM_DATA_8BIT == dataType) {
      p->params.load_4x4_windows_proc = load_4x4_windows_8u;
      if ((8 == windowSize) && (4 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_8x4_int_8u)
                                         : (sum_windows_8x4_float_8u);
#if NEW_SIMD_FUNC
      } else if ((8 == windowSize) && (8 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_8x8_int_8u)
                                         : (sum_windows_8x8_float_8u);
      } else if ((16 == windowSize) && (4 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_16x4_int_8u)
                                         : (sum_windows_16x4_float_8u);
      } else if ((16 == windowSize) && (8 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_16x8_int_8u)
                                         : (sum_windows_16x8_float_8u);
      } else if ((16 == windowSize) && (16 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_16x16_int_8u)
                                         : (sum_windows_16x16_float_8u);
#endif
      } else if ((12 == windowSize) && (4 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_12x4_int_8u)
                                         : (sum_windows_12x4_float_8u);
      } else {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_int_8u)
                                         : (sum_windows_float_8u);
      }
#if NEW_SIMD_FUNC
    } else if (bitDepthMinus8 == 2) {
      p->params.load_4x4_windows_proc = load_4x4_windows_10u;
      if ((8 == windowSize) && (4 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_8x4_int_10u)
                                         : (sum_windows_8x4_float_10u);
      } else if ((8 == windowSize) && (8 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_8x8_int_10u)
                                         : (sum_windows_8x8_float_10u);
      } else if ((16 == windowSize) && (4 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_16x4_int_10u)
                                         : (sum_windows_16x4_float_10u);
      } else if ((16 == windowSize) && (8 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_16x8_int_10u)
                                         : (sum_windows_16x8_float_10u);
      } else if ((16 == windowSize) && (16 == windowStride)) {
        p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                         ? (sum_windows_16x16_int_10u)
                                         : (sum_windows_16x16_float_10u);
      }
#endif
    } else {
    p->params.load_4x4_windows_proc = load_4x4_windows_16u;
    p->params.sum_windows_proc = (SSIM_MODE_INT == mode)
                                       ? (sum_windows_int_16u)
                                       : (sum_windows_float_16u);
    }
  } else {
    p->params.load_window_proc =
        (SSIM_DATA_8BIT == dataType) ? (load_window_8u_c) : (load_window_16u_c);

    if(bitDepthMinus8 == 0) {
      p->params.calc_window_ssim_proc = calc_window_ssim_int_8u;
    } else if (bitDepthMinus8 ==2) {
      p->params.calc_window_ssim_proc = calc_window_ssim_int_10bd;
    } else {
      p->params.calc_window_ssim_proc = calc_window_ssim_int_16u;
    }
  }
  p->d2h = d2h;

  const size_t arraySize = sizeof(SSIM_CTX *) * numCtx;
  p->ctx = malloc(arraySize);
  if (NULL == p->ctx) {
    ssim_free_ctx_array(p);
    return NULL;
  }
  memset(p->ctx, 0, sizeof(arraySize));
  p->numCtx = numCtx;

  uint32_t SSIMValRtShiftBits = 0;
  uint32_t SSIMValRtShiftHalfRound = 0;
  uint32_t* div_lookup_ptr = NULL;

  const uint64_t MAX_SSIM_ACCUMULATED_SUM_VALUE
                 = essim_mink_value == 4 ? 18446744073709551615U : (uint64_t)1 << 63;
  if(mode != SSIM_MODE_PERF) {
    /*generating LUT to avoid final stage division in cal window for ssim_val*/
    div_lookup_ptr = div_lookup_generator();
    uint32_t numWindows = GetTotalWindows(width, height, windowSize, windowStride);
    uint32_t ssimFinalPrecisionMaxVal =
                ceil(pow(MAX_SSIM_ACCUMULATED_SUM_VALUE/(double)numWindows, 1.0/essim_mink_value)/2);
    uint32_t ssimFinalPrecisionInBits = GetTotalBitsInNumber(ssimFinalPrecisionMaxVal);
    int32_t additionlRtShiftBits = DEFAULT_Q_FORMAT_FOR_SSIM_VAL - (int32_t)ssimFinalPrecisionInBits;
    SSIMValRtShiftBits = DEFAULT_Q_FORMAT_FOR_SSIM_VAL + additionlRtShiftBits;
    SSIMValRtShiftHalfRound = 1 << (SSIMValRtShiftBits-1);
  }

  /* initialize individual contexts */
  for (size_t i = 0; i < numCtx; ++i) {
    p->ctx[i] = ssim_allocate_ctx(width, dataType, windowSize, mode);
    if (NULL == p->ctx[i]) {
      ssim_free_ctx_array(p);
      return NULL;
    }

    p->ctx[i]->params = &p->params;
    p->ctx[i]->div_lookup_ptr = div_lookup_ptr;
    p->ctx[i]->SSIMValRtShiftBits = SSIMValRtShiftBits ;
    p->ctx[i]->SSIMValRtShiftHalfRound = SSIMValRtShiftHalfRound;
  }

  return p;

} /* SSIM_CTX_ARRAY* ssim_allocate_ctx_array(const size_t numCtx, */

SSIM_CTX *ssim_access_ctx(const SSIM_CTX_ARRAY *const ctxa,
                          const size_t ctxIdx) {
  if (NULL == ctxa) {
    return NULL;
  }
  if (ctxIdx >= ctxa->numCtx) {
    return NULL;
  }

  return ctxa->ctx[ctxIdx];

} /* SSIM_CTX* ssim_access_ctx(const SSIM_CTX_ARRAY * const ctxa, */

void ssim_reset_ctx(SSIM_CTX *const ctx) {
  if (NULL == ctx) {
    return;
  }

  memset(&ctx->res, 0, sizeof(SSIM_RES));

  for (size_t r = 0; r < ctx->numWindowRows; ++r) {
    ctx->windowRows[r].y = (uint32_t)-1;
  }

} /* void ssim_reset_ctx(SSIM_CTX * const ctx) */

eSSIMResult ssim_compute_ctx(SSIM_CTX *const ctx, const void *ref,
                             const ptrdiff_t refStride, const void *cmp,
                             const ptrdiff_t cmpStride, const uint32_t roiY,
                             const uint32_t roiHeight,
                             const uint32_t essim_mink_value) {
  /* check error(s) */
  if ((NULL == ctx) || (NULL == ref) || (NULL == cmp)) {
    return SSIM_ERR_NULL_PTR;
  }

  const uint32_t height = ctx->params->height;
  const uint32_t windowSize = ctx->params->windowSize;
  const uint32_t windowStride = ctx->params->windowStride;
  const eSSIMMode mode = ctx->params->mode;

  if ((SSIM_MODE_REF == mode) && ((0 != roiY) || (height != roiHeight))) {
    /* precission mode supports only single call per frame */
    return SSIM_ERR_BAD_PARAM;
  }
  if ((height <= roiY) || (height < roiY + roiHeight)) {
    return SSIM_ERR_BAD_PARAM;
  }
  if ((8 > windowSize) || (16 < windowSize)) {
    return SSIM_ERR_BAD_PARAM;
  }

  if ((SSIM_MODE_REF != mode) && ((windowSize & 3) || (windowStride & 3))) {
    /* optimized versions support dimensions that product of 4 only */
    return SSIM_ERR_BAD_PARAM;
  }

  if (SSIM_MODE_REF == mode) {
    return ssim_compute_prec(ctx, ref, refStride, cmp, cmpStride,
                             essim_mink_value);
  } else {
    return ssim_compute_perf(ctx, ref, refStride, cmp, cmpStride, roiY,
                             roiHeight, essim_mink_value);
  }

} /* eSSIMResult ssim_compute_ctx(SSIM_CTX * const ctx, */

eSSIMResult ssim_aggregate_score(float *const pSsimScore,
                                 float *const pEssimScore,
                                 const SSIM_CTX_ARRAY *ctxa,
                                 const uint32_t essim_mink_value) {

  if ((NULL == pSsimScore) || (NULL == pEssimScore) || (NULL == ctxa)) {
    return SSIM_ERR_NULL_PTR;
  }

  if (SSIM_MODE_PERF == ctxa->params.mode) {
    double ssim_sum = 0.0f;
    double ssim_mink_sum = 0.0f;
    size_t numWindows = 0;

    for (size_t i = 0; i < ctxa->numCtx; ++i) {
      SSIM_CTX *const ctx = ssim_access_ctx(ctxa, i);

      ssim_sum += ctx->res.ssim_sum_f;
      ssim_mink_sum += ctx->res.ssim_mink_sum_f;
      numWindows += ctx->res.numWindows;
    }

    if (SSIM_SPATIAL_POOLING_AVERAGE == ctxa->params.flags ||
        SSIM_SPATIAL_POOLING_BOTH == ctxa->params.flags) {
      if (numWindows) {
        *pSsimScore = ssim_sum / (float)numWindows;
      } else {
        return SSIM_ERR_FAILED;
      }
    }
    if (SSIM_SPATIAL_POOLING_MINK == ctxa->params.flags ||
        SSIM_SPATIAL_POOLING_BOTH == ctxa->params.flags) {

      if (numWindows) {
        if(ssim_mink_sum > 0.000000001 || ssim_mink_sum < -0.000000001) {
            *pEssimScore = 1.0 - pow(ssim_mink_sum / (float)numWindows,
                                 1.0 / essim_mink_value);
        }
        else {
          *pEssimScore = 1.0;
        }
      } else {
        return SSIM_ERR_FAILED;
      }
    }
  } else {
    int64_t ssim_sum = 0;
    int64_t ssim_mink_sum = 0;
    size_t numWindows = 0;
    uint32_t SSIMValRtShiftBits = 0;

    for (size_t i = 0; i < ctxa->numCtx; ++i) {
      SSIM_CTX *const ctx = ssim_access_ctx(ctxa, i);

      ssim_sum += ctx->res.ssim_sum;
      ssim_mink_sum += ctx->res.ssim_mink_sum;
      numWindows += ctx->res.numWindows;
      SSIMValRtShiftBits = ctx->SSIMValRtShiftBits;
    }
    int32_t extraRtShiftBitsForSSIMVal =
            (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
    uint32_t const_1 =
        1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);

    if (SSIM_SPATIAL_POOLING_AVERAGE == ctxa->params.flags ||
        SSIM_SPATIAL_POOLING_BOTH == ctxa->params.flags) {
      if (numWindows) {
        double avg_ssim_sum =  (double)ssim_sum/numWindows;
        *pSsimScore = avg_ssim_sum / const_1;
      } else {
        return SSIM_ERR_FAILED;
      }
    }
    if (SSIM_SPATIAL_POOLING_MINK == ctxa->params.flags ||
        SSIM_SPATIAL_POOLING_BOTH == ctxa->params.flags) {

      if (numWindows) {
        double avg_ssim_mink_sum =  (double)ssim_mink_sum/numWindows;
        *pEssimScore =
            1.0 -
            (double)(pow(avg_ssim_mink_sum, 1.0 / essim_mink_value)) / const_1;
      } else {
        return SSIM_ERR_FAILED;
      }
    }
  }

  return SSIM_OK;

} /* eSSIMResult ssim_aggregate_score(float * const pSsimScore, */

void ssim_free_ctx_array(SSIM_CTX_ARRAY *ctxa) {
  if (NULL == ctxa) {
    return;
  }

  if (ctxa->ctx) {
    for (size_t i = 0; i < ctxa->numCtx; ++i) {
      SSIM_CTX *ctx = ssim_access_ctx(ctxa, i);
      if (ctx) {
        ssim_free_ctx(ctx);
      }
    }
    free(ctxa->ctx);
    ctxa->ctx = NULL;
    ctxa->numCtx = 0;
  }
  free(ctxa);

} /* void ssim_free_ctx_array(SSIM_CTX_ARRAY *ctxa) */
