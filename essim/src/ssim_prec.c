/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/essim.h>
#include <essim/inc/internal.h>

void load_window_8u_c(LOAD_WINDOW_FORMAL_ARGS) {
  const uint8_t *pR = pSrc->ref;
  const ptrdiff_t refStride = pSrc->refStride;
  const uint8_t *pC = pSrc->cmp;
  const ptrdiff_t cmpStride = pSrc->cmpStride;

  uint16_t ref_sum = 0;
  uint16_t cmp_sum = 0;
  uint32_t ref_sigma_sqd = 0;
  uint32_t cmp_sigma_sqd = 0;
  uint32_t sigma_both = 0;

  for (uint32_t y = 0; y < windowSize; ++y) {
    for (size_t x = 0; x < windowSize; ++x) {
      ref_sum += pR[x];
      cmp_sum += pC[x];
      ref_sigma_sqd += pR[x] * pR[x];
      cmp_sigma_sqd += pC[x] * pC[x];
      sigma_both += pR[x] * pC[x];
    }

    pR = AdvancePointer(pR, refStride);
    pC = AdvancePointer(pC, cmpStride);
  }

  pWnd->ref_sum = ref_sum;
  pWnd->cmp_sum = cmp_sum;
  pWnd->ref_sigma_sqd = ref_sigma_sqd;
  pWnd->cmp_sigma_sqd = cmp_sigma_sqd;
  pWnd->sigma_both = sigma_both;

} /* void load_window_8u_c(LOAD_WINDOW_FORMAL_ARGS) */

void load_window_16u_c(LOAD_WINDOW_FORMAL_ARGS) {
  const uint16_t *pR = pSrc->ref;
  const ptrdiff_t refStride = pSrc->refStride;
  const uint16_t *pC = pSrc->cmp;
  const ptrdiff_t cmpStride = pSrc->cmpStride;

  uint32_t ref_sum = 0;
  uint32_t cmp_sum = 0;
  uint64_t ref_sigma_sqd = 0;
  uint64_t cmp_sigma_sqd = 0;
  uint64_t sigma_both = 0;

  for (uint32_t y = 0; y < windowSize; ++y) {
    for (size_t x = 0; x < windowSize; ++x) {
      ref_sum += pR[x];
      cmp_sum += pC[x];
      ref_sigma_sqd += pR[x] * pR[x];
      cmp_sigma_sqd += pC[x] * pC[x];
      sigma_both += pR[x] * pC[x];
    }

    pR = AdvancePointer(pR, refStride);
    pC = AdvancePointer(pC, cmpStride);
  }

  pWnd->ref_sum = ref_sum;
  pWnd->cmp_sum = cmp_sum;
  pWnd->ref_sigma_sqd = ref_sigma_sqd;
  pWnd->cmp_sigma_sqd = cmp_sigma_sqd;
  pWnd->sigma_both = sigma_both;

} /* void load_window_16u_c(LOAD_WINDOW_FORMAL_ARGS) */

eSSIMResult ssim_compute_prec(SSIM_CTX *const ctx, const void *ref,
                              const ptrdiff_t refStride, const void *cmp,
                              const ptrdiff_t cmpStride,
                              const uint32_t essim_mink_value) {
  const load_window_proc_t load_window_proc = ctx->params->load_window_proc;
  const calc_window_ssim_proc_t calc_window_ssim_proc =
      ctx->params->calc_window_ssim_proc;

  const uint32_t width = ctx->params->width;
  const uint32_t height = ctx->params->height;
  const uint32_t windowSize = ctx->params->windowSize;
  const uint32_t windowStride = ctx->params->windowStride;
  const size_t windowStep =
      windowStride * ((SSIM_DATA_8BIT == ctx->params->dataType) ? (1) : (2));

  const uint32_t bitDepthMinus8 = ctx->params->bitDepthMinus8;
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);

  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)ctx->SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  int64_t const_1 =
      1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);

  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  SSIM_SRC src;

  src.refStride = refStride;
  src.cmpStride = cmpStride;

  for (uint32_t y = 0; y + windowSize <= height; y += windowStride) {
    src.ref = AdvancePointer(ref, y * refStride);
    src.cmp = AdvancePointer(cmp, y * cmpStride);

    for (size_t x = 0; x + windowSize <= width; x += windowStride) {
      WINDOW_STATS wnd;

      /* STEP 1. load data, no scaling */
      load_window_proc(&wnd, &src, windowSize);
      src.ref = AdvancePointer(src.ref, windowStep);
      src.cmp = AdvancePointer(src.cmp, windowStep);
      const int64_t ssim_val = calc_window_ssim_proc(&wnd, windowSize, C1, C2,
                    ctx->div_lookup_ptr, ctx->SSIMValRtShiftBits,
                    ctx->SSIMValRtShiftHalfRound);

      ssim_sum += ssim_val;
      int64_t const_1_minus_ssim_val = const_1 - ssim_val;
      if(essim_mink_value == 4) {
        mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val
                          * const_1_minus_ssim_val * const_1_minus_ssim_val;
      } else {
        /*essim_mink_value == 3*/
        mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val
                            * const_1_minus_ssim_val;
      }
      ssim_mink_sum += mink_pow_ssim_val;
    }
  }

  ctx->res.ssim_sum = ssim_sum;
  ctx->res.ssim_mink_sum = ssim_mink_sum;
  ctx->res.numWindows =
      GetTotalWindows(width, height, windowSize, windowStride);

  return SSIM_OK;

} /* eSSIMResult ssim_compute_prec(SSIM_CTX * const ctx, */
