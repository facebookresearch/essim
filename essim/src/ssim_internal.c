/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/inc/internal.h>
#include <math.h>

uint32_t GetNum4x4Windows(const uint32_t value, const uint32_t windowSize,
                          const uint32_t windowStride) {
  return (windowSize / 4) +
         ((value - windowSize) / windowStride) * (windowStride / 4);
}

uint32_t GetNumWindows(const uint32_t value, const uint32_t windowSize,
                       const uint32_t windowStride) {
  return 1 + (value - windowSize) / windowStride;
}

uint32_t GetTotalWindows(const uint32_t width, const uint32_t height,
                         const uint32_t windowSize,
                         const uint32_t windowStride) {
  const uint32_t widthWindows = GetNumWindows(width, windowSize, windowStride);
  const uint32_t heightWindows =
      GetNumWindows(height, windowSize, windowStride);

  return widthWindows * heightWindows;
}

void *AdvancePointer(const void *p, const ptrdiff_t stride) {
  return (void *)((const uint8_t *)p + stride);
}

uint32_t get_ssim_int_constant(const uint32_t constIdx,
                               const uint32_t bitDepthMinus8,
                               const uint32_t windowSize) {
  if ((1 > constIdx) || (3 < constIdx)) {
    return 0;
  }

  const uint32_t L = (1u << (bitDepthMinus8 + 8)) - 1;
  const uint32_t K1 = 1;
  const uint32_t K2 = 3;
  const uint32_t windowSize_sqd = windowSize * windowSize;
  const uint32_t windowSize_qd = windowSize_sqd * windowSize_sqd;

  uint64_t c;
  if (1 == constIdx) {
    c = (K1 * L) * (K1 * L);
  } else {
    c = (uint64_t)(K2 * L) * (K2 * L);
  }
  c = (windowSize_qd * c + 5000) / 10000;
  return (uint32_t)((3 == constIdx) ? (c / 2) : (c));

} /* uint32_t get_ssim_int_constant(const uint32_t constIdx, const uint32_t
     bitDepthMinus8, */

float get_ssim_float_constant(const uint32_t constIdx,
                              const uint32_t bitDepthMinus8) {
  if ((1 > constIdx) || (3 < constIdx)) {
    return 0;
  }

  const float L = (float)(1u << (bitDepthMinus8 + 8)) - 1;
  const float K1 = 0.01f;
  const float K2 = 0.03f;

  float c;
  if (1 == constIdx) {
    c = (K1 * L) * (K1 * L);
  } else {
    c = (K2 * L) * (K2 * L);
  }
  return (3 == constIdx) ? (c / 2.0f) : (c);

} /* float get_ssim_float_constant(const uint32_t constIdx, const uint32_t
     bitDepthMinus8) */

uint32_t GetTotalBitsInNumber(uint32_t number) {

  return (uint32_t)log2(number);

} /*uint32_t GetTotalBitsInNumber(uint32_t number)*/

uint16_t get_best_i16_from_u64(uint64_t temp, int *power) {
    assert(temp >= 0x20000);
    int k = __builtin_clzll(temp);
    k = 49 - k;
    temp = temp >> k;
    *power = k;
    return (uint16_t) temp;
} /*uint16_t get_best_i16_from_u64(uint64_t temp, int *power)*/

int64_t calc_window_ssim_int_8u(CALC_WINDOW_SSIM_FORMAL_ARGS) {
  const uint32_t windowSize_sqd = windowSize * windowSize;

  const uint16_t ref_sum = (uint16_t)pWnd->ref_sum;
  const uint16_t cmp_sum = (uint16_t)pWnd->cmp_sum;
  uint32_t ref_sigma_sqd = (uint32_t)pWnd->ref_sigma_sqd * windowSize_sqd;
  uint32_t cmp_sigma_sqd = (uint32_t)pWnd->cmp_sigma_sqd * windowSize_sqd;
  uint32_t sigma_both = (uint32_t)pWnd->sigma_both * windowSize_sqd;

  /* STEP 2. adjust values */

  const uint32_t ref_sum_sqd = ref_sum * ref_sum ;
  ref_sigma_sqd -= ref_sum_sqd;
  const uint32_t cmp_sum_sqd = cmp_sum * cmp_sum ;
  cmp_sigma_sqd -= cmp_sum_sqd;
  const uint32_t both_sum_mul = ref_sum * cmp_sum;
  /* both sigma_both and both_sum are divided on 2 to avoid
  overflowing while uinsigned -> int casting */
  const int32_t sigma_both_a =
      (int32_t)(sigma_both / 2) - (int32_t)(both_sum_mul / 2);

  /* STEP 3. process numbers, do scale */

  /* all following 2^8 * 2^8 * windowSize^4.
  to avoid overflowing some values are divided on 2 */
  const uint32_t a = (both_sum_mul >> 1) + (C1 >> 2);
  const int32_t b = (sigma_both_a + (C2 >> 2));
  const uint32_t c = (ref_sum_sqd >> 2) + (cmp_sum_sqd  >> 2) + (C1 >> 2);
  const uint32_t d = (ref_sigma_sqd >> 1) + (cmp_sigma_sqd >> 1) + (C2 >> 1);

  const int64_t num = (int64_t)a * b * 2;
  const int64_t denom = ((int64_t)c * d);

  int power_val;
  uint16_t i16_map_denom = get_best_i16_from_u64((uint64_t)denom, &power_val);

  int64_t num_map = (num >> power_val);

  const int64_t ssim_val = (( num_map * div_lookup_ptr[i16_map_denom])
                            + SSIMValRtShiftHalfRound) >> SSIMValRtShiftBits;

  return ssim_val;

} /* int64_t calc_window_ssim_int_8u(CALC_WINDOW_SSIM_FORMAL_ARGS) */

int64_t calc_window_ssim_int_10bd(CALC_WINDOW_SSIM_FORMAL_ARGS) {
  const uint32_t windowSize_sqd = windowSize * windowSize;

  const uint32_t ref_sum = pWnd->ref_sum;
  const uint32_t cmp_sum = pWnd->cmp_sum;
  uint64_t ref_sigma_sqd = pWnd->ref_sigma_sqd * windowSize_sqd;
  uint64_t cmp_sigma_sqd = pWnd->cmp_sigma_sqd * windowSize_sqd;
  int64_t sigma_both = pWnd->sigma_both * windowSize_sqd;

  /* STEP 2. adjust values */

  const uint64_t ref_sum_sqd = (uint64_t)ref_sum * ref_sum;
  ref_sigma_sqd -= ref_sum_sqd;
  const uint64_t cmp_sum_sqd = (uint64_t)cmp_sum * cmp_sum;
  cmp_sigma_sqd -= cmp_sum_sqd;
  const uint64_t both_sum_mul = (uint64_t)ref_sum * cmp_sum;
  sigma_both -= (int64_t)both_sum_mul;

  /* STEP 3. process numbers, do scale */

  /* all following 2^X * 2^X * windowSize^4 */
  const uint64_t a = 2 * both_sum_mul + C1;
  const int64_t b = sigma_both + C2 / 2;
  const uint64_t c = ref_sum_sqd + cmp_sum_sqd + C1;
  const uint64_t d = ref_sigma_sqd + cmp_sigma_sqd + C2;

  const int64_t num = (int64_t)(a >> 5)  * (b >> 5);
  const uint64_t denom = ((uint64_t)(c >> 5) * (d >> 5)) >> 1;

  int power_val;
  uint16_t i16_map_denom = get_best_i16_from_u64((uint64_t)denom, &power_val);
  int64_t num_map = (num >> power_val);

  const int64_t ssim_val = (( num_map * div_lookup_ptr[i16_map_denom])
                            + SSIMValRtShiftHalfRound) >> SSIMValRtShiftBits;
  return ssim_val;

} /* int64_t calc_window_ssim_int_10bd(CALC_WINDOW_SSIM_FORMAL_ARGS) */

int64_t calc_window_ssim_int_16u(CALC_WINDOW_SSIM_FORMAL_ARGS) {
  const uint32_t windowSize_sqd = windowSize * windowSize;

  const uint32_t ref_sum = pWnd->ref_sum;
  const uint32_t cmp_sum = pWnd->cmp_sum;
  uint64_t ref_sigma_sqd = pWnd->ref_sigma_sqd * windowSize_sqd;
  uint64_t cmp_sigma_sqd = pWnd->cmp_sigma_sqd * windowSize_sqd;
  int64_t sigma_both = pWnd->sigma_both * windowSize_sqd;

  /* STEP 2. adjust values */

  const uint64_t ref_sum_sqd = (uint64_t)ref_sum * ref_sum;
  ref_sigma_sqd -= ref_sum_sqd;
  const uint64_t cmp_sum_sqd = (uint64_t)cmp_sum * cmp_sum;
  cmp_sigma_sqd -= cmp_sum_sqd;
  const uint64_t both_sum_mul = (uint64_t)ref_sum * cmp_sum;
  sigma_both -= (int64_t)both_sum_mul;

  /* STEP 3. process numbers, do scale */

  /* all following 2^X * 2^X * windowSize^4 */
  const uint64_t a = 2 * both_sum_mul + C1;
  const int64_t b = sigma_both + C2 / 2;
  const uint64_t c = ref_sum_sqd + cmp_sum_sqd + C1;
  const uint64_t d = ref_sigma_sqd + cmp_sigma_sqd + C2;

  /* can't scale num on SSIM_SCALE to avoid overflowing.
  dividing denom is an only option available, but it make results
  a bit noisy */
  const int64_t num = (int64_t)a * b;
  const int64_t denom = ((int64_t)c * d) / (2 * (1 << SSIM_LOG2_SCALE));
  /* in most cases denom is a huge value, and | 1 doesn't affect
  anything. when denom is small and | 1 affects computation,
  ssim_value is still very noisy and not reliable */
  const int64_t ssim_val = (num + denom / 2) / (denom | 1);

  return ssim_val;

} /* int64_t calc_window_ssim_int_16u(CALC_WINDOW_SSIM_FORMAL_ARGS) */

float calc_window_ssim_float(WINDOW_STATS *const pWnd,
                             const uint32_t windowSize, const float C1,
                             const float C2) {
  const float invWindowSize_sqd = 1.0f / (float)(windowSize * windowSize);
  const float invWindowSize_qd = invWindowSize_sqd * invWindowSize_sqd;

  const float ref_sum = pWnd->ref_sum;
  const float cmp_sum = pWnd->cmp_sum;
  float ref_sigma_sqd = (float)pWnd->ref_sigma_sqd * invWindowSize_sqd;
  float cmp_sigma_sqd = (float)pWnd->cmp_sigma_sqd * invWindowSize_sqd;
  float sigma_both = (float)pWnd->sigma_both * invWindowSize_sqd;

  /* STEP 2. adjust values */

  const float ref_sum_sqd = (ref_sum * ref_sum) * invWindowSize_qd;
  ref_sigma_sqd -= ref_sum_sqd;
  const float cmp_sum_sqd = (cmp_sum * cmp_sum) * invWindowSize_qd;
  cmp_sigma_sqd -= cmp_sum_sqd;
  const float both_sum_mul = (ref_sum * cmp_sum) * invWindowSize_qd;
  sigma_both -= both_sum_mul;

  /* STEP 3. process numbers, do scale */

  /* all following 2^X * 2^X * windowSize^4 */
  const float a = 2.0f * both_sum_mul + C1;
  const float b = sigma_both + C2 / 2.0f;
  const float c = ref_sum_sqd + cmp_sum_sqd + C1;
  const float d = ref_sigma_sqd + cmp_sigma_sqd + C2;

  const float ssim_val = (a * b * 2.0f) / (c * d);

  return ssim_val;

} /* float calc_window_ssim_float(WINDOW_STATS * const pWnd, */

void load_4x4_windows_8u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;

  const uint8_t *pR = pSrc->ref;
  const ptrdiff_t refStride = pSrc->refStride;
  const uint8_t *pC = pSrc->cmp;
  const ptrdiff_t cmpStride = pSrc->cmpStride;

  for (size_t i = 0; i < num4x4Windows; ++i) {
    uint16_t ref_sum = 0;
    uint16_t cmp_sum = 0;
    uint32_t ref_sigma_sqd = 0;
    uint32_t cmp_sigma_sqd = 0;
    uint32_t sigma_both = 0;

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      for (size_t x = 0; x < WIN_SIZE; ++x) {
        ref_sum += pR[x];
        cmp_sum += pC[x];
        ref_sigma_sqd += pR[x] * pR[x];
        cmp_sigma_sqd += pC[x] * pC[x];
        sigma_both += pR[x] * pC[x];
      }

      pR = AdvancePointer(pR, refStride);
      pC = AdvancePointer(pC, cmpStride);
    }

    ((uint16_t *)(pDst + 0 * dstStride))[0] = ref_sum;
    ((uint16_t *)(pDst + 0 * dstStride))[1] = cmp_sum;
    ((uint32_t *)(pDst + 1 * dstStride))[0] = ref_sigma_sqd;
    ((uint32_t *)(pDst + 2 * dstStride))[0] = cmp_sigma_sqd;
    ((uint32_t *)(pDst + 3 * dstStride))[0] = sigma_both;
    pDst += sizeof(uint32_t);

    /* set the next window */
    pR = AdvancePointer(pR + WIN_SIZE, -WIN_SIZE * refStride);
    pC = AdvancePointer(pC + WIN_SIZE, -WIN_SIZE * cmpStride);
  }

} /* void load_4x4_windows_8u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

#if NEW_10BIT_C_FUNC
void load_4x4_windows_10u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;

  const uint16_t *pR = pSrc->ref;
  const ptrdiff_t refStride = pSrc->refStride;
  const uint16_t *pC = pSrc->cmp;
  const ptrdiff_t cmpStride = pSrc->cmpStride;

  for (size_t i = 0; i < num4x4Windows; ++i) {
    uint16_t ref_sum = 0;
    uint16_t cmp_sum = 0;
    uint32_t ref_sigma_sqd = 0;
    uint32_t cmp_sigma_sqd = 0;
    uint32_t sigma_both = 0;

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      for (size_t x = 0; x < WIN_SIZE; ++x) {
        ref_sum += pR[x];
        cmp_sum += pC[x];
        ref_sigma_sqd += (uint32_t)pR[x] * pR[x];
        cmp_sigma_sqd += (uint32_t)pC[x] * pC[x];
        sigma_both += (uint32_t)pR[x] * pC[x];
      }

      pR = AdvancePointer(pR, refStride);
      pC = AdvancePointer(pC, cmpStride);
    }
    ((uint16_t *)(pDst + 0 * dstStride))[0] = ref_sum;
    ((uint16_t *)(pDst + 0 * dstStride))[1] = cmp_sum;
    ((uint32_t *)(pDst + 1 * dstStride))[0] = ref_sigma_sqd;
    ((uint32_t *)(pDst + 2 * dstStride))[0] = cmp_sigma_sqd;
    ((uint32_t *)(pDst + 3 * dstStride))[0] = sigma_both;
    pDst += sizeof(uint32_t);

    /* set the next window */
    pR = AdvancePointer(pR + WIN_SIZE, -WIN_SIZE * refStride);
    pC = AdvancePointer(pC + WIN_SIZE, -WIN_SIZE * cmpStride);
  }

} /* void load_4x4_windows_10u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS) */
#endif

void load_4x4_windows_16u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;

  const uint16_t *pR = pSrc->ref;
  const ptrdiff_t refStride = pSrc->refStride;
  const uint16_t *pC = pSrc->cmp;
  const ptrdiff_t cmpStride = pSrc->cmpStride;

  for (size_t i = 0; i < num4x4Windows; ++i) {
    uint32_t ref_sum = 0;
    uint32_t cmp_sum = 0;
    uint64_t ref_sigma_sqd = 0;
    uint64_t cmp_sigma_sqd = 0;
    uint64_t sigma_both = 0;

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      for (size_t x = 0; x < WIN_SIZE; ++x) {
        ref_sum += pR[x];
        cmp_sum += pC[x];
        ref_sigma_sqd += (uint64_t)pR[x] * pR[x];
        cmp_sigma_sqd += (uint64_t)pC[x] * pC[x];
        sigma_both += (uint64_t)pR[x] * pC[x];
      }

      pR = AdvancePointer(pR, refStride);
      pC = AdvancePointer(pC, cmpStride);
    }

    ((uint32_t *)(pDst + 0 * dstStride))[0] = ref_sum;
    ((uint32_t *)(pDst + 0 * dstStride))[1] = cmp_sum;
    ((uint64_t *)(pDst + 1 * dstStride))[0] = ref_sigma_sqd;
    ((uint64_t *)(pDst + 2 * dstStride))[0] = cmp_sigma_sqd;
    ((uint64_t *)(pDst + 3 * dstStride))[0] = sigma_both;
    pDst += sizeof(uint64_t);

    /* set the next window */
    pR = AdvancePointer(pR + WIN_SIZE, -WIN_SIZE * refStride);
    pC = AdvancePointer(pC + WIN_SIZE, -WIN_SIZE * cmpStride);
  }

} /* void load_4x4_windows_16u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

void sum_windows_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  UNUSED(bitDepthMinus8);

  const uint32_t windowSizeDiv4 = windowSize / 4;

  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);

  int64_t ssim_mink_sum = 0, ssim_sum = 0;

  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  int64_t const_1 =
      1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);

  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  for (size_t i = 0; i < numWindows; ++i) {
     WINDOW_STATS wnd = {0};
    /* STEP 1. load data, no scaling */

    /* windows are summed up from 4x4 regions */
    for (uint32_t y = 0; y < windowSizeDiv4; ++y) {
      for (size_t x = 0; x < windowSizeDiv4; ++x) {
        wnd.ref_sum += (((uint16_t *)(pSrc + 0 * srcStride))[2 * x + 0]);
        wnd.cmp_sum += (((uint16_t *)(pSrc + 0 * srcStride))[2 * x + 1]);
        wnd.ref_sigma_sqd += ((uint32_t *)(pSrc + 1 * srcStride))[x];
        wnd.cmp_sigma_sqd += ((uint32_t *)(pSrc + 2 * srcStride))[x];
        wnd.sigma_both += ((uint32_t *)(pSrc + 3 * srcStride))[x];
      }

      pSrc = AdvancePointer(pSrc, 4 * srcStride);
    }

    /* set the next window */
    pSrc = AdvancePointer(pSrc + windowStride, -srcStride * windowSize);

    const int64_t ssim_val = calc_window_ssim_int_8u(&wnd, windowSize, C1, C2,
                                                    div_lookup_ptr, SSIMValRtShiftBits,
                                                    SSIMValRtShiftHalfRound);
    ssim_sum += ssim_val;

 #if DEBUG_PRINTS
    if(const_1 < abs((int)ssim_val)) {
      printf("WARNING: Overflow can happen in ssim_mink_sum");
    }
 #endif

    int64_t const_1_minus_ssim_val = const_1 - ssim_val;
    if(essim_mink_value == 4) {
      mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val
                          * const_1_minus_ssim_val* const_1_minus_ssim_val;
    } else {
      /*essim_mink_value == 3*/
      mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val
                          * const_1_minus_ssim_val;
    }
    ssim_mink_sum += mink_pow_ssim_val;
  }

  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += numWindows;

} /* void sum_windows_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) */

void sum_windows_8x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_12x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

#if NEW_SIMD_FUNC
void sum_windows_8x8_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x8_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x16_int_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
#if NEW_10BIT_C_FUNC
void sum_windows_8x4_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_10u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_8x8_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_10u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x4_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_10u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x8_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_10u_c(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x16_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_int_10u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  const uint32_t windowSizeDiv4 = windowSize / 4;

  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);

  int64_t ssim_mink_sum = 0, ssim_sum = 0;

  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;

  int64_t const_1 =
            1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);

  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  for (size_t i = 0; i < numWindows; ++i) {
    WINDOW_STATS wnd = {0};

    /* STEP 1. load data, no scaling */

    /* windows are summed up from 4x4 regions */
    for (uint32_t y = 0; y < windowSizeDiv4; ++y) {
      for (size_t x = 0; x < windowSizeDiv4; ++x) {
        wnd.ref_sum += ((uint16_t *)(pSrc + 0 * srcStride))[2 * x + 0];
        wnd.cmp_sum += ((uint16_t *)(pSrc + 0 * srcStride))[2 * x + 1];
        wnd.ref_sigma_sqd += ((uint32_t *)(pSrc + 1 * srcStride))[x];
        wnd.cmp_sigma_sqd += ((uint32_t *)(pSrc + 2 * srcStride))[x];
        wnd.sigma_both += ((uint32_t *)(pSrc + 3 * srcStride))[x];
      }

      pSrc = AdvancePointer(pSrc, 4 * srcStride);
    }
    /* set the next window */
    pSrc = AdvancePointer(pSrc + windowStride, - srcStride * windowSize);

    const int64_t ssim_val = calc_window_ssim_int_10bd(&wnd, windowSize, C1, C2,
                             div_lookup_ptr, SSIMValRtShiftBits,
                             SSIMValRtShiftHalfRound);

    ssim_sum += ssim_val;
 #if DEBUG_PRINTS
    if(const_1 < abs((int)ssim_val)) {
      printf("WARNING: Overflow can happen in ssim_mink_sum");
    }
 #endif

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

  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += numWindows;

} /* void sum_windows_int_10u_c(SUM_WINDOWS_FORMAL_ARGS) */
#endif

#endif
void sum_windows_int_16u_c(SUM_WINDOWS_FORMAL_ARGS) {
  const uint32_t windowSizeDiv4 = windowSize / 4;

  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const calc_window_ssim_proc_t calc_window_ssim_proc = (bitDepthMinus8==2) ?
                                                        (calc_window_ssim_int_10bd):
                                                        (calc_window_ssim_int_16u);
  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  int64_t const_1 =
      1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);
  int64_t ssim_mink_sum = 0, ssim_sum = 0;

  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;
  const size_t windowStep = windowStride * sizeof(uint16_t);

  for (size_t i = 0; i < numWindows; ++i) {
    WINDOW_STATS wnd = {0};

    /* STEP 1. load data, no scaling */

    /* windows are summed up from 4x4 regions */
    for (uint32_t y = 0; y < windowSizeDiv4; ++y) {
      for (size_t x = 0; x < windowSizeDiv4; ++x) {
        wnd.ref_sum += ((uint32_t *)(pSrc + 0 * srcStride))[2 * x + 0];
        wnd.cmp_sum += ((uint32_t *)(pSrc + 0 * srcStride))[2 * x + 1];
        wnd.ref_sigma_sqd += ((uint64_t *)(pSrc + 1 * srcStride))[x];
        wnd.cmp_sigma_sqd += ((uint64_t *)(pSrc + 2 * srcStride))[x];
        wnd.sigma_both += ((uint64_t *)(pSrc + 3 * srcStride))[x];
      }

      pSrc = AdvancePointer(pSrc, 4 * srcStride);
    }

    /* set the next window */
    pSrc = AdvancePointer(pSrc, windowStep - srcStride * windowSize);
    const int64_t ssim_val = calc_window_ssim_proc(&wnd, windowSize, C1, C2,
                             div_lookup_ptr, SSIMValRtShiftBits,
                             SSIMValRtShiftHalfRound);
    ssim_sum += ssim_val;

 #if DEBUG_PRINTS
    if(const_1 < abs((int)ssim_val)) {
      printf("WARNING: Overflow can happen in ssim_mink_sum");
    }
 #endif
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

  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += numWindows;

} /* void sum_windows_int_16u_c(SUM_WINDOWS_FORMAL_ARGS) */

void sum_windows_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  const uint32_t windowSizeDiv4 = windowSize / 4;

  const float C1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float C2 = get_ssim_float_constant(2, bitDepthMinus8);

  double ssim_mink_sum = 0, ssim_sum = 0;

  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  for (size_t i = 0; i < numWindows; ++i) {
    WINDOW_STATS wnd = {0};

    /* STEP 1. load data, no scaling */

    /* windows are summed up from 4x4 regions */
    for (uint32_t y = 0; y < windowSizeDiv4; ++y) {
      for (size_t x = 0; x < windowSizeDiv4; ++x) {
        wnd.ref_sum += ((uint16_t *)(pSrc + 0 * srcStride))[2 * x + 0];
        wnd.cmp_sum += ((uint16_t *)(pSrc + 0 * srcStride))[2 * x + 1];
        wnd.ref_sigma_sqd += ((uint32_t *)(pSrc + 1 * srcStride))[x];
        wnd.cmp_sigma_sqd += ((uint32_t *)(pSrc + 2 * srcStride))[x];
        wnd.sigma_both += ((uint32_t *)(pSrc + 3 * srcStride))[x];
      }

      pSrc = AdvancePointer(pSrc, 4 * srcStride);
    }

    /* set the next window */
    pSrc = AdvancePointer(pSrc + windowStride, -srcStride * windowSize);

    const float ssim_val = calc_window_ssim_float(&wnd, windowSize, C1, C2);

    ssim_sum += ssim_val;
    ssim_mink_sum += pow(1 - ssim_val, essim_mink_value);
  }

  res->ssim_sum_f += ssim_sum;
  res->ssim_mink_sum_f += ssim_mink_sum;
  res->numWindows += numWindows;

} /* void sum_windows_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) */

void sum_windows_8x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_12x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_8x8_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x8_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x16_float_8u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_8x4_float_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_8x8_float_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x4_float_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x8_float_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x16_float_10u_c(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_float_8u_c(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_float_16u_c(SUM_WINDOWS_FORMAL_ARGS) {
  const uint32_t windowSizeDiv4 = windowSize / 4;

  const float C1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float C2 = get_ssim_float_constant(2, bitDepthMinus8);

  double ssim_mink_sum = 0;
  float ssim_sum = 0;

  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;
  const size_t windowStep = windowStride * sizeof(uint16_t);

  for (size_t i = 0; i < numWindows; ++i) {
    WINDOW_STATS wnd = {0};

    /* STEP 1. load data, no scaling */

    /* windows are summed up from 4x4 regions */
    for (uint32_t y = 0; y < windowSizeDiv4; ++y) {
      for (size_t x = 0; x < windowSizeDiv4; ++x) {
        wnd.ref_sum += ((uint32_t *)(pSrc + 0 * srcStride))[2 * x + 0];
        wnd.cmp_sum += ((uint32_t *)(pSrc + 0 * srcStride))[2 * x + 1];
        wnd.ref_sigma_sqd += ((uint64_t *)(pSrc + 1 * srcStride))[x];
        wnd.cmp_sigma_sqd += ((uint64_t *)(pSrc + 2 * srcStride))[x];
        wnd.sigma_both += ((uint64_t *)(pSrc + 3 * srcStride))[x];
      }

      pSrc = AdvancePointer(pSrc, 4 * srcStride);
    }

    /* set the next window */
    pSrc = AdvancePointer(pSrc, windowStep - srcStride * windowSize);

    const float ssim_val = calc_window_ssim_float(&wnd, windowSize, C1, C2);
    ssim_sum += ssim_val;
    ssim_mink_sum += pow((1 - ssim_val), essim_mink_value);
  }

  res->ssim_sum_f += ssim_sum;
  res->ssim_mink_sum_f += ssim_mink_sum;
  res->numWindows += numWindows;

} /* void sum_windows_float_16u_c(SUM_WINDOWS_FORMAL_ARGS) */
