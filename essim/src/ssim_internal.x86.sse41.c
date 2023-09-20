/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/inc/internal.h>

#if defined(_X86) || defined(_X64)

#if defined(_WINDOWS)
#include <intrin.h>
#elif defined(_LINUX)
#include <x86intrin.h>
#endif /* defined(_WINDOWS) */

/* GCC bug fix */
#define _mm_loadu_si32(addr) _mm_cvtsi32_si128(*((uint32_t *)(addr)))

#define ASM_LOAD_8X8_WINDOW_8_WORD_VALUES_SSSE3(value, idx)                    \
  {                                                                            \
    __m128i _r;                                                                \
    value = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));         \
    _r = _mm_loadu_si32(pSrc + (idx)*srcStride + 16);                          \
    value = _mm_add_epi16(                                                     \
        value, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride))); \
    _r = _mm_add_epi16(_r, _mm_loadu_si32(pSrcNext + (idx)*srcStride + 16));   \
    _r = _mm_alignr_epi8(_r, value, 8);                                        \
    value = _mm_shuffle_epi8(value, c_sum_shuffle_pattern);                    \
    _r = _mm_shuffle_epi8(_r, c_sum_shuffle_pattern);                          \
    value = _mm_hadd_epi16(value, _r);                                         \
  }

#define ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_SSSE3(value, idx)                   \
  {                                                                            \
    __m128i _r;                                                                \
    value = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));         \
    _r = _mm_loadu_si32(pSrc + (idx)*srcStride + 16);                          \
    value = _mm_add_epi32(                                                     \
        value, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride))); \
    _r = _mm_add_epi32(_r, _mm_loadu_si32(pSrcNext + (idx)*srcStride + 16));   \
    _r = _mm_alignr_epi8(_r, value, 8);                                        \
    value = _mm_shuffle_epi32(value, 0x94);                                    \
    _r = _mm_shuffle_epi32(_r, 0x94);                                          \
    value = _mm_hadd_epi32(value, _r);                                         \
    value = _mm_mullo_epi32(value, windowSize_sqd);                            \
  }

#define ASM_CALC_4_QDWORD_SSIM_SSE41(num, denom)                               \
  {                                                                            \
    __m128i a, b, c, d;                                                        \
    /* STEP 2. adjust values */                                                \
    __m128i _r0 = _mm_srli_epi32(_mm_slli_epi32(sum, 16), 16);                 \
    __m128i _r1 = _mm_srli_epi32(sum, 16);                                     \
    __m128i both_sum_mul = _mm_mullo_epi32(_r0, _r1);                          \
    __m128i ref_sum_sqd = _mm_mullo_epi32(_r0, _r0);                           \
    __m128i cmp_sum_sqd = _mm_mullo_epi32(_r1, _r1);                           \
    ref_sigma_sqd = _mm_sub_epi32(ref_sigma_sqd, ref_sum_sqd);                 \
    cmp_sigma_sqd = _mm_sub_epi32(cmp_sigma_sqd, cmp_sum_sqd);                 \
    sigma_both = _mm_srli_epi32(sigma_both, 1);                                \
    sigma_both = _mm_sub_epi32(sigma_both, _mm_srli_epi32(both_sum_mul, 1));   \
    /* STEP 3. process numbers, do scale */                                    \
    a = _mm_add_epi32(both_sum_mul, halfC1);                                   \
    b = _mm_add_epi32(sigma_both, quarterC2);                                  \
    ref_sum_sqd = _mm_srli_epi32(ref_sum_sqd, 1);                              \
    cmp_sum_sqd = _mm_srli_epi32(cmp_sum_sqd, 1);                              \
    c = _mm_add_epi32(_mm_add_epi32(ref_sum_sqd, cmp_sum_sqd), halfC1);        \
    ref_sigma_sqd = _mm_srli_epi32(ref_sigma_sqd, 1);                          \
    cmp_sigma_sqd = _mm_srli_epi32(cmp_sigma_sqd, 1);                          \
    d = _mm_add_epi32(_mm_add_epi32(ref_sigma_sqd, cmp_sigma_sqd), halfC2);    \
    /* process numerators */                                                   \
    _r0 = _mm_unpackhi_epi32(a, zero);                                         \
    a = _mm_cvtepu32_epi64(a);                                                 \
    _r1 = _mm_unpackhi_epi32(b, zero);                                         \
    b = _mm_cvtepu32_epi64(b);                                                 \
    a = _mm_mul_epi32(a, b);                                                   \
    _r0 = _mm_mul_epi32(_r0, _r1);                                             \
    _mm_storeu_si128((__m128i *)num + 0, a);                                   \
    _mm_storeu_si128((__m128i *)num + 1, _r0);                                 \
    /* process denominators */                                                 \
    _r0 = _mm_unpackhi_epi32(c, zero);                                         \
    c = _mm_cvtepu32_epi64(c);                                                 \
    _r1 = _mm_unpackhi_epi32(d, zero);                                         \
    d = _mm_cvtepu32_epi64(d);                                                 \
    c = _mm_mul_epi32(c, d);                                                   \
    _r0 = _mm_mul_epi32(_r0, _r1);                                             \
    c = _mm_srli_epi64(c, SSIM_LOG2_SCALE + 1);                                \
    _r0 = _mm_srli_epi64(_r0, SSIM_LOG2_SCALE + 1);                            \
    _mm_storeu_si128((__m128i *)denom + 0, c);                                 \
    _mm_storeu_si128((__m128i *)denom + 1, _r0);                               \
  }

static const int8_t sum_8x8_shuffle[16] = {0, 1, 4, 5, 2, 3, 6,  7,
                                           4, 5, 8, 9, 6, 7, 10, 11};

void sum_windows_8x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 8 };

  const __m128i zero = _mm_setzero_si128();
  const __m128i c_sum_shuffle_pattern =
      _mm_loadu_si128((const __m128i *)sum_8x8_shuffle);
  const __m128i windowSize_sqd = _mm_set1_epi32(windowSize * windowSize);
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const __m128i halfC1 = _mm_set1_epi32(C1 / 2);
  const __m128i halfC2 = _mm_set1_epi32(C2 / 2);
  const __m128i quarterC2 = _mm_set1_epi32(C2 / 4);

  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;

    __m128i sum;
    __m128i ref_sigma_sqd;
    __m128i cmp_sigma_sqd;
    __m128i sigma_both;

    ASM_LOAD_8X8_WINDOW_8_WORD_VALUES_SSSE3(sum, 0);
    ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_SSSE3(ref_sigma_sqd, 1);
    ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_SSSE3(cmp_sigma_sqd, 2);
    ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_SSSE3(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_4_QDWORD_SSIM_SSE41(num, denom);

    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      const int64_t ssim_val = (num[w] + denom[w] / 2) / (denom[w] | 1);

      ssim_sum += ssim_val;
      ssim_mink_sum +=
          (int64_t)ssim_val * ssim_val; // TODO replace with (1 - ssim) ** 4
    }
  }

  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum_f += ssim_mink_sum;
  res->numWindows += i;

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_8x4_int_8u_c(res, &buf, numWindows - i, windowSize,
                             windowStride, bitDepthMinus8, div_lookup_ptr,
                             SSIMValRtShiftBits, SSIMValRtShiftHalfRound,
                             essim_mink_value);
  }

} /* void sum_windows_8x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS) */

#define ASM_LOAD_12X12_WINDOW_8_WORD_VALUES_SSSE3(value, idx)                  \
  {                                                                            \
    __m128i _r0, _r1, _r2;                                                     \
    value = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));         \
    _r2 = _mm_loadl_epi64((const __m128i *)(pSrc + (idx)*srcStride + 16));     \
    value = _mm_add_epi16(                                                     \
        value, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride))); \
    _r2 = _mm_add_epi16(                                                       \
        _r2,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext + (idx)*srcStride + 16)));  \
    value = _mm_add_epi16(                                                     \
        value,                                                                 \
        _mm_load_si128((const __m128i *)(pSrcNext2 + (idx)*srcStride)));       \
    _r2 = _mm_add_epi16(                                                       \
        _r2,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext2 + (idx)*srcStride + 16))); \
    _r0 = _mm_srli_si128(value, 4);                                            \
    _r1 = _mm_alignr_epi8(_r2, value, 8);                                      \
    _r2 = _mm_alignr_epi8(_r2, value, 12);                                     \
    value = _mm_shuffle_epi8(value, c_sum_shuffle_pattern);                    \
    _r0 = _mm_shuffle_epi8(_r0, c_sum_shuffle_pattern);                        \
    _r1 = _mm_shuffle_epi8(_r1, c_sum_shuffle_pattern);                        \
    _r2 = _mm_shuffle_epi8(_r2, c_sum_shuffle_pattern);                        \
    value = _mm_hadd_epi16(value, _r0);                                        \
    _r1 = _mm_hadd_epi16(_r1, _r2);                                            \
    value = _mm_hadd_epi16(value, _r1);                                        \
  }

#define ASM_LOAD_12X12_WINDOW_4_DWORD_VALUES_SSSE3(value, idx)                 \
  {                                                                            \
    __m128i _r0, _r1, _r2;                                                     \
    value = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));         \
    _r2 = _mm_loadl_epi64((const __m128i *)(pSrc + (idx)*srcStride + 16));     \
    value = _mm_add_epi32(                                                     \
        value, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride))); \
    _r2 = _mm_add_epi32(                                                       \
        _r2,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext + (idx)*srcStride + 16)));  \
    value = _mm_add_epi32(                                                     \
        value,                                                                 \
        _mm_load_si128((const __m128i *)(pSrcNext2 + (idx)*srcStride)));       \
    _r2 = _mm_add_epi32(                                                       \
        _r2,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext2 + (idx)*srcStride + 16))); \
    _r0 = _mm_srli_si128(value, 4);                                            \
    _r1 = _mm_alignr_epi8(_r2, value, 8);                                      \
    _r1 = _mm_slli_si128(_r1, 4);                                              \
    _r2 = _mm_alignr_epi8(_r2, value, 12);                                     \
    value = _mm_slli_si128(value, 4);                                          \
    value = _mm_hadd_epi32(value, _r0);                                        \
    _r1 = _mm_hadd_epi32(_r1, _r2);                                            \
    value = _mm_hadd_epi32(value, _r1);                                        \
    value = _mm_mullo_epi32(value, windowSize_sqd);                            \
  }

static const int8_t sum_12x12_shuffle[16] = {0, 1, 4, 5, 8,  9,  -1, -1,
                                             2, 3, 6, 7, 10, 11, -1, -1};

void sum_windows_12x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 12 };

  const __m128i zero = _mm_setzero_si128();
  const __m128i c_sum_shuffle_pattern =
      _mm_loadu_si128((const __m128i *)sum_12x12_shuffle);
  const __m128i windowSize_sqd = _mm_set1_epi32(windowSize * windowSize);
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const __m128i halfC1 = _mm_set1_epi32(C1 / 2);
  const __m128i halfC2 = _mm_set1_epi32(C2 / 2);
  const __m128i quarterC2 = _mm_set1_epi32(C2 / 4);

  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;
    const uint8_t *pSrcNext2 = pSrc + 8 * srcStride;

    __m128i sum;
    __m128i ref_sigma_sqd;
    __m128i cmp_sigma_sqd;
    __m128i sigma_both;

    ASM_LOAD_12X12_WINDOW_8_WORD_VALUES_SSSE3(sum, 0);
    ASM_LOAD_12X12_WINDOW_4_DWORD_VALUES_SSSE3(ref_sigma_sqd, 1);
    ASM_LOAD_12X12_WINDOW_4_DWORD_VALUES_SSSE3(cmp_sigma_sqd, 2);
    ASM_LOAD_12X12_WINDOW_4_DWORD_VALUES_SSSE3(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_4_QDWORD_SSIM_SSE41(num, denom);

    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      const int64_t ssim_val = (num[w] + denom[w] / 2) / (denom[w] | 1);

      ssim_sum += ssim_val;
      ssim_mink_sum +=
          (int64_t)ssim_val * ssim_val; // TODO replace with (1 - ssim) ** 4
    }
  }

  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += i;

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_8x4_int_8u_c(res, &buf, numWindows - i, windowSize,
                             windowStride, bitDepthMinus8, div_lookup_ptr,
                             SSIMValRtShiftBits, SSIMValRtShiftHalfRound,
                             essim_mink_value);
  }

} /* void sum_windows_12x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS) */

#endif /* defined(_X86) || defined(_X64) */
