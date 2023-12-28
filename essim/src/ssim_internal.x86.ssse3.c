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
#define _mm_storeu_si32(addr, r) *((uint32_t *)(addr)) = _mm_cvtsi128_si32(r)

void load_4x4_windows_8u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { PREFETCH = 160, WIN_CHUNK = 4, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint8_t) * WIN_SIZE;

  const __m128i zero = _mm_setzero_si128();

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    __m128i sum = _mm_setzero_si128();
    __m128i ref_sigma_sqd = _mm_setzero_si128();
    __m128i cmp_sigma_sqd = _mm_setzero_si128();
    __m128i sigma_both = _mm_setzero_si128();

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      __m128i r0, r1, r2, r3, r4, r5, r6;

      _mm_prefetch((const char *)src.ref + PREFETCH, _MM_HINT_NTA);
      r0 = _mm_loadu_si128((const __m128i *)(src.ref));
      r1 = _mm_loadu_si128((const __m128i *)(src.cmp));

      /* sum ref_sum & cmp_sum */
      r2 = _mm_unpackhi_epi8(r0, zero);
      r3 = _mm_unpackhi_epi8(r1, zero);
      r0 = _mm_unpacklo_epi8(r0, zero);
      r1 = _mm_unpacklo_epi8(r1, zero);
      r4 = _mm_sad_epu8(r0, zero);
      r5 = _mm_sad_epu8(r1, zero);
      r5 = _mm_slli_epi64(r5, 32);
      r4 = _mm_or_si128(r4, r5);
      r5 = _mm_sad_epu8(r2, zero);
      r6 = _mm_sad_epu8(r3, zero);
      r6 = _mm_slli_epi64(r6, 32);
      r5 = _mm_or_si128(r5, r6);
      r4 = _mm_packs_epi32(r4, r5);
      sum = _mm_add_epi16(sum, r4);

      _mm_prefetch((const char *)src.cmp + PREFETCH, _MM_HINT_NTA);
      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      r4 = r0;
      r5 = r2;
      r0 = _mm_madd_epi16(r0, r0);
      r4 = _mm_madd_epi16(r4, r1);
      r1 = _mm_madd_epi16(r1, r1);
      r2 = _mm_madd_epi16(r2, r2);
      r5 = _mm_madd_epi16(r5, r3);
      r3 = _mm_madd_epi16(r3, r3);
      r0 = _mm_hadd_epi32(r0, r2);
      r1 = _mm_hadd_epi32(r1, r3);
      r4 = _mm_hadd_epi32(r4, r5);
      ref_sigma_sqd = _mm_add_epi32(ref_sigma_sqd, r0);
      cmp_sigma_sqd = _mm_add_epi32(cmp_sigma_sqd, r1);
      sigma_both = _mm_add_epi32(sigma_both, r4);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    _mm_store_si128((__m128i *)(pDst + 0 * dstStride), sum);
    _mm_store_si128((__m128i *)(pDst + 1 * dstStride), ref_sigma_sqd);
    _mm_store_si128((__m128i *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    _mm_store_si128((__m128i *)(pDst + 3 * dstStride), sigma_both);
    pDst = AdvancePointer(pDst, WIN_CHUNK * sizeof(uint32_t));

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }

  for (; i < num4x4Windows; i += 1) {
    __m128i sum = _mm_setzero_si128();
    __m128i ref_sigma_sqd = _mm_setzero_si128();
    __m128i cmp_sigma_sqd = _mm_setzero_si128();
    __m128i sigma_both = _mm_setzero_si128();

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      __m128i r0, r1, r4, r5;

      r0 = _mm_loadu_si32(src.ref);
      r1 = _mm_loadu_si32(src.cmp);

      /* sum ref_sum & cmp_sum */
      r0 = _mm_unpacklo_epi8(r0, zero);
      r1 = _mm_unpacklo_epi8(r1, zero);
      r4 = _mm_sad_epu8(r0, zero);
      r5 = _mm_sad_epu8(r1, zero);
      r5 = _mm_slli_epi64(r5, 16);
      r4 = _mm_or_si128(r4, r5);
      sum = _mm_add_epi16(sum, r4);

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      r4 = r0;
      r0 = _mm_madd_epi16(r0, r0);
      r4 = _mm_madd_epi16(r4, r1);
      r1 = _mm_madd_epi16(r1, r1);
      r0 = _mm_hadd_epi32(r0, r0);
      r1 = _mm_hadd_epi32(r1, r1);
      r4 = _mm_hadd_epi32(r4, r4);
      ref_sigma_sqd = _mm_add_epi32(ref_sigma_sqd, r0);
      cmp_sigma_sqd = _mm_add_epi32(cmp_sigma_sqd, r1);
      sigma_both = _mm_add_epi32(sigma_both, r4);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    _mm_storeu_si32((pDst + 0 * dstStride), sum);
    _mm_storeu_si32((pDst + 1 * dstStride), ref_sigma_sqd);
    _mm_storeu_si32((pDst + 2 * dstStride), cmp_sigma_sqd);
    _mm_storeu_si32((pDst + 3 * dstStride), sigma_both);
    pDst = AdvancePointer(pDst, sizeof(uint32_t));

    /* advance source pointers */
    src.ref = AdvancePointer(src.ref, srcStep - WIN_SIZE * src.refStride);
    src.cmp = AdvancePointer(src.cmp, srcStep - WIN_SIZE * src.cmpStride);
  }

} /* void load_4x4_windows_8u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

static const uint8_t lower_dword_mask[16] = {255, 255, 255, 255, 0, 0, 0, 0,
                                             255, 255, 255, 255, 0, 0, 0, 0};

enum { SWAP_MIDDLE_DWORD = 0xd8 };

void load_4x4_windows_16u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { PREFETCH = 160, WIN_CHUNK = 2, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint16_t) * WIN_SIZE;

  const __m128i mask = _mm_loadu_si128((const __m128i *)lower_dword_mask);
  const __m128i ones = _mm_set1_epi16(1);

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    __m128i sum = _mm_setzero_si128();
    __m128i ref_sigma_sqd = _mm_setzero_si128();
    __m128i cmp_sigma_sqd = _mm_setzero_si128();
    __m128i sigma_both = _mm_setzero_si128();

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      __m128i r0, r1, r2, r3, r4, r5;

      _mm_prefetch((const char *)src.ref + PREFETCH, _MM_HINT_NTA);
      r0 = _mm_loadu_si128((const __m128i *)(src.ref));
      r2 = _mm_loadu_si128((const __m128i *)(src.cmp));

      /* sum ref_sum & cmp_sum */
      r1 = _mm_madd_epi16(r0, ones);
      r3 = _mm_madd_epi16(r2, ones);
      r1 = _mm_hadd_epi32(r1, r3);
      r1 = _mm_shuffle_epi32(r1, SWAP_MIDDLE_DWORD);
      sum = _mm_add_epi32(sum, r1);

      _mm_prefetch((const char *)src.cmp + PREFETCH, _MM_HINT_NTA);
      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      r1 = r0;
      r0 = _mm_madd_epi16(r0, r0);
      r1 = _mm_madd_epi16(r1, r2);
      r2 = _mm_madd_epi16(r2, r2);
      r3 = _mm_srli_epi64(r0, 32);
      r0 = _mm_and_si128(r0, mask);
      r0 = _mm_add_epi64(r0, r3);
      r4 = _mm_srli_epi64(r1, 32);
      r1 = _mm_and_si128(r1, mask);
      r1 = _mm_add_epi64(r1, r4);
      r5 = _mm_srli_epi64(r2, 32);
      r2 = _mm_and_si128(r2, mask);
      r2 = _mm_add_epi64(r2, r5);
      ref_sigma_sqd = _mm_add_epi64(ref_sigma_sqd, r0);
      cmp_sigma_sqd = _mm_add_epi64(cmp_sigma_sqd, r2);
      sigma_both = _mm_add_epi64(sigma_both, r1);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    _mm_store_si128((__m128i *)(pDst + 0 * dstStride), sum);
    _mm_store_si128((__m128i *)(pDst + 1 * dstStride), ref_sigma_sqd);
    _mm_store_si128((__m128i *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    _mm_store_si128((__m128i *)(pDst + 3 * dstStride), sigma_both);
    pDst = AdvancePointer(pDst, WIN_CHUNK * sizeof(uint64_t));

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }

  for (; i < num4x4Windows; i += 1) {
    __m128i sum = _mm_setzero_si128();
    __m128i ref_sigma_sqd = _mm_setzero_si128();
    __m128i cmp_sigma_sqd = _mm_setzero_si128();
    __m128i sigma_both = _mm_setzero_si128();

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      __m128i r0, r1, r2, r3, r4, r5;

      r0 = _mm_loadl_epi64((const __m128i *)(src.ref));
      r2 = _mm_loadl_epi64((const __m128i *)(src.cmp));

      /* sum ref_sum & cmp_sum */
      r1 = _mm_madd_epi16(r0, ones);
      r3 = _mm_madd_epi16(r2, ones);
      r1 = _mm_hadd_epi32(r1, r3);
      r1 = _mm_shuffle_epi32(r1, SWAP_MIDDLE_DWORD);
      sum = _mm_add_epi32(sum, r1);

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      r1 = r0;
      r0 = _mm_madd_epi16(r0, r0);
      r1 = _mm_madd_epi16(r1, r2);
      r2 = _mm_madd_epi16(r2, r2);
      r3 = _mm_srli_epi64(r0, 32);
      r0 = _mm_and_si128(r0, mask);
      r0 = _mm_add_epi64(r0, r3);
      r4 = _mm_srli_epi64(r1, 32);
      r1 = _mm_and_si128(r1, mask);
      r1 = _mm_add_epi64(r1, r4);
      r5 = _mm_srli_epi64(r2, 32);
      r2 = _mm_and_si128(r2, mask);
      r2 = _mm_add_epi64(r2, r5);
      ref_sigma_sqd = _mm_add_epi64(ref_sigma_sqd, r0);
      cmp_sigma_sqd = _mm_add_epi64(cmp_sigma_sqd, r2);
      sigma_both = _mm_add_epi64(sigma_both, r1);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    _mm_storel_epi64((__m128i *)(pDst + 0 * dstStride), sum);
    _mm_storel_epi64((__m128i *)(pDst + 1 * dstStride), ref_sigma_sqd);
    _mm_storel_epi64((__m128i *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    _mm_storel_epi64((__m128i *)(pDst + 3 * dstStride), sigma_both);
    pDst = AdvancePointer(pDst, sizeof(uint64_t));

    /* advance source pointers */
    src.ref = AdvancePointer(src.ref, srcStep - WIN_SIZE * src.refStride);
    src.cmp = AdvancePointer(src.cmp, srcStep - WIN_SIZE * src.cmpStride);
  }

} /* void load_4x4_windows_16u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

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

static const int8_t sum_8x8_shuffle[16] = {0, 1, 4, 5, 2, 3, 6,  7,
                                           4, 5, 8, 9, 6, 7, 10, 11};

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

#define ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_SSSE3(value0, value1, idx)          \
  {                                                                            \
    __m128i _r0, _r1;                                                          \
    _r0 = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));           \
    _r1 = _mm_loadu_si32(pSrc + (idx)*srcStride + 16);                         \
    _r0 = _mm_add_epi16(                                                       \
        _r0, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride)));   \
    _r1 = _mm_add_epi16(_r1, _mm_loadu_si32(pSrcNext + (idx)*srcStride + 16)); \
    _r1 = _mm_alignr_epi8(_r1, _r0, 8);                                        \
    _r0 = _mm_shuffle_epi8(_r0, c_sum_shuffle_pattern);                        \
    _r1 = _mm_shuffle_epi8(_r1, c_sum_shuffle_pattern);                        \
    _r0 = _mm_hadd_epi16(_r0, _r1);                                            \
    _r1 = _mm_srli_epi32(_r0, 16);                                             \
    _r0 = _mm_srli_epi32(_mm_slli_epi32(_r0, 16), 16);                         \
    value0 = _mm_cvtepi32_ps(_r0);                                             \
    value1 = _mm_cvtepi32_ps(_r1);                                             \
  }

#define ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_SSSE3(value, idx)                   \
  {                                                                            \
    __m128i _r0, _r1;                                                          \
    _r0 = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));           \
    _r1 = _mm_loadu_si32(pSrc + (idx)*srcStride + 16);                         \
    _r0 = _mm_add_epi32(                                                       \
        _r0, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride)));   \
    _r1 = _mm_add_epi32(_r1, _mm_loadu_si32(pSrcNext + (idx)*srcStride + 16)); \
    _r1 = _mm_alignr_epi8(_r1, _r0, 8);                                        \
    _r0 = _mm_shuffle_epi32(_r0, 0x94);                                        \
    _r1 = _mm_shuffle_epi32(_r1, 0x94);                                        \
    _r0 = _mm_hadd_epi32(_r0, _r1);                                            \
    value = _mm_mul_ps(_mm_cvtepi32_ps(_r0), invWindowSize_sqd);               \
  }

#define ASM_CALC_4_FLOAT_SSIM_SSE()                                            \
  {                                                                            \
    __m128 one = _mm_set1_ps(1);                                               \
    /* STEP 2. adjust values */                                                \
    __m128 both_sum_mul =                                                      \
        _mm_mul_ps(_mm_mul_ps(ref_sum, cmp_sum), invWindowSize_qd);            \
    __m128 ref_sum_sqd =                                                       \
        _mm_mul_ps(_mm_mul_ps(ref_sum, ref_sum), invWindowSize_qd);            \
    __m128 cmp_sum_sqd =                                                       \
        _mm_mul_ps(_mm_mul_ps(cmp_sum, cmp_sum), invWindowSize_qd);            \
    ref_sigma_sqd = _mm_sub_ps(ref_sigma_sqd, ref_sum_sqd);                    \
    cmp_sigma_sqd = _mm_sub_ps(cmp_sigma_sqd, cmp_sum_sqd);                    \
    sigma_both = _mm_sub_ps(sigma_both, both_sum_mul);                         \
    /* STEP 3. process numbers, do scale */                                    \
    {                                                                          \
      __m128 a = _mm_add_ps(_mm_add_ps(both_sum_mul, both_sum_mul), C1);       \
      __m128 b = _mm_add_ps(sigma_both, halfC2);                               \
      __m128 c = _mm_add_ps(_mm_add_ps(ref_sum_sqd, cmp_sum_sqd), C1);         \
      __m128 d = _mm_add_ps(_mm_add_ps(ref_sigma_sqd, cmp_sigma_sqd), C2);     \
      __m128 ssim_val = _mm_mul_ps(a, b);                                      \
      ssim_val = _mm_add_ps(ssim_val, ssim_val);                               \
      ssim_val = _mm_div_ps(ssim_val, _mm_mul_ps(c, d));                       \
      ssim_sum = _mm_add_ps(ssim_sum, ssim_val);                               \
      ssim_val = _mm_sub_ps(one, ssim_val);                                    \
      if (essim_mink_value == 4) {                                             \
        ssim_val = _mm_mul_ps(ssim_val, ssim_val);                             \
        ssim_val = _mm_mul_ps(ssim_val, ssim_val);                             \
      } else {                                                                 \
        ssim_val = _mm_mul_ps(_mm_mul_ps(ssim_val, ssim_val), ssim_val);       \
      }                                                                        \
      ssim_mink_sum = _mm_add_ps(ssim_mink_sum, ssim_val);                     \
    }                                                                          \
  }

void sum_windows_8x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 8 };

  const __m128i c_sum_shuffle_pattern =
      _mm_loadu_si128((const __m128i *)sum_8x8_shuffle);
  const __m128 invWindowSize_sqd =
      _mm_set1_ps(1.0f / (float)(windowSize * windowSize));
  const __m128 invWindowSize_qd =
      _mm_mul_ps(invWindowSize_sqd, invWindowSize_sqd);
  const float fC1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float fC2 = get_ssim_float_constant(2, bitDepthMinus8);
  const __m128 C1 = _mm_set1_ps(fC1);
  const __m128 C2 = _mm_set1_ps(fC2);
  const __m128 halfC2 = _mm_set1_ps(fC2 / 2.0f);

  __m128 ssim_mink_sum = _mm_setzero_ps();
  __m128 ssim_sum = _mm_setzero_ps();
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;

    __m128 ref_sum, cmp_sum;
    __m128 ref_sigma_sqd;
    __m128 cmp_sigma_sqd;
    __m128 sigma_both;

    ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_SSSE3(ref_sum, cmp_sum, 0);
    ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_SSSE3(ref_sigma_sqd, 1);
    ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_SSSE3(cmp_sigma_sqd, 2);
    ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_SSSE3(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_4_FLOAT_SSIM_SSE();
  }

  ssim_sum = _mm_hadd_ps(ssim_sum, ssim_mink_sum);
  ssim_sum = _mm_hadd_ps(ssim_sum, ssim_sum);

  res->ssim_sum_f += _mm_cvtss_f32(ssim_sum);
  ssim_sum = _mm_shuffle_ps(ssim_sum, ssim_sum, 0x39);
  res->ssim_mink_sum_f += _mm_cvtss_f32(ssim_sum);
  res->numWindows += i;

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_8x4_float_8u_c(res, &buf, numWindows - i, windowSize,
                               windowStride, bitDepthMinus8, NULL, 0, 0,
                               essim_mink_value);
  }

} /* void sum_windows_8x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS) */

#define ASM_LOAD_12X12_WINDOW_8_FLOAT_VALUES_SSSE3(value0, value1, idx)        \
  {                                                                            \
    __m128i _r0, _r1, _r2, _r3;                                                \
    _r0 = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));           \
    _r3 = _mm_loadl_epi64((const __m128i *)(pSrc + (idx)*srcStride + 16));     \
    _r0 = _mm_add_epi16(                                                       \
        _r0, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride)));   \
    _r3 = _mm_add_epi16(                                                       \
        _r3,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext + (idx)*srcStride + 16)));  \
    _r0 = _mm_add_epi16(                                                       \
        _r0, _mm_load_si128((const __m128i *)(pSrcNext2 + (idx)*srcStride)));  \
    _r3 = _mm_add_epi16(                                                       \
        _r3,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext2 + (idx)*srcStride + 16))); \
    _r1 = _mm_srli_si128(_r0, 4);                                              \
    _r2 = _mm_alignr_epi8(_r3, _r0, 8);                                        \
    _r3 = _mm_alignr_epi8(_r3, _r0, 12);                                       \
    _r0 = _mm_shuffle_epi8(_r0, c_sum_shuffle_pattern);                        \
    _r1 = _mm_shuffle_epi8(_r1, c_sum_shuffle_pattern);                        \
    _r2 = _mm_shuffle_epi8(_r2, c_sum_shuffle_pattern);                        \
    _r3 = _mm_shuffle_epi8(_r3, c_sum_shuffle_pattern);                        \
    _r0 = _mm_hadd_epi16(_r0, _r1);                                            \
    _r2 = _mm_hadd_epi16(_r2, _r3);                                            \
    _r0 = _mm_hadd_epi16(_r0, _r2);                                            \
    _r1 = _mm_srli_epi32(_r0, 16);                                             \
    _r0 = _mm_srli_epi32(_mm_slli_epi32(_r0, 16), 16);                         \
    value0 = _mm_cvtepi32_ps(_r0);                                             \
    value1 = _mm_cvtepi32_ps(_r1);                                             \
  }

#define ASM_LOAD_12X12_WINDOW_4_FLOAT_VALUES_SSSE3(value, idx)                 \
  {                                                                            \
    __m128i _r0, _r1, _r2, _r3;                                                \
    _r0 = _mm_load_si128((const __m128i *)(pSrc + (idx)*srcStride));           \
    _r3 = _mm_loadl_epi64((const __m128i *)(pSrc + (idx)*srcStride + 16));     \
    _r0 = _mm_add_epi32(                                                       \
        _r0, _mm_load_si128((const __m128i *)(pSrcNext + (idx)*srcStride)));   \
    _r3 = _mm_add_epi32(                                                       \
        _r3,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext + (idx)*srcStride + 16)));  \
    _r0 = _mm_add_epi32(                                                       \
        _r0, _mm_load_si128((const __m128i *)(pSrcNext2 + (idx)*srcStride)));  \
    _r3 = _mm_add_epi32(                                                       \
        _r3,                                                                   \
        _mm_loadl_epi64((const __m128i *)(pSrcNext2 + (idx)*srcStride + 16))); \
    _r1 = _mm_srli_si128(_r0, 4);                                              \
    _r2 = _mm_alignr_epi8(_r3, _r0, 8);                                        \
    _r2 = _mm_slli_si128(_r2, 4);                                              \
    _r3 = _mm_alignr_epi8(_r3, _r0, 12);                                       \
    _r0 = _mm_slli_si128(_r0, 4);                                              \
    _r0 = _mm_hadd_epi32(_r0, _r1);                                            \
    _r2 = _mm_hadd_epi32(_r2, _r3);                                            \
    _r0 = _mm_hadd_epi32(_r0, _r2);                                            \
    value = _mm_mul_ps(_mm_cvtepi32_ps(_r0), invWindowSize_sqd);               \
  }

void sum_windows_12x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 12 };

  const __m128i c_sum_shuffle_pattern =
      _mm_loadu_si128((const __m128i *)sum_12x12_shuffle);
  const __m128 invWindowSize_sqd =
      _mm_set1_ps(1.0f / (float)(windowSize * windowSize));
  const __m128 invWindowSize_qd =
      _mm_mul_ps(invWindowSize_sqd, invWindowSize_sqd);
  const float fC1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float fC2 = get_ssim_float_constant(2, bitDepthMinus8);
  const __m128 C1 = _mm_set1_ps(fC1);
  const __m128 C2 = _mm_set1_ps(fC2);
  const __m128 halfC2 = _mm_set1_ps(fC2 / 2.0f);

  __m128 ssim_mink_sum = _mm_setzero_ps();
  __m128 ssim_sum = _mm_setzero_ps();
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;
    const uint8_t *pSrcNext2 = pSrc + 8 * srcStride;

    __m128 ref_sum, cmp_sum;
    __m128 ref_sigma_sqd;
    __m128 cmp_sigma_sqd;
    __m128 sigma_both;

    ASM_LOAD_12X12_WINDOW_8_FLOAT_VALUES_SSSE3(ref_sum, cmp_sum, 0);
    ASM_LOAD_12X12_WINDOW_4_FLOAT_VALUES_SSSE3(ref_sigma_sqd, 1);
    ASM_LOAD_12X12_WINDOW_4_FLOAT_VALUES_SSSE3(cmp_sigma_sqd, 2);
    ASM_LOAD_12X12_WINDOW_4_FLOAT_VALUES_SSSE3(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_4_FLOAT_SSIM_SSE();
  }

  ssim_sum = _mm_hadd_ps(ssim_sum, ssim_mink_sum);
  ssim_sum = _mm_hadd_ps(ssim_sum, ssim_sum);

  res->ssim_sum_f += _mm_cvtss_f32(ssim_sum);
  ssim_sum = _mm_shuffle_ps(ssim_sum, ssim_sum, 0x39);
  res->ssim_mink_sum_f +=
      _mm_cvtss_f32(ssim_sum); // TODO replace with (1 - ssim) ** 4
  res->numWindows += i;

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_12x4_float_8u_c(res, &buf, numWindows - i, windowSize,
                                windowStride, bitDepthMinus8, NULL, 0, 0,
                                essim_mink_value);
  }

} /* void sum_windows_12x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS) */

#if NEW_SIMD_FUNC

#define ASM_LOAD_16X16_WINDOW_8_FLOAT_VALUES_SSSE3(value0, value1, idx)        \
  {                                                                            \
    __m128i _r0, _r1;                                                          \
    _r0 = _mm_loadu_si128((const __m128i *)(pSrc + (idx)*srcStride));          \
    _r1 = _mm_loadu_si32(pSrc + (idx)*srcStride + 16);                         \
    _r0 = _mm_add_epi16(                                                       \
        _r0, _mm_loadu_si128((const __m128i *)(pSrc + (idx + 4)*srcStride)));  \
    _r1 = _mm_add_epi16(_r1, _mm_loadu_si32(pSrc + (idx + 4)*srcStride + 16)); \
    _r1 = _mm_alignr_epi8(_r1, _r0, 8);                                        \
    _r0 = _mm_shuffle_epi8(_r0, c_sum_shuffle_pattern);                        \
    _r1 = _mm_shuffle_epi8(_r1, c_sum_shuffle_pattern);                        \
    _r0 = _mm_hadd_epi16(_r0, _r1);                                            \
    _r1 = _mm_srli_epi32(_r0, 16);                                             \
    _r0 = _mm_srli_epi32(_mm_slli_epi32(_r0, 16), 16);                         \
    value0 = _mm_add_ps(value0, _mm_cvtepi32_ps(_r0));                         \
    value1 = _mm_add_ps(value1, _mm_cvtepi32_ps(_r1));                         \
  }

#define ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(value, idx)                 \
  {                                                                            \
    __m128i _r0, _r1;                                                          \
    _r0 = _mm_loadu_si128((const __m128i *)(pSrc + (idx)*srcStride));          \
    _r1 = _mm_loadu_si32(pSrc + (idx)*srcStride + 16);                         \
    _r0 = _mm_add_epi32(                                                       \
        _r0, _mm_loadu_si128((const __m128i *)(pSrc + (idx + 4)*srcStride)));  \
    _r1 = _mm_add_epi32(_r1, _mm_loadu_si32(pSrc + (idx + 4)*srcStride + 16)); \
    _r1 = _mm_alignr_epi8(_r1, _r0, 8);                                        \
    _r0 = _mm_shuffle_epi32(_r0, 0x94);                                        \
    _r1 = _mm_shuffle_epi32(_r1, 0x94);                                        \
    _r0 = _mm_hadd_epi32(_r0, _r1);                                            \
    value = _mm_add_ps(value, _mm_mul_ps(                                      \
                              _mm_cvtepi32_ps(_r0), invWindowSize_sqd));       \
  }

void sum_windows_16x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 16 };
  const __m128i c_sum_shuffle_pattern =
      _mm_loadu_si128((const __m128i *)sum_8x8_shuffle);
  const __m128 invWindowSize_sqd =
      _mm_set1_ps(1.0f / (float)(windowSize * windowSize));
  const __m128 invWindowSize_qd =
      _mm_mul_ps(invWindowSize_sqd, invWindowSize_sqd);
  const float fC1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float fC2 = get_ssim_float_constant(2, bitDepthMinus8);
  const __m128 C1 = _mm_set1_ps(fC1);
  const __m128 C2 = _mm_set1_ps(fC2);
  const __m128 halfC2 = _mm_set1_ps(fC2 / 2.0f);

  __m128 ssim_mink_sum = _mm_setzero_ps();
  __m128 ssim_sum = _mm_setzero_ps();
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    __m128 ref_sum = _mm_setzero_ps();
    __m128 cmp_sum = _mm_setzero_ps();
    __m128 ref_sigma_sqd = _mm_setzero_ps();
    __m128 cmp_sigma_sqd = _mm_setzero_ps();
    __m128 sigma_both = _mm_setzero_ps();

    for (uint32_t x = 0; x < 2; x++) {

      ASM_LOAD_16X16_WINDOW_8_FLOAT_VALUES_SSSE3(ref_sum, cmp_sum, 0);
      ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(ref_sigma_sqd, 1);
      ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(cmp_sigma_sqd, 2);
      ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(sigma_both, 3);

      ASM_LOAD_16X16_WINDOW_8_FLOAT_VALUES_SSSE3(ref_sum, cmp_sum, 8);
      ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(ref_sigma_sqd, 9);
      ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(cmp_sigma_sqd, 10);
      ASM_LOAD_16X16_WINDOW_4_FLOAT_VALUES_SSSE3(sigma_both, 11);

      pSrc += sizeof(uint32_t) * 2;
    }
    ASM_CALC_4_FLOAT_SSIM_SSE();
  }

  ssim_sum = _mm_hadd_ps(ssim_sum, ssim_mink_sum);
  ssim_sum = _mm_hadd_ps(ssim_sum, ssim_sum);

  res->ssim_sum_f += _mm_cvtss_f32(ssim_sum);
  ssim_sum = _mm_shuffle_ps(ssim_sum, ssim_sum, 0x39);
  res->ssim_mink_sum_f += _mm_cvtss_f32(ssim_sum);
  res->numWindows += i;

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_16x4_float_8u_c(res, &buf, numWindows - i, windowSize,
                               windowStride, bitDepthMinus8, NULL, 0, 0,
                               essim_mink_value);
  }

} /* void sum_windows_16x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS) */

#endif

#endif /* defined(_X86) || defined(_X64) */
