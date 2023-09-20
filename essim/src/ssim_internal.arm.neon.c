/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/inc/internal.h>

#if defined(_ARM) || defined(_ARM64)

#include <arm_neon.h>

void load_4x4_windows_8u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint8_t) * WIN_SIZE;

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    uint16x8_t sum = vdupq_n_u16(0);
    uint32x4_t ref_sigma_sqd = vdupq_n_u32(0);
    uint32x4_t cmp_sigma_sqd = vdupq_n_u32(0);
    uint32x4_t sigma_both = vdupq_n_u32(0);

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
#if ARM_BUILD_FIXES
      uint8x16_t r0, r1;
      uint16x8_t r2, r3, r4, r5, r6, r7;
      uint32x4_t r8, r9;
#elif !ARM_BUILD_FIXES
      uint8x16_t r0, r1, r2, r3, r4, r5;
#endif

      r0 = vld1q_u8(src.ref);
      r1 = vld1q_u8(src.cmp);

      /* sum ref_sum & cmp_sum */
      r4 = vpaddlq_u8(r0);
      r5 = vpaddlq_u8(r1);
#if ARM_BUILD_FIXES
      r8 = vpaddlq_u16(r4);
      r9 = vpaddlq_u16(r5);
      r9 = vshlq_n_u32(r9, 16);
      sum = vaddq_u16(sum, vreinterpretq_u16_u32(r8));
      sum = vaddq_u16(sum, vreinterpretq_u16_u32(r9));
#elif !ARM_BUILD_FIXES
      r4 = vpaddlq_u16(r4);
      r5 = vpaddlq_u16(r5);
      r5 = vshlq_n_u32(r5, 16);
      sum = vaddq_u16(sum, r4);
      sum = vaddq_u16(sum, r5);
#endif

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      uint8x16x2_t t0 = vuzpq_u8(r0, r1);

      r0 = t0.val[0];
      r1 = t0.val[1];

      uint8x8_t s0, s1, s2, s3;

      s0 = vget_low_u8(r0);
      s1 = vget_high_u8(r0);
      s2 = vget_low_u8(r1);
      s3 = vget_high_u8(r1);

#if ARM_BUILD_FIXES
      r2 = vmull_u8(s0, s0);
      r3 = vmull_u8(s1, s0);
      r4 = vmull_u8(s1, s1);
      r5 = vmull_u8(s2, s2);
      r6 = vmull_u8(s3, s2);
      r7 = vmull_u8(s3, s3);
      ref_sigma_sqd = vpadalq_u16(ref_sigma_sqd, r2);
      sigma_both = vpadalq_u16(sigma_both, r3);
      cmp_sigma_sqd = vpadalq_u16(cmp_sigma_sqd, r4);
      ref_sigma_sqd = vpadalq_u16(ref_sigma_sqd, r5);
      sigma_both = vpadalq_u16(sigma_both, r6);
      cmp_sigma_sqd = vpadalq_u16(cmp_sigma_sqd, r7);
#elif !ARM_BUILD_FIXES
      r0 = vmull_u8(s0, s0);
      r4 = vmull_u8(s1, s0);
      r1 = vmull_u8(s1, s1);
      r2 = vmull_u8(s2, s2);
      r5 = vmull_u8(s3, s2);
      r3 = vmull_u8(s3, s3);
      ref_sigma_sqd = vpadalq_u16(ref_sigma_sqd, r0);
      sigma_both = vpadalq_u16(sigma_both, r4);
      cmp_sigma_sqd = vpadalq_u16(cmp_sigma_sqd, r1);
      ref_sigma_sqd = vpadalq_u16(ref_sigma_sqd, r2);
      sigma_both = vpadalq_u16(sigma_both, r5);
      cmp_sigma_sqd = vpadalq_u16(cmp_sigma_sqd, r3);
#endif

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

#if ARM_BUILD_FIXES
    vst1q_u8(pDst + 0 * dstStride, vreinterpretq_u8_u16(sum));
    vst1q_u8(pDst + 1 * dstStride, vreinterpretq_u8_u32(ref_sigma_sqd));
    vst1q_u8(pDst + 2 * dstStride, vreinterpretq_u8_u32(cmp_sigma_sqd));
    vst1q_u8(pDst + 3 * dstStride, vreinterpretq_u8_u32(sigma_both));
#elif !ARM_BUILD_FIXES
    vst1q_u8(pDst + 0 * dstStride, sum);
    vst1q_u8(pDst + 1 * dstStride, ref_sigma_sqd);
    vst1q_u8(pDst + 2 * dstStride, cmp_sigma_sqd);
    vst1q_u8(pDst + 3 * dstStride, sigma_both);
#endif
    pDst = AdvancePointer(pDst, WIN_CHUNK * sizeof(uint32_t));

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }

  for (; i < num4x4Windows; i += 1) {
    uint16x4_t sum = vdup_n_u16(0);
    uint32x2_t ref_sigma_sqd = vdup_n_u32(0);
    uint32x2_t cmp_sigma_sqd = vdup_n_u32(0);
    uint32x2_t sigma_both = vdup_n_u32(0);

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
#if ARM_BUILD_FIXES
      uint32x2_t r0, r1, r2, r3;
      uint16x4_t r4, r5;
      uint8x8_t r8, r9;
#elif !ARM_BUILD_FIXES
      uint8x8_t r0, r1, r2, r3, r4, r5;
#endif

      r0 = vld1_dup_u32(src.ref);
      r1 = vld1_dup_u32(src.cmp);

      /* sum ref_sum & cmp_sum */
#if ARM_BUILD_FIXES
      r4 = vpaddl_u8(vreinterpret_u8_u32(r0));
      r5 = vpaddl_u8(vreinterpret_u8_u32(r1));
      r2 = vpaddl_u16(r4);
      r3 = vpaddl_u16(r5);
      r3 = vshl_n_u32(r3, 16);
      sum = vadd_u16(sum, vreinterpret_u16_u32(r2));
      sum = vadd_u16(sum, vreinterpret_u16_u32(r3));

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      uint8x8x2_t t0 = vuzp_u8(
                  vreinterpret_u8_u32(r0), vreinterpret_u8_u32(r1));

      r8 = t0.val[0];
      r9 = t0.val[1];
#elif !ARM_BUILD_FIXES
      r4 = vpaddl_u8(r0);
      r5 = vpaddl_u8(r1);
      r4 = vpaddl_u16(r4);
      r5 = vpaddl_u16(r5);
      r5 = vshl_n_u32(r5, 16);
      sum = vadd_u16(sum, r4);
      sum = vadd_u16(sum, r5);

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      uint8x8x2_t t0 = vuzp_u8(r0, r1);

      r0 = t0.val[0];
      r1 = t0.val[1];
#endif

      uint8x8_t s0, s1, s2, s3;

#if ARM_BUILD_FIXES
      s0 = r8;
      s1 = vreinterpret_u8_u32(vrev64_u32(vreinterpret_u32_u8(r8)));
      s2 = r9;
      s3 = vreinterpret_u8_u32(vrev64_u32(vreinterpret_u32_u8(r9)));

      r4 = vget_low_u16(vmull_u8(s0, s0));
      r5 = vget_low_u16(vmull_u8(s1, s0));
      ref_sigma_sqd = vpadal_u16(ref_sigma_sqd, r4);
      sigma_both = vpadal_u16(sigma_both, r5);
      r4 = vget_low_u16(vmull_u8(s1, s1));
      r5 = vget_low_u16(vmull_u8(s2, s2));
      cmp_sigma_sqd = vpadal_u16(cmp_sigma_sqd, r4);
      ref_sigma_sqd = vpadal_u16(ref_sigma_sqd, r5);
      r4 = vget_low_u16(vmull_u8(s3, s2));
      r5 = vget_low_u16(vmull_u8(s3, s3));
      sigma_both = vpadal_u16(sigma_both, r4);
      cmp_sigma_sqd = vpadal_u16(cmp_sigma_sqd, r5);
#elif !ARM_BUILD_FIXES
      s0 = r0;
      s1 = vrev64_u32(r0);
      s2 = r1;
      s3 = vrev64_u32(r1);

      r0 = vget_low_u16(vmull_u8(s0, s0));
      r4 = vget_low_u16(vmull_u8(s1, s0));
      r1 = vget_low_u16(vmull_u8(s1, s1));
      r2 = vget_low_u16(vmull_u8(s2, s2));
      r5 = vget_low_u16(vmull_u8(s3, s2));
      r3 = vget_low_u16(vmull_u8(s3, s3));
      ref_sigma_sqd = vpadal_u16(ref_sigma_sqd, r0);
      sigma_both = vpadal_u16(sigma_both, r4);
      cmp_sigma_sqd = vpadal_u16(cmp_sigma_sqd, r1);
      ref_sigma_sqd = vpadal_u16(ref_sigma_sqd, r2);
      sigma_both = vpadal_u16(sigma_both, r5);
      cmp_sigma_sqd = vpadal_u16(cmp_sigma_sqd, r3);
#endif

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

#if ARM_BUILD_FIXES
    vst1_lane_u32((uint32_t *)(pDst + 0 * dstStride),
                              vreinterpret_u32_u16(sum), 0);
    vst1_lane_u32((uint32_t *)(pDst + 1 * dstStride), ref_sigma_sqd, 0);
    vst1_lane_u32((uint32_t *)(pDst + 2 * dstStride), cmp_sigma_sqd, 0);
    vst1_lane_u32((uint32_t *)(pDst + 3 * dstStride), sigma_both, 0);
#elif !ARM_BUILD_FIXES
    vst1_lane_u32(pDst + 0 * dstStride, sum, 0);
    vst1_lane_u32(pDst + 1 * dstStride, ref_sigma_sqd, 0);
    vst1_lane_u32(pDst + 2 * dstStride, cmp_sigma_sqd, 0);
    vst1_lane_u32(pDst + 3 * dstStride, sigma_both, 0);
#endif
    pDst = AdvancePointer(pDst, sizeof(uint32_t));

    /* advance source pointers */
    src.ref = AdvancePointer(src.ref, srcStep - WIN_SIZE * src.refStride);
    src.cmp = AdvancePointer(src.cmp, srcStep - WIN_SIZE * src.cmpStride);
  }

} /* void load_4x4_windows_8u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

#if NEW_SIMD_FUNC
void load_4x4_windows_10u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint16_t) * WIN_SIZE;

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    uint16x8_t sum = vdupq_n_u16(0);
    uint32x4_t ref_sigma_sqd = vdupq_n_u32(0);
    uint32x4_t cmp_sigma_sqd = vdupq_n_u32(0);
    uint32x4_t sigma_both = vdupq_n_u32(0);

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      uint16x8_t r0, r1, r2, r3;
      uint32x4_t r4, r5;
      uint8_t *refNext = AdvancePointer(src.ref, 16);
      uint8_t *cmpNext = AdvancePointer(src.cmp, 16);

      r0 = vld1q_u16(src.ref);
      r1 = vld1q_u16(src.cmp);
      r2 = vld1q_u16((uint16_t *)refNext);
      r3 = vld1q_u16((uint16_t *)cmpNext);

      /* sum ref_sum & cmp_sum */
      r4 = vpaddlq_u16(vpaddq_u16(r0, r2));
      r5 = vpaddlq_u16(vpaddq_u16(r1, r3));
      r5 = vshlq_n_u32(r5, 16);
      sum = vaddq_u16(sum, vreinterpretq_u16_u32(r4));
      sum = vaddq_u16(sum, vreinterpretq_u16_u32(r5));

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      uint16x4_t s0, s1, s2, s3, s4, s5, s6, s7;

      s0 = vget_low_u16(r0);
      s1 = vget_high_u16(r0);
      s2 = vget_low_u16(r1);
      s3 = vget_high_u16(r1);
      s4 = vget_low_u16(r2);
      s5 = vget_high_u16(r2);
      s6 = vget_low_u16(r3);
      s7 = vget_high_u16(r3);

      r4 = vpaddq_u32(vmull_u16(s0, s0),vmull_u16(s1, s1));
      r5 = vpaddq_u32(vmull_u16(s4, s4),vmull_u16(s5, s5));
      r4 = vpaddq_u32(r4, r5);
      ref_sigma_sqd = vaddq_u32(ref_sigma_sqd, r4);

      r4 = vpaddq_u32(vmull_u16(s2, s2),vmull_u16(s3, s3));
      r5 = vpaddq_u32(vmull_u16(s6, s6),vmull_u16(s7, s7));
      r4 = vpaddq_u32(r4, r5);
      cmp_sigma_sqd = vaddq_u32(cmp_sigma_sqd, r4);

      r4 = vpaddq_u32(vmull_u16(s0, s2),vmull_u16(s1, s3));
      r5 = vpaddq_u32(vmull_u16(s4, s6),vmull_u16(s5, s7));
      r4 = vpaddq_u32(r4, r5);
      sigma_both = vaddq_u32(sigma_both, r4);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    vst1q_u8(pDst, vreinterpretq_u8_u16(sum));
    vst1q_u8(pDst + dstStride, vreinterpretq_u8_u32(ref_sigma_sqd));
    vst1q_u8(pDst + 2 * dstStride, vreinterpretq_u8_u32(cmp_sigma_sqd));
    vst1q_u8(pDst + 3 * dstStride, vreinterpretq_u8_u32(sigma_both));
    pDst = AdvancePointer(pDst, WIN_CHUNK * sizeof(uint32_t));

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }

  if (i < num4x4Windows) {
    SSIM_4X4_WINDOW_BUFFER buf = {pDst, dstStride};

    load_4x4_windows_10u_c(&buf, num4x4Windows - i, &src);
  }

} /* void load_4x4_windows_10u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS) */
#endif

void load_4x4_windows_16u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 2, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint16_t) * WIN_SIZE;

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    uint32x4_t sum = vdupq_n_u32(0);
    uint64x2_t ref_sigma_sqd = vdupq_n_u64(0);
    uint64x2_t cmp_sigma_sqd = vdupq_n_u64(0);
    uint64x2_t sigma_both = vdupq_n_u64(0);

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
#if ARM_BUILD_FIXES
      uint16x8_t r0, r1;
      uint32x4_t r2, r3, r4, r5, r6, r7;
      uint64x2_t r8, r9;
#elif !ARM_BUILD_FIXES
      uint16x8_t r0, r1, r2, r3, r4, r5;
#endif

      r0 = vld1q_u16(src.ref);
      r1 = vld1q_u16(src.cmp);

      /* sum ref_sum & cmp_sum */
      r4 = vpaddlq_u16(r0);
      r5 = vpaddlq_u16(r1);
#if ARM_BUILD_FIXES
      r8 = vpaddlq_u32(r4);
      r9 = vpaddlq_u32(r5);
      r9 = vshlq_n_u64(r9, 32);
      sum = vaddq_u32(sum, vreinterpretq_u32_u64(r8));
      sum = vaddq_u32(sum, vreinterpretq_u32_u64(r9));
#elif !ARM_BUILD_FIXES
      r4 = vpaddlq_u32(r4);
      r5 = vpaddlq_u32(r5);
      r5 = vshlq_n_u64(r5, 32);
      sum = vaddq_u32(sum, r4);
      sum = vaddq_u32(sum, r5);
#endif

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      uint16x8x2_t t0 = vuzpq_u16(r0, r1);

      r0 = t0.val[0];
      r1 = t0.val[1];

      uint16x4_t s0, s1, s2, s3;

      s0 = vget_low_u16(r0);
      s1 = vget_high_u16(r0);
      s2 = vget_low_u16(r1);
      s3 = vget_high_u16(r1);

#if ARM_BUILD_FIXES
      r2 = vmull_u16(s0, s0);
      r3 = vmull_u16(s1, s0);
      r4 = vmull_u16(s1, s1);
      r5 = vmull_u16(s2, s2);
      r6 = vmull_u16(s3, s2);
      r7 = vmull_u16(s3, s3);
      ref_sigma_sqd = vpadalq_u32(ref_sigma_sqd, r2);
      sigma_both = vpadalq_u32(sigma_both, r3);
      cmp_sigma_sqd = vpadalq_u32(cmp_sigma_sqd, r4);
      ref_sigma_sqd = vpadalq_u32(ref_sigma_sqd, r5);
      sigma_both = vpadalq_u32(sigma_both, r6);
      cmp_sigma_sqd = vpadalq_u32(cmp_sigma_sqd, r7);
#elif !ARM_BUILD_FIXES
      r0 = vmull_u16(s0, s0);
      r4 = vmull_u16(s1, s0);
      r1 = vmull_u16(s1, s1);
      r2 = vmull_u16(s2, s2);
      r5 = vmull_u16(s3, s2);
      r3 = vmull_u16(s3, s3);
      ref_sigma_sqd = vpadalq_u32(ref_sigma_sqd, r0);
      sigma_both = vpadalq_u32(sigma_both, r4);
      cmp_sigma_sqd = vpadalq_u32(cmp_sigma_sqd, r1);
      ref_sigma_sqd = vpadalq_u32(ref_sigma_sqd, r2);
      sigma_both = vpadalq_u32(sigma_both, r5);
      cmp_sigma_sqd = vpadalq_u32(cmp_sigma_sqd, r3);
#endif

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

#if ARM_BUILD_FIXES
    vst1q_u16((uint16_t *)(pDst + 0 * dstStride),
                          vreinterpretq_u16_u32(sum));
    vst1q_u16((uint16_t *)(pDst + 1 * dstStride),
                          vreinterpretq_u16_u64(ref_sigma_sqd));
    vst1q_u16((uint16_t *)(pDst + 2 * dstStride),
                          vreinterpretq_u16_u64(cmp_sigma_sqd));
    vst1q_u16((uint16_t *)(pDst + 3 * dstStride),
                          vreinterpretq_u16_u64(sigma_both));
#elif !ARM_BUILD_FIXES
    vst1q_u16((uint16_t *)(pDst + 0 * dstStride), sum);
    vst1q_u16((uint16_t *)(pDst + 1 * dstStride), ref_sigma_sqd);
    vst1q_u16((uint16_t *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    vst1q_u16((uint16_t *)(pDst + 3 * dstStride), sigma_both);
#endif
    pDst = AdvancePointer(pDst, WIN_CHUNK * sizeof(uint64_t));

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }

  for (; i < num4x4Windows; i += 1) {
    uint32x2_t sum = vdup_n_u32(0);
    uint64x1_t ref_sigma_sqd = vdup_n_u64(0);
    uint64x1_t cmp_sigma_sqd = vdup_n_u64(0);
    uint64x1_t sigma_both = vdup_n_u64(0);

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
#if ARM_BUILD_FIXES
      uint64x1_t r2, r3;
      uint32x2_t r4, r5;
      uint16x4_t r0, r1;
#elif !ARM_BUILD_FIXES
      uint16x4_t r0, r1, r2, r3, r4, r5;
#endif

      r0 = vld1_u16(src.ref);
      r1 = vld1_u16(src.cmp);

      /* sum ref_sum & cmp_sum */
#if ARM_BUILD_FIXES
      r4 = vpaddl_u16(r0);
      r5 = vpaddl_u16(r1);
      r2 = vpaddl_u32(r4);
      r3 = vpaddl_u32(r5);
      r3 = vshl_n_u64(r3, 32);
      sum = vadd_u32(sum, vreinterpret_u32_u64(r2));
      sum = vadd_u32(sum, vreinterpret_u32_u64(r3));
#elif !ARM_BUILD_FIXES
      r4 = vpaddl_u16(r0);
      r5 = vpaddl_u16(r1);
      r4 = vpaddl_u32(r4);
      r5 = vpaddl_u32(r5);
      r5 = vshl_n_u64(r5, 32);
      sum = vadd_u32(sum, r4);
      sum = vadd_u32(sum, r5);
#endif

      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      uint16x4x2_t t0 = vuzp_u16(r0, r1);

      r0 = t0.val[0];
      r1 = t0.val[1];

      uint16x4_t s0, s1, s2, s3;

#if ARM_BUILD_FIXES
      s0 = r0;
      s1 = vreinterpret_u16_u32(vrev64_u32(vreinterpret_u32_u16(r0)));
      s2 = r1;
      s3 = vreinterpret_u16_u32(vrev64_u32(vreinterpret_u32_u16(r1)));

      r4 = vget_low_u32(vmull_u16(s0, s0));
      r5 = vget_low_u32(vmull_u16(s1, s0));
      ref_sigma_sqd = vpadal_u32(ref_sigma_sqd, r4);
      sigma_both = vpadal_u32(sigma_both, r5);
      r4 = vget_low_u32(vmull_u16(s1, s1));
      r5 = vget_low_u32(vmull_u16(s2, s2));
      cmp_sigma_sqd = vpadal_u32(cmp_sigma_sqd, r4);
      ref_sigma_sqd = vpadal_u32(ref_sigma_sqd, r5);
      r4 = vget_low_u32(vmull_u16(s3, s2));
      r5 = vget_low_u32(vmull_u16(s3, s3));
      sigma_both = vpadal_u32(sigma_both, r4);
      cmp_sigma_sqd = vpadal_u32(cmp_sigma_sqd, r5);
#elif !ARM_BUILD_FIXES
      s0 = r0;
      s1 = vrev64_u32(r0);
      s2 = r1;
      s3 = vrev64_u32(r1);

      r0 = vget_low_u32(vmull_u16(s0, s0));
      r4 = vget_low_u32(vmull_u16(s1, s0));
      r1 = vget_low_u32(vmull_u16(s1, s1));
      r2 = vget_low_u32(vmull_u16(s2, s2));
      r5 = vget_low_u32(vmull_u16(s3, s2));
      r3 = vget_low_u32(vmull_u16(s3, s3));
      ref_sigma_sqd = vpadal_u32(ref_sigma_sqd, r0);
      sigma_both = vpadal_u32(sigma_both, r4);
      cmp_sigma_sqd = vpadal_u32(cmp_sigma_sqd, r1);
      ref_sigma_sqd = vpadal_u32(ref_sigma_sqd, r2);
      sigma_both = vpadal_u32(sigma_both, r5);
      cmp_sigma_sqd = vpadal_u32(cmp_sigma_sqd, r3);
#endif

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }
#if ARM_BUILD_FIXES
    vst1_u64((uint64_t *)(pDst + 0 * dstStride), vreinterpret_u64_u32(sum));
#elif !ARM_BUILD_FIXES
    vst1_u64((uint64_t *)(pDst + 0 * dstStride), sum);
#endif
    vst1_u64((uint64_t *)(pDst + 1 * dstStride), ref_sigma_sqd);
    vst1_u64((uint64_t *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    vst1_u64((uint64_t *)(pDst + 3 * dstStride), sigma_both);
    pDst = AdvancePointer(pDst, sizeof(uint64_t));

    /* advance source pointers */
    src.ref = AdvancePointer(src.ref, srcStep - WIN_SIZE * src.refStride);
    src.cmp = AdvancePointer(src.cmp, srcStep - WIN_SIZE * src.cmpStride);
  }

} /* void load_4x4_windows_16u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

#if defined(_ARM)

/* emulate missed instructions under arm-v7a */

#define vmull_high_s32(a, b) vmull_s32(vget_high_s32(a), vget_high_s32(b))

#define vpaddq_u16(a, b)                                                       \
  vcombine_u16(vpadd_u16(vget_low_u16(a), vget_high_u16(a)),                   \
               vpadd_u16(vget_low_u16(b), vget_high_u16(b)))

#define vpaddq_u32(a, b)                                                       \
  vcombine_u32(vpadd_u32(vget_low_u32(a), vget_high_u32(a)),                   \
               vpadd_u32(vget_low_u32(b), vget_high_u32(b)))

#define vpaddq_f32(a, b)                                                       \
  vcombine_f32(vpadd_f32(vget_low_f32(a), vget_high_f32(a)),                   \
               vpadd_f32(vget_low_f32(b), vget_high_f32(b)))

#define vdivq_f32(a, b) vmulq_f32(a, vrecpeq_f32(b))

#endif /* defined(_ARM) */

#if !NEW_SIMD_FUNC

#define ASM_LOAD_8X8_WINDOW_8_WORD_VALUES_NEON(value, idx)                     \
  {                                                                            \
    uint16x8_t _r0, _r1;                                                       \
    _r0 = vld1q_u16((const uint16_t *)(pSrc + (idx)*srcStride));               \
    _r1 = vld1q_u16((const uint16_t *)(pSrc + (idx)*srcStride + 4));           \
    _r0 = vaddq_u16(                                                           \
        _r0, vld1q_u16((const uint16_t *)(pSrcNext + (idx)*srcStride)));       \
    _r1 = vaddq_u16(                                                           \
        _r1, vld1q_u16((const uint16_t *)(pSrcNext + (idx)*srcStride + 4)));   \
    uint16x8x2_t d = vzipq_u16(_r0, _r1);                                      \
    _r0 = d.val[0];                                                            \
    _r1 = d.val[1];                                                            \
    _r0 = vpaddq_u16(_r0, _r1);                                                \
    value = vreinterpretq_u32_u16(_r0);                                        \
  }

#define ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_NEON(value, idx)                    \
  {                                                                            \
    uint32x4_t _r0, _r1;                                                       \
    _r0 = vld1q_u32((const uint32_t *)(pSrc + (idx)*srcStride));               \
    _r1 = vld1q_u32((const uint32_t *)(pSrc + (idx)*srcStride + 4));           \
    _r0 = vaddq_u32(                                                           \
        _r0, vld1q_u32((const uint32_t *)(pSrcNext + (idx)*srcStride)));       \
    _r1 = vaddq_u32(                                                           \
        _r1, vld1q_u32((const uint32_t *)(pSrcNext + (idx)*srcStride + 4)));   \
    uint32x4x2_t d = vzipq_u32(_r0, _r1);                                      \
    _r0 = d.val[0];                                                            \
    _r1 = d.val[1];                                                            \
    _r0 = vpaddq_u32(_r0, _r1);                                                \
    value = vmulq_u32(_r0, windowSize_sqd);                                    \
  }

#if ARM_BUILD_FIXES
#define ASM_CALC_4_QDWORD_SSIM_NEON(num, denom)                                \
  {                                                                            \
    uint32x4_t a, c, d;                                                        \
    int32x4_t b, sigma_both_a;                                                 \
    /* STEP 2. adjust values */                                                \
    uint32x4_t _r0 = vshrq_n_u32(vshlq_n_u32(sum, 16), 16);                    \
    uint32x4_t _r1 = vshrq_n_u32(sum, 16);                                     \
    uint32x4_t both_sum_mul = vmulq_u32(_r0, _r1);                             \
    uint32x4_t ref_sum_sqd = vmulq_u32(_r0, _r0);                              \
    uint32x4_t cmp_sum_sqd = vmulq_u32(_r1, _r1);                              \
    ref_sigma_sqd = vsubq_u32(ref_sigma_sqd, ref_sum_sqd);                     \
    cmp_sigma_sqd = vsubq_u32(cmp_sigma_sqd, cmp_sum_sqd);                     \
    sigma_both_a = vreinterpretq_s32_u32(vshrq_n_u32(sigma_both, 1));          \
    sigma_both_a = vsubq_s32(                                                  \
        sigma_both_a, vreinterpretq_s32_u32(vshrq_n_u32(both_sum_mul, 1)));    \
    /* STEP 3. process numbers, do scale */                                    \
    a = vaddq_u32(both_sum_mul, halfC1);                                       \
    b = vaddq_s32(sigma_both_a, quarterC2);                                    \
    ref_sum_sqd = vshrq_n_u32(ref_sum_sqd, 1);                                 \
    cmp_sum_sqd = vshrq_n_u32(cmp_sum_sqd, 1);                                 \
    c = vaddq_u32(vaddq_u32(ref_sum_sqd, cmp_sum_sqd), halfC1);                \
    ref_sigma_sqd = vshrq_n_u32(ref_sigma_sqd, 1);                             \
    cmp_sigma_sqd = vshrq_n_u32(cmp_sigma_sqd, 1);                             \
    d = vaddq_u32(vaddq_u32(ref_sigma_sqd, cmp_sigma_sqd), halfC2);            \
    /* process numerators */                                                   \
    {                                                                          \
      int64x2_t _r;                                                            \
      _r = vmull_s32(vreinterpret_s32_u32(vget_low_u32(a)), vget_low_s32(b));  \
      vst1q_s64(num + 0, _r);                                                  \
      _r = vmull_high_s32(vreinterpretq_s32_u32(a), b);                        \
      vst1q_s64(num + 2, _r);                                                  \
    }                                                                          \
    /* process denominators */                                                 \
    {                                                                          \
      int64x2_t _r;                                                            \
      _r = vmull_s32(vreinterpret_s32_u32(vget_low_u32(c)),                    \
                     vreinterpret_s32_u32(vget_low_u32(d)));                   \
      _r = vshrq_n_s64(_r, SSIM_LOG2_SCALE + 1);                               \
      vst1q_s64(denom + 0, _r);                                                \
      _r = vmull_high_s32(vreinterpretq_s32_u32(c), vreinterpretq_s32_u32(d)); \
      _r = vshrq_n_s64(_r, SSIM_LOG2_SCALE + 1);                               \
      vst1q_s64(denom + 2, _r);                                                \
    }                                                                          \
  }
#elif !ARM_BUILD_FIXES
#define ASM_CALC_4_QDWORD_SSIM_NEON(num, denom)                                \
  {                                                                            \
    uint32x4_t a, c, d;                                                        \
    int32x4_t b, sigma_both_a;                                                 \
    /* STEP 2. adjust values */                                                \
    uint32x4_t _r0 = vshrq_n_u32(vshlq_n_u32(sum, 16), 16);                    \
    uint32x4_t _r1 = vshrq_n_u32(sum, 16);                                     \
    uint32x4_t both_sum_mul = vmulq_u32(_r0, _r1);                             \
    uint32x4_t ref_sum_sqd = vmulq_u32(_r0, _r0);                              \
    uint32x4_t cmp_sum_sqd = vmulq_u32(_r1, _r1);                              \
    ref_sigma_sqd = vsubq_u32(ref_sigma_sqd, ref_sum_sqd);                     \
    cmp_sigma_sqd = vsubq_u32(cmp_sigma_sqd, cmp_sum_sqd);                     \
    sigma_both_a = vreinterpretq_s32_u32(vshrq_n_u32(sigma_both, 1));          \
    sigma_both_a = vsubq_s32(                                                  \
        sigma_both_a, vreinterpretq_s32_u32(vshrq_n_u32(both_sum_mul, 1)));    \
    /* STEP 3. process numbers, do scale */                                    \
    a = vaddq_u32(both_sum_mul, halfC1);                                       \
    b = vaddq_s32(sigma_both_a, quarterC2);                                    \
    ref_sum_sqd = vshrq_n_u32(ref_sum_sqd, 1);                                 \
    cmp_sum_sqd = vshrq_n_u32(cmp_sum_sqd, 1);                                 \
    c = vaddq_u32(vaddq_u32(ref_sum_sqd, cmp_sum_sqd), halfC1);                \
    ref_sigma_sqd = vshrq_n_u32(ref_sigma_sqd, 1);                             \
    cmp_sigma_sqd = vshrq_n_u32(cmp_sigma_sqd, 1);                             \
    d = vaddq_u32(vaddq_u32(ref_sigma_sqd, cmp_sigma_sqd), halfC2);            \
    /* process numerators */                                                   \
    {                                                                          \
      int64x2_t _r;                                                            \
      _r = vmull_s32(vget_low_u32(a), vget_low_u32(b));                        \
      vst1q_s64(num + 0, _r);                                                  \
      _r = vmull_high_s32(a, b);                                               \
      vst1q_s64(num + 2, _r);                                                  \
    }                                                                          \
    /* process denominators */                                                 \
    {                                                                          \
      int64x2_t _r;                                                            \
      _r = vmull_s32(vget_low_u32(c), vget_low_u32(d));                        \
      _r = vshrq_n_s64(_r, SSIM_LOG2_SCALE + 1);                               \
      vst1q_s64(denom + 0, _r);                                                \
      _r = vmull_high_s32(c, d);                                               \
      _r = vshrq_n_s64(_r, SSIM_LOG2_SCALE + 1);                               \
      vst1q_s64(denom + 2, _r);                                                \
    }                                                                          \
  }
#endif

void sum_windows_8x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 8 };

  const uint32x4_t windowSize_sqd = vdupq_n_u32(windowSize * windowSize);
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const uint32x4_t halfC1 = vdupq_n_u32(C1 / 2);
  const uint32x4_t halfC2 = vdupq_n_u32(C2 / 2);
  const int32x4_t quarterC2 = vdupq_n_s32(C2 / 4);

  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;

    uint32x4_t sum;
    uint32x4_t ref_sigma_sqd;
    uint32x4_t cmp_sigma_sqd;
    uint32x4_t sigma_both;

    ASM_LOAD_8X8_WINDOW_8_WORD_VALUES_NEON(sum, 0);
    ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_NEON(ref_sigma_sqd, 1);
    ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_NEON(cmp_sigma_sqd, 2);
    ASM_LOAD_8X8_WINDOW_4_DWORD_VALUES_NEON(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_4_QDWORD_SSIM_NEON(num, denom);

    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      const int64_t ssim_val = (num[w] + denom[w] / 2) / (denom[w] | 1);
      ssim_sum += ssim_val;
      ssim_mink_sum += ssim_val * ssim_val * ssim_val * ssim_val;
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

} /* void sum_windows_8x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS) */
#endif
#define ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_NEON(value0, value1, idx)           \
  {                                                                            \
    uint16x8_t _r0, _r1;                                                       \
    _r0 = vld1q_u16((const uint16_t *)(pSrc + (idx)*srcStride));               \
    _r1 = vld1q_u16((const uint16_t *)(pSrc + (idx)*srcStride + 4));           \
    _r0 = vaddq_u16(                                                           \
        _r0, vld1q_u16((const uint16_t *)(pSrcNext + (idx)*srcStride)));       \
    _r1 = vaddq_u16(                                                           \
        _r1, vld1q_u16((const uint16_t *)(pSrcNext + (idx)*srcStride + 4)));   \
    uint16x8x2_t d = vzipq_u16(_r0, _r1);                                      \
    _r0 = d.val[0];                                                            \
    _r1 = d.val[1];                                                            \
    _r0 = vpaddq_u16(_r0, _r1);                                                \
    uint32x4_t _r0d = vreinterpretq_u32_u16(_r0);                              \
    value1 = vcvtq_f32_u32(vshrq_n_u32(_r0d, 16));                             \
    _r0d = vshrq_n_u32(vshlq_n_u32(_r0d, 16), 16);                             \
    value0 = vcvtq_f32_u32(_r0d);                                              \
  }

#define ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_NEON(value, idx)                    \
  {                                                                            \
    uint32x4_t _r0, _r1;                                                       \
    _r0 = vld1q_u32((const uint32_t *)(pSrc + (idx)*srcStride));               \
    _r1 = vld1q_u32((const uint32_t *)(pSrc + (idx)*srcStride + 4));           \
    _r0 = vaddq_u32(                                                           \
        _r0, vld1q_u32((const uint32_t *)(pSrcNext + (idx)*srcStride)));       \
    _r1 = vaddq_u32(                                                           \
        _r1, vld1q_u32((const uint32_t *)(pSrcNext + (idx)*srcStride + 4)));   \
    uint32x4x2_t d = vzipq_u32(_r0, _r1);                                      \
    _r0 = d.val[0];                                                            \
    _r1 = d.val[1];                                                            \
    _r0 = vpaddq_u32(_r0, _r1);                                                \
    value = vmulq_f32(vcvtq_f32_u32(_r0), invWindowSize_sqd);                  \
  }

#define ASM_CALC_4_FLOAT_SSIM_NEON()                                           \
  {                                                                            \
    float32x4_t one = vdupq_n_f32(1);                                          \
    /* STEP 2. adjust values */                                                \
    float32x4_t both_sum_mul =                                                 \
        vmulq_f32(vmulq_f32(ref_sum, cmp_sum), invWindowSize_qd);              \
    float32x4_t ref_sum_sqd =                                                  \
        vmulq_f32(vmulq_f32(ref_sum, ref_sum), invWindowSize_qd);              \
    float32x4_t cmp_sum_sqd =                                                  \
        vmulq_f32(vmulq_f32(cmp_sum, cmp_sum), invWindowSize_qd);              \
    ref_sigma_sqd = vsubq_f32(ref_sigma_sqd, ref_sum_sqd);                     \
    cmp_sigma_sqd = vsubq_f32(cmp_sigma_sqd, cmp_sum_sqd);                     \
    sigma_both = vsubq_f32(sigma_both, both_sum_mul);                          \
    /* STEP 3. process numbers, do scale */                                    \
    {                                                                          \
      float32x4_t a = vaddq_f32(vaddq_f32(both_sum_mul, both_sum_mul), C1);    \
      float32x4_t b = vaddq_f32(sigma_both, halfC2);                           \
      float32x4_t c = vaddq_f32(vaddq_f32(ref_sum_sqd, cmp_sum_sqd), C1);      \
      float32x4_t d = vaddq_f32(vaddq_f32(ref_sigma_sqd, cmp_sigma_sqd), C2);  \
      float32x4_t ssim_val = vmulq_f32(a, b);                                  \
      ssim_val = vaddq_f32(ssim_val, ssim_val);                                \
      ssim_val = vdivq_f32(ssim_val, vmulq_f32(c, d));                         \
      ssim_sum = vaddq_f32(ssim_sum, ssim_val);                                \
      ssim_val = vsubq_f32(one, ssim_val);                                     \
      if (essim_mink_value == 4) {                                             \
        ssim_val = vmulq_f32(ssim_val, ssim_val);                              \
        ssim_val = vmulq_f32(ssim_val, ssim_val);                              \
      } else {                                                                 \
        ssim_val = vmulq_f32(vmulq_f32(ssim_val, ssim_val), ssim_val);         \
      }                                                                        \
      ssim_mink_sum = vaddq_f32(ssim_mink_sum, ssim_val);                      \
    }                                                                          \
  }

void sum_windows_8x4_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 8 };
  const float32x4_t invWindowSize_sqd =
      vdupq_n_f32(1.0f / (float)(windowSize * windowSize));
  const float32x4_t invWindowSize_qd =
      vmulq_f32(invWindowSize_sqd, invWindowSize_sqd);
  const float fC1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float fC2 = get_ssim_float_constant(2, bitDepthMinus8);
  const float32x4_t C1 = vdupq_n_f32(fC1);
  const float32x4_t C2 = vdupq_n_f32(fC2);
  const float32x4_t halfC2 = vdupq_n_f32(fC2 / 2.0f);

  float32x4_t ssim_mink_sum = vdupq_n_f32(0.0f);
  float32x4_t ssim_sum = vdupq_n_f32(0.0f);
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;

    float32x4_t ref_sum, cmp_sum;
    float32x4_t ref_sigma_sqd;
    float32x4_t cmp_sigma_sqd;
    float32x4_t sigma_both;

    ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_NEON(ref_sum, cmp_sum, 0);
    ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_NEON(ref_sigma_sqd, 1);
    ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_NEON(cmp_sigma_sqd, 2);
    ASM_LOAD_8X8_WINDOW_4_FLOAT_VALUES_NEON(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_4_FLOAT_SSIM_NEON();
  }

  ssim_sum = vpaddq_f32(ssim_sum, ssim_mink_sum);
  ssim_sum = vpaddq_f32(ssim_sum, ssim_sum);

  res->ssim_sum_f += vgetq_lane_f32(ssim_sum, 0);
  res->ssim_mink_sum_f += vgetq_lane_f32(ssim_sum, 1);
  res->numWindows += i;

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_8x4_float_8u_c(res, &buf, numWindows - i, windowSize,
                               windowStride, bitDepthMinus8,NULL,0,0,
                               essim_mink_value);
  }

} /* void sum_windows_8x4_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS) */

#if NEW_SIMD_FUNC

#define calc_window_ssim_int_8u_neon() \
  { \
    /* STEP 2. adjust values */ \
    _r0 = vbicq_u32(sum, fullMSB); \
    _r1 = vshrq_n_u32(sum, 16); \
    uint32x4_t both_sum_mul = vmulq_u32(_r0, _r1); \
    uint32x4_t ref_sum_sqd = vmulq_u32(_r0, _r0); \
    uint32x4_t cmp_sum_sqd = vmulq_u32(_r1, _r1); \
    ref_sigma_sqd = vsubq_u32(ref_sigma_sqd, ref_sum_sqd); \
    cmp_sigma_sqd = vsubq_u32(cmp_sigma_sqd, cmp_sum_sqd); \
    sigma_both_a = vreinterpretq_s32_u32(vshrq_n_u32(sigma_both, 1)); \
    sigma_both_a = vsubq_s32( \
        sigma_both_a, vreinterpretq_s32_u32(vshrq_n_u32(both_sum_mul, 1))); \
    /* STEP 3. process numbers, do scale */ \
    a = vaddq_u32(vshrq_n_u32(both_sum_mul, 1), quarterC1); \
    b = vaddq_s32(sigma_both_a, quarterC2); \
    ref_sum_sqd = vshrq_n_u32(ref_sum_sqd, 2); \
    cmp_sum_sqd = vshrq_n_u32(cmp_sum_sqd, 2); \
    c = vaddq_u32(vaddq_u32(ref_sum_sqd, cmp_sum_sqd), quarterC1); \
    ref_sigma_sqd = vshrq_n_u32(ref_sigma_sqd, 1); \
    cmp_sigma_sqd = vshrq_n_u32(cmp_sigma_sqd, 1); \
    d = vaddq_u32(vaddq_u32(ref_sigma_sqd, cmp_sigma_sqd), halfC2); \
    /* process numerators */ \
    { \
      int64x2_t _r; \
      _r = vmull_s32(vreinterpret_s32_u32(vget_low_u32(a)), vget_low_s32(b)); \
      vst1q_s64(num + 0, _r); \
      _r = vmull_high_s32(vreinterpretq_s32_u32(a), b); \
      vst1q_s64(num + 2, _r); \
    } \
    /* process denominators */ \
    { \
      int64x2_t _r; \
      _r = vreinterpretq_s64_u64(vmull_u32(vget_low_u32(c), vget_low_u32(d))); \
      vst1q_s64(denom + 0, _r); \
      _r = vreinterpretq_s64_u64(vmull_high_u32(c, d)); \
      vst1q_s64(denom + 2, _r); \
    } \
  } \

void sum_windows_8x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 8 };
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const uint32x4_t quarterC1 = vdupq_n_u32(C1 / 4);
  const uint32x4_t halfC2 = vdupq_n_u32(C2 / 2);
  const int32x4_t quarterC2 = vdupq_n_s32(C2 / 4);
  const uint32x4_t fullMSB = vdupq_n_u32(MSB);
  uint32x4_t a, c, d;
  int32x4_t b, sigma_both_a;
  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;
  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];
  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  float const_1 =
        1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);
  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;
    uint32x4_t sum;
    uint32x4_t ref_sigma_sqd;
    uint32x4_t cmp_sigma_sqd;
    uint32x4_t sigma_both;
    uint32x4_t _r0, _r1;
    uint16x8_t _r2, _r3;
    uint32x4x2_t t;
    uint16x8x2_t t_1;
    _r2 = vld1q_u16((const uint16_t *)(pSrc));
    _r3 = vld1q_u16((const uint16_t *)(pSrc + 4));
    _r2 = vaddq_u16(
          _r2, vld1q_u16((const uint16_t *)(pSrcNext)));
    _r3 = vaddq_u16(
          _r3, vld1q_u16((const uint16_t *)(pSrcNext + 4)));
    t_1 = vzipq_u16(_r2, _r3);
    _r2 = t_1.val[0];
    _r3 = t_1.val[1];
    sum = vreinterpretq_u32_u16(vpaddq_u16(_r2, _r3));
    _r0 = vld1q_u32((const uint32_t *)(pSrc + srcStride));
    _r1 = vld1q_u32((const uint32_t *)(pSrc + srcStride + 4));
    _r0 = vaddq_u32(
          _r0, vld1q_u32((const uint32_t *)(pSrcNext + srcStride)));
    _r1 = vaddq_u32(
          _r1, vld1q_u32((const uint32_t *)(pSrcNext + srcStride + 4)));
    t = vzipq_u32(_r0, _r1);
    _r0 = t.val[0];
    _r1 = t.val[1];
    _r0 = vpaddq_u32(_r0, _r1);
    ref_sigma_sqd = vshlq_n_u32(_r0, 6);
    _r0 = vld1q_u32((const uint32_t *)(pSrc + 2*srcStride));
    _r1 = vld1q_u32((const uint32_t *)(pSrc + 2*srcStride + 4));
    _r0 = vaddq_u32(
          _r0, vld1q_u32((const uint32_t *)(pSrcNext + 2*srcStride)));
    _r1 = vaddq_u32(
          _r1, vld1q_u32((const uint32_t *)(pSrcNext + 2*srcStride + 4)));
    t = vzipq_u32(_r0, _r1);
    _r0 = t.val[0];
    _r1 = t.val[1];
    _r0 = vpaddq_u32(_r0, _r1);
    cmp_sigma_sqd = vshlq_n_u32(_r0, 6);
    _r0 = vld1q_u32((const uint32_t *)(pSrc + 3*srcStride));
    _r1 = vld1q_u32((const uint32_t *)(pSrc + 3*srcStride + 4));
    _r0 = vaddq_u32(
          _r0, vld1q_u32((const uint32_t *)(pSrcNext + 3*srcStride)));
    _r1 = vaddq_u32(
          _r1, vld1q_u32((const uint32_t *)(pSrcNext + 3*srcStride + 4)));
    t = vzipq_u32(_r0, _r1);
    _r0 = t.val[0];
    _r1 = t.val[1];
    _r0 = vpaddq_u32(_r0, _r1);
    sigma_both = vshlq_n_u32(_r0, 6);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;
    // CALC

    calc_window_ssim_int_8u_neon();

    int power_val;
    uint16_t i16_map_denom;
    int64_t ssim_val;
    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      i16_map_denom = get_best_i16_from_u64((uint64_t)denom[w], &power_val);
      num[w] = num[w] << 1;
      ssim_val = (((num[w] >> power_val) * div_lookup_ptr[i16_map_denom]) +
                  SSIMValRtShiftHalfRound) >> SSIMValRtShiftBits;
      ssim_sum += ssim_val;
      int64_t const_1_minus_ssim_val = const_1 - ssim_val;
      if(essim_mink_value == 4) {
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val * const_1_minus_ssim_val;
      } else {
            /*essim_mink_value == 3*/
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val;
      }
      ssim_mink_sum += mink_pow_ssim_val;
    }
  }
  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += i;
  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_int_8u_c(res, &buf, numWindows - i, windowSize,
                             windowStride, bitDepthMinus8, div_lookup_ptr,
                             SSIMValRtShiftBits, SSIMValRtShiftHalfRound,
                             essim_mink_value);
  }
} /* void sum_windows_8x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS) */

void sum_windows_8x8_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 8 };
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const uint32x4_t quarterC1 = vdupq_n_u32(C1 / 4);
  const uint32x4_t halfC2 = vdupq_n_u32(C2 / 2);
  const int32x4_t quarterC2 = vdupq_n_s32(C2 / 4);
  const uint32x4_t fullMSB = vdupq_n_u32(MSB);
  uint32x4_t a, c, d;
  int32x4_t b, sigma_both_a;
  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;
  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];
  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  float const_1 =
        1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);
  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;
    uint32x4_t sum;
    uint32x4_t ref_sigma_sqd;
    uint32x4_t cmp_sigma_sqd;
    uint32x4_t sigma_both;
    uint32x4_t _r0, _r1, _r2, _r3, _r4;
    for (uint32_t x = 0; x <= 16; x+=16) {
      sum = _r1;
      ref_sigma_sqd = _r2;
      cmp_sigma_sqd = _r3;
      sigma_both = _r4;
      _r0 = vld1q_u32((const uint32_t *)(pSrc + x));
      _r1 = vld1q_u32((const uint32_t *)(pSrcNext + x));
      _r1 = vaddq_u32(_r0, _r1);
      _r0 = vld1q_u32((const uint32_t *)(pSrc + x + srcStride));
      _r2 = vld1q_u32((const uint32_t *)(pSrcNext + x + srcStride));
      _r2 = vaddq_u32(_r0, _r2);
      _r0 = vld1q_u32((const uint32_t *)(pSrc + x + 2*srcStride));
      _r3 = vld1q_u32((const uint32_t *)(pSrcNext + x + 2*srcStride));
      _r3 = vaddq_u32(_r0, _r3);
      _r0 = vld1q_u32((const uint32_t *)(pSrc + x + 3*srcStride));
      _r4 = vld1q_u32((const uint32_t *)(pSrcNext + x + 3*srcStride));
      _r4 = vaddq_u32(_r0, _r4);
    }
    sum = vpaddq_u32(sum, _r1);
    ref_sigma_sqd = vpaddq_u32(ref_sigma_sqd, _r2);
    cmp_sigma_sqd = vpaddq_u32(cmp_sigma_sqd, _r3);
    sigma_both = vpaddq_u32(sigma_both, _r4);
    pSrc += sizeof(uint32_t) * WIN_SIZE;
    // CALC
    ref_sigma_sqd = vshlq_n_u32(ref_sigma_sqd, 6);
    cmp_sigma_sqd = vshlq_n_u32(cmp_sigma_sqd, 6);
    sigma_both = vshlq_n_u32(sigma_both, 6);

    calc_window_ssim_int_8u_neon();

    int power_val;
    uint16_t i16_map_denom;
    int64_t ssim_val;
    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      i16_map_denom = get_best_i16_from_u64((uint64_t)denom[w], &power_val);
      num[w] = num[w] << 1;
      ssim_val = (((num[w] >> power_val) * div_lookup_ptr[i16_map_denom]) +
                  SSIMValRtShiftHalfRound) >> SSIMValRtShiftBits;
      ssim_sum += ssim_val;
      int64_t const_1_minus_ssim_val = const_1 - ssim_val;
      if(essim_mink_value == 4) {
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val * const_1_minus_ssim_val;
      } else {
            /*essim_mink_value == 3*/
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val;
      }
      ssim_mink_sum += mink_pow_ssim_val;
    }
  }
  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += i;
  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_int_8u_c(res, &buf, numWindows - i, windowSize,
                             windowStride, bitDepthMinus8, div_lookup_ptr,
                             SSIMValRtShiftBits, SSIMValRtShiftHalfRound,
                             essim_mink_value);
  }
}

void sum_windows_16x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS){
  sum_windows_16_int_8u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x8_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS){
  sum_windows_16_int_8u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}
void sum_windows_16x16_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS){
  sum_windows_16_int_8u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 4, WIN_SIZE = 16 };
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const uint32x4_t quarterC1 = vdupq_n_u32(C1 / 4);
  const uint32x4_t halfC2 = vdupq_n_u32(C2 / 2);
  const int32x4_t quarterC2 = vdupq_n_s32(C2 / 4);
  const uint32x4_t fullMSB = vdupq_n_u32(MSB);
  uint32x4_t a, c, d;
  int32x4_t b, sigma_both_a;
  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;
  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];
  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  float const_1 =
        1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);
  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    uint32x4_t sum = vdupq_n_u32(0);
    uint32x4_t ref_sigma_sqd = vdupq_n_u32(0);
    uint32x4_t cmp_sigma_sqd = vdupq_n_u32(0);
    uint32x4_t sigma_both = vdupq_n_u32(0);
    uint32x4_t _r0, _r1, _r2, _r3;
    for (uint32_t x = 0; x < 16; x+=4){
      _r0 = vld1q_u32(
                (const uint32_t *)(pSrc + x*srcStride));
      _r1 = vld1q_u32(
                (const uint32_t *)(pSrc + windowStride + x*srcStride));
      _r2 = vld1q_u32(
                (const uint32_t *)(pSrc + 2*windowStride + x*srcStride));
      _r3 = vld1q_u32(
                (const uint32_t *)(pSrc + 3*windowStride + x*srcStride));
      _r1 = vpaddq_u32(_r0, _r1);
      _r3 = vpaddq_u32(_r2, _r3);
      _r1 = vpaddq_u32(_r1, _r3);
      sum = vaddq_u32(sum, _r1);
      _r0 = vld1q_u32(
                (const uint32_t *)(pSrc + (x+1)*srcStride));
      _r1 = vld1q_u32(
                (const uint32_t *)(pSrc + windowStride + (x+1)*srcStride));
      _r2 = vld1q_u32(
                (const uint32_t *)(pSrc + 2*windowStride + (x+1)*srcStride));
      _r3 = vld1q_u32(
                (const uint32_t *)(pSrc + 3*windowStride + (x+1)*srcStride));
      _r1 = vpaddq_u32(_r0, _r1);
      _r3 = vpaddq_u32(_r2, _r3);
      _r1 = vpaddq_u32(_r1, _r3);
      ref_sigma_sqd = vaddq_u32(ref_sigma_sqd, _r1);
      _r0 = vld1q_u32(
                (const uint32_t *)(pSrc + (x+2)*srcStride));
      _r1 = vld1q_u32(
                (const uint32_t *)(pSrc + windowStride + (x+2)*srcStride));
      _r2 = vld1q_u32(
                (const uint32_t *)(pSrc + 2*windowStride + (x+2)*srcStride));
      _r3 = vld1q_u32(
                (const uint32_t *)(pSrc + 3*windowStride + (x+2)*srcStride));
      _r1 = vpaddq_u32(_r0, _r1);
      _r3 = vpaddq_u32(_r2, _r3);
      _r1 = vpaddq_u32(_r1, _r3);
      cmp_sigma_sqd = vaddq_u32(cmp_sigma_sqd, _r1);
      _r0 = vld1q_u32(
                (const uint32_t *)(pSrc + (x+3)*srcStride));
      _r1 = vld1q_u32(
                (const uint32_t *)(pSrc + windowStride + (x+3)*srcStride));
      _r2 = vld1q_u32(
                (const uint32_t *)(pSrc + 2*windowStride + (x+3)*srcStride));
      _r3 = vld1q_u32(
                (const uint32_t *)(pSrc + 3*windowStride + (x+3)*srcStride));
      _r1 = vpaddq_u32(_r0, _r1);
      _r3 = vpaddq_u32(_r2, _r3);
      _r1 = vpaddq_u32(_r1, _r3);
      sigma_both = vaddq_u32(sigma_both, _r1);
    }
    pSrc += sizeof(uint32_t) * windowStride;
    // CALC
    ref_sigma_sqd = vshlq_n_u32(ref_sigma_sqd, 8);
    cmp_sigma_sqd = vshlq_n_u32(cmp_sigma_sqd, 8);
    sigma_both = vshlq_n_u32(sigma_both, 8);

    calc_window_ssim_int_8u_neon();

    int power_val;
    uint16_t i16_map_denom;
    int64_t ssim_val;
    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      i16_map_denom = get_best_i16_from_u64((uint64_t)denom[w], &power_val);
      num[w] = num[w] << 1;
      ssim_val = (((num[w] >> power_val) * div_lookup_ptr[i16_map_denom]) +
                  SSIMValRtShiftHalfRound) >> SSIMValRtShiftBits;
      ssim_sum += ssim_val;
      int64_t const_1_minus_ssim_val = const_1 - ssim_val;
      if(essim_mink_value == 4) {
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val * const_1_minus_ssim_val;
      } else {
            /*essim_mink_value == 3*/
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val;
      }
      ssim_mink_sum += mink_pow_ssim_val;
    }
  }
  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += i;
  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_int_8u_c(res, &buf, numWindows - i, windowSize,
                             windowStride, bitDepthMinus8, div_lookup_ptr,
                             SSIMValRtShiftBits, SSIMValRtShiftHalfRound,
                             essim_mink_value);
  }
}

#define calc_window_ssim_int_10u_neon() \
  { \
    /* STEP 2. adjust values */ \
    uint64_t a_64[2], c_64[2], d_64[2]; \
    int64_t b_64[2]; \
    uint32x2_t ref_sum_2 = vreinterpret_u32_u64(vadd_u64(vget_low_u64(ref_sum), \
                            vshl_n_u64(vget_high_u64(ref_sum), 32))); \
    uint32x2_t cmp_sum_2 = vreinterpret_u32_u64(vadd_u64(vget_low_u64(cmp_sum), \
                            vshl_n_u64(vget_high_u64(cmp_sum), 32))); \
    uint64x2_t both_sum_mul = vmull_u32(ref_sum_2, cmp_sum_2); \
    uint64x2_t ref_sum_sqd = vmull_u32(ref_sum_2, ref_sum_2); \
    uint64x2_t cmp_sum_sqd = vmull_u32(cmp_sum_2, cmp_sum_2); \
    ref_sigma_sqd = vsubq_u64(ref_sigma_sqd, ref_sum_sqd); \
    cmp_sigma_sqd = vsubq_u64(cmp_sigma_sqd, cmp_sum_sqd); \
    sigma_both_a = vsubq_s64( \
      vreinterpretq_s64_u64(sigma_both),vreinterpretq_s64_u64(both_sum_mul)); \
    /* STEP 3. process numbers, do scale */ \
    a = vaddq_u64(vshlq_n_u64(both_sum_mul, 1), fullC1); \
    b = vaddq_s64(sigma_both_a, halfC2); \
    c = vaddq_u64(vaddq_u64(ref_sum_sqd, cmp_sum_sqd), fullC1); \
    d = vaddq_u64(vaddq_u64(ref_sigma_sqd, cmp_sigma_sqd), fullC2); \
    /* process numerators */ \
    vst1q_u64(a_64, a); \
    vst1q_s64(b_64, b); \
    num[0] = (a_64[0] >> 5) * (b_64[0] >> 5); \
    num[1] = (a_64[1] >> 5) * (b_64[1] >> 5); \
    /* process denominators */ \
    vst1q_u64(c_64, c); \
    vst1q_u64(d_64, d); \
    denom[0] = ((c_64[0] >> 5) * (d_64[0] >> 5) >> 1); \
    denom[1] = ((c_64[1] >> 5) * (d_64[1] >> 5) >> 1); \
  } \

void sum_windows_8x4_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_8x4_int_8u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_8x8_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_8x8_int_8u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x4_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_16_int_10u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x8_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_16_int_10u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16x16_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  sum_windows_16_int_10u_neon(SUM_WINDOWS_ACTUAL_ARGS);
}

void sum_windows_16_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 2, WIN_SIZE = 16 };
  const uint32_t C1 = get_ssim_int_constant(1, bitDepthMinus8, windowSize);
  const uint32_t C2 = get_ssim_int_constant(2, bitDepthMinus8, windowSize);
  const uint64x2_t fullC1 = vdupq_n_u64(C1);
  const uint64x2_t fullC2 = vdupq_n_u64(C2);
  const int64x2_t halfC2 = vdupq_n_s64(C2>>1);
  const uint64x2_t fullMSB = vdupq_n_u64(MSB);
  uint64x2_t a, c, d;
  int64x2_t b, sigma_both_a;
  int64_t ssim_mink_sum = 0, ssim_sum = 0;
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;
  int64_t num[WIN_CHUNK];
  int64_t denom[WIN_CHUNK];
  int32_t extraRtShiftBitsForSSIMVal =
          (int32_t)SSIMValRtShiftBits - DEFAULT_Q_FORMAT_FOR_SSIM_VAL;
  int64_t mink_pow_ssim_val = 0;
  float const_1 =
        1 << (DEFAULT_Q_FORMAT_FOR_SSIM_VAL - extraRtShiftBitsForSSIMVal);
  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    uint64x2_t ref_sum = vdupq_n_u64(0);
    uint64x2_t cmp_sum = vdupq_n_u64(0);
    uint64x2_t ref_sigma_sqd = vdupq_n_u64(0);
    uint64x2_t cmp_sigma_sqd = vdupq_n_u64(0);
    uint64x2_t sigma_both = vdupq_n_u64(0);
    uint32x4_t _r0, _r1;
    uint64x2_t _r2;
    for (uint32_t x = 0; x < 16; x+=4){
      _r0 = vld1q_u32((const uint32_t *)(pSrc + x*srcStride));
      _r1 = vld1q_u32((const uint32_t *)(pSrc + windowStride + x*srcStride));
      _r2 = vpaddlq_u32(vpaddq_u32(_r0, _r1));
      ref_sum = vaddq_u64(ref_sum, (vbicq_u64(_r2, fullMSB)));
      cmp_sum = vaddq_u64(cmp_sum, (vshrq_n_u64(_r2, 16)));
      _r0 = vld1q_u32((const uint32_t *)(pSrc + (x+1)*srcStride));
      _r1 = vld1q_u32((const uint32_t *)(pSrc + windowStride + (x+1)*srcStride));
      _r2 = vpaddlq_u32(vpaddq_u32(_r0, _r1));
      ref_sigma_sqd = vaddq_u64(ref_sigma_sqd, _r2);
      _r0 = vld1q_u32((const uint32_t *)(pSrc + (x+2)*srcStride));
      _r1 = vld1q_u32((const uint32_t *)(pSrc + windowStride + (x+2)*srcStride));
      _r2 = vpaddlq_u32(vpaddq_u32(_r0, _r1));
      cmp_sigma_sqd = vaddq_u64(cmp_sigma_sqd, _r2);
      _r0 = vld1q_u32((const uint32_t *)(pSrc + (x+3)*srcStride));
      _r1 = vld1q_u32((const uint32_t *)(pSrc + windowStride + (x+3)*srcStride));
      _r2 = vpaddlq_u32(vpaddq_u32(_r0, _r1));
      sigma_both = vaddq_u64(sigma_both, _r2);
    }
    pSrc += sizeof(uint16_t) * windowStride;
    // CALC
    ref_sigma_sqd = vshlq_n_u64(ref_sigma_sqd, 8);
    cmp_sigma_sqd = vshlq_n_u64(cmp_sigma_sqd, 8);
    sigma_both = vshlq_n_u64(sigma_both, 8);

    calc_window_ssim_int_10u_neon();

    int power_val;
    uint16_t i16_map_denom;
    int64_t ssim_val;
    for (size_t w = 0; w < WIN_CHUNK; ++w) {
      i16_map_denom = get_best_i16_from_u64((uint64_t)denom[w], &power_val);
      ssim_val = (((num[w] >> power_val) * div_lookup_ptr[i16_map_denom]) +
                  SSIMValRtShiftHalfRound) >> SSIMValRtShiftBits;
      ssim_sum += ssim_val;
      int64_t const_1_minus_ssim_val = const_1 - ssim_val;
      if(essim_mink_value == 4) {
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val * const_1_minus_ssim_val;
      } else {
            /*essim_mink_value == 3*/
            mink_pow_ssim_val = const_1_minus_ssim_val * const_1_minus_ssim_val *
                                const_1_minus_ssim_val;
      }
      ssim_mink_sum += mink_pow_ssim_val;
    }
  }
  res->ssim_sum += ssim_sum;
  res->ssim_mink_sum += ssim_mink_sum;
  res->numWindows += i;
  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
    sum_windows_int_10u_c(res, &buf, numWindows - i, windowSize,
                             windowStride, bitDepthMinus8, div_lookup_ptr,
                             SSIMValRtShiftBits, SSIMValRtShiftHalfRound,
							 essim_mink_value);
  }
}

#endif
#endif /* defined(_ARM) || defined(_ARM64) */
