/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(__SSIM_SSIM_INTERNAL_H)
#define __SSIM_SSIM_INTERNAL_H

#include <essim/inc/xplatform.h>
#include <essim/essim.h>

#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

enum { LOG2_ALIGN = 6, ALIGN = 1 << LOG2_ALIGN };

#define INTR_PROFILING_PRINTS 0
#define ENABLE_ONLY_C_PATH 0
#define ARM_BUILD_FIXES 1
#define NEW_SIMD_FUNC 1
#define NEW_10BIT_C_FUNC 1

/*Max WxH eSSSIM can support*/
#define MAX_FRAME_WIDTH 7680
#define MAX_FRAME_HEIGHT 4320

/*Default Q format, per window size ssim_val is 15 bits.
  But it will vary based on frame WxH, window size, window stride & mink pooling
*/
#define DEFAULT_Q_FORMAT_FOR_SSIM_VAL 15
/*ssim_val can be in the range of -1 to 1. so if ssim_val is -ve & mink pooling is 3,
then ssim_accumulated_sum can be -ve, so we need one bit for sign representation.
Based on above cases, we consider SSIM_ACCUMULATED_SUM maximum value can be
(2^64)-1 (for mink 4) or 2^63 (for mink 3).
*/

#define div_Q_factor 1073741824  //2^30

#if NEW_SIMD_FUNC
#define MSB 4294901760 //2^32-2^16
#define LSB 65535 //2^16
#endif
/*Max WxH = 3840x2160
  Max WindowSize = 16x16
  Min windowStride = 4
  With above parameters we get maximum total No. of windows per frame,
  when we convert that number into bits, we get 19 bits.
*/

#pragma pack(push, 1)

typedef struct WINDOW_STATS {
  uint32_t ref_sum;
  uint32_t cmp_sum;
  uint64_t ref_sigma_sqd;
  uint64_t cmp_sigma_sqd;
  uint64_t sigma_both;
} WINDOW_STATS;

/* the temporal buffer has the following structure:
a row of 16u(32u) interleaving sum values for ref and cmp,
a row of 32u(64u) ref squared values,
a row of 32u(64u) cmp squared values,
a row of 32u(64u) sigma values, then again a line of 16u values etc.
the byte distance between rows is stride */

typedef struct SSIM_4X4_WINDOW_BUFFER {
  void* p;
  ptrdiff_t stride;

} SSIM_4X4_WINDOW_BUFFER;

typedef struct SSIM_4X4_WINDOW_ROW {
  /* pointers to row data */
  SSIM_4X4_WINDOW_BUFFER ptrs;

  /* vertical coordinate of the last processed row */
  uint32_t y;

} SSIM_4X4_WINDOW_ROW;

typedef struct SSIM_RES {
  /* integer function results */
  uint64_t ssim_sum;
  uint64_t ssim_mink_sum;

  /* float function results */
  float ssim_sum_f;
  double ssim_mink_sum_f;

  /* number of windows summed */
  size_t numWindows;
} SSIM_RES;

typedef struct SSIM_SRC {
  /* pointer to reference data */
  const void* ref;
  /* reference data stride */
  ptrdiff_t refStride;
  /* pointer to reconstructed data */
  const void* cmp;
  /* reconstructed data stride */
  ptrdiff_t cmpStride;
} SSIM_SRC;

/*
    precision function types
*/

#define LOAD_WINDOW_FORMAL_ARGS                         \
  WINDOW_STATS *const pWnd, const SSIM_SRC *const pSrc, \
      const uint32_t windowSize
#define LOAD_WINDOW_ACTUAL_ARGS pWnd, pSrc, windowSize

#define CALC_WINDOW_SSIM_FORMAL_ARGS                                      \
  WINDOW_STATS *const pWnd, const uint32_t windowSize, const uint32_t C1, \
      const uint32_t C2, uint32_t* div_lookup_ptr, \
      uint32_t SSIMValRtShiftBits, uint32_t SSIMValRtShiftHalfRound
#define CALC_WINDOW_SSIM_ACTUAL_ARGS                                           \
  pWnd, windowSize, C1, C2, div_lookup_ptr,                                    \
      SSIMValRtShiftBits SSIMValRtShiftHalfRound

typedef void (*load_window_proc_t)(LOAD_WINDOW_FORMAL_ARGS);
typedef int64_t (*calc_window_ssim_proc_t)(CALC_WINDOW_SSIM_FORMAL_ARGS);

/*
    performance function types
*/

#define LOAD_4x4_WINDOWS_FORMAL_ARGS                                    \
  const SSIM_4X4_WINDOW_BUFFER *const pBuf, const size_t num4x4Windows, \
      const SSIM_SRC *const pSrc
#define LOAD_4x4_WINDOWS_ACTUAL_ARGS pBuf, num4x4Windows, pSrc

#define SUM_WINDOWS_FORMAL_ARGS                            \
  SSIM_RES *const res, SSIM_4X4_WINDOW_BUFFER *const pBuf, \
      const size_t numWindows, const uint32_t windowSize,  \
      const uint32_t windowStride, const uint32_t bitDepthMinus8, \
      uint32_t *div_lookup_ptr, uint32_t SSIMValRtShiftBits, \
      uint32_t SSIMValRtShiftHalfRound, const uint32_t essim_mink_value
#define SUM_WINDOWS_ACTUAL_ARGS                                                \
  res, pBuf, numWindows, windowSize, windowStride, bitDepthMinus8,             \
      div_lookup_ptr, SSIMValRtShiftBits, SSIMValRtShiftHalfRound,             \
      essim_mink_value

typedef void (*load_4x4_windows_proc_t)(LOAD_4x4_WINDOWS_FORMAL_ARGS);
typedef void (*sum_windows_proc_t)(SUM_WINDOWS_FORMAL_ARGS);

typedef struct SSIM_PARAMS {
  /* stream parameters */
  uint32_t width;
  uint32_t height;
  uint32_t bitDepthMinus8;
  eSSIMDataType dataType;

  /* SSIM parameters */
  uint32_t windowSize;
  uint32_t windowStride;
  eSSIMMode mode;
  eSSIMFlags flags;

  /* processing functions */
  load_window_proc_t load_window_proc;
  calc_window_ssim_proc_t calc_window_ssim_proc;
  load_4x4_windows_proc_t load_4x4_windows_proc;
  sum_windows_proc_t sum_windows_proc;
} SSIM_PARAMS;

struct SSIM_CTX {
  void* buffer;
  size_t bufferSize;
  ptrdiff_t bufferStride;

  SSIM_4X4_WINDOW_ROW* windowRows;
  size_t numWindowRows;

  SSIM_RES res;

  const SSIM_PARAMS *params;
  uint32_t * div_lookup_ptr;

 /*Default Q format bits for window ssim_val is 15, but sometimes it will cause
  over flow. So we need additional Rt shift bits, based on frame WxH, window size, window stride,
  and mink pooling, below parameter will hold final amount of Rt shift bits.*/
  uint32_t SSIMValRtShiftBits;
  uint32_t SSIMValRtShiftHalfRound;
};

struct SSIM_CTX_ARRAY {
  SSIM_CTX** ctx;
  size_t numCtx;

  SSIM_PARAMS params;

  uint32_t d2h;
};

#pragma pack(pop)

/*
    declare tool functions
*/

/* get the number of windows 1D */
uint32_t GetNum4x4Windows(
    const uint32_t value,
    const uint32_t windowSize,
    const uint32_t windowStride);
uint32_t GetNumWindows(
    const uint32_t value,
    const uint32_t windowSize,
    const uint32_t windowStride);

/* get the number of windows 2D */
uint32_t GetTotalWindows(
    const uint32_t width,
    const uint32_t height,
    const uint32_t windowSize,
    const uint32_t windowStride);

/* advance a pointer on a stride in bytes */
void* AdvancePointer(const void* p, const ptrdiff_t stride);

uint32_t get_ssim_int_constant(
    const uint32_t constIdx,
    const uint32_t bitDepthMinus8,
    const uint32_t windowSize);
float get_ssim_float_constant(
    const uint32_t constIdx,
    const uint32_t bitDepthMinus8);

/* Function to get no of bits in binary
   representation of positive integer. */
uint32_t GetTotalBitsInNumber(uint32_t number);

/*generating lookup table for Q16 format*/
extern uint32_t div_lookup[];
//const uint32_t div_Q_factor = 1073741824; // 2^30
uint32_t* div_lookup_generator(void);

/*get best 16 bits*/
uint16_t get_best_i16_from_u64(uint64_t temp, int *power);

void load_window_8u_c(LOAD_WINDOW_FORMAL_ARGS);
void load_window_16u_c(LOAD_WINDOW_FORMAL_ARGS);

int64_t calc_window_ssim_int_8u(CALC_WINDOW_SSIM_FORMAL_ARGS);
int64_t calc_window_ssim_int_10bd(CALC_WINDOW_SSIM_FORMAL_ARGS);
int64_t calc_window_ssim_int_16u(CALC_WINDOW_SSIM_FORMAL_ARGS);
float calc_window_ssim_float(WINDOW_STATS *const pWnd,
                             const uint32_t windowSize, const float C1,
                             const float C2);
eSSIMResult ssim_compute_prec(
    SSIM_CTX* const ctx,
    const void* ref,
    const ptrdiff_t refStride,
    const void* cmp,
    const ptrdiff_t cmpStride,
    const uint32_t essim_mink_value);

eSSIMResult ssim_compute_perf(SSIM_CTX *const ctx, const void *ref,
                              const ptrdiff_t refStride, const void *cmp,
                              const ptrdiff_t cmpStride, const uint32_t roiY,
                              const uint32_t roiHeight,
                              const uint32_t essim_mink_value);
/*
    declare optimized functions callers
*/

void load_4x4_windows_8u(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_int_8u(SUM_WINDOWS_FORMAL_ARGS);
#if NEW_SIMD_FUNC
void load_4x4_windows_10u(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x8_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_8u(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_int_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_int_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_int_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_10u(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x8_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_8u(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_float_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_10u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_10u(SUM_WINDOWS_FORMAL_ARGS);
#endif
void sum_windows_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u(SUM_WINDOWS_FORMAL_ARGS);

void load_4x4_windows_16u(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_16u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_float_16u(SUM_WINDOWS_FORMAL_ARGS);

/*
    declare reference functions
*/

void load_4x4_windows_8u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);

#if NEW_10BIT_C_FUNC
void load_4x4_windows_10u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_int_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_int_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_int_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_10u_c(SUM_WINDOWS_FORMAL_ARGS);
#endif

#if NEW_SIMD_FUNC
void sum_windows_8x8_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x8_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_float_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_10u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_10u_c(SUM_WINDOWS_FORMAL_ARGS);
#endif
void sum_windows_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);

void load_4x4_windows_16u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_16u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_float_16u_c(SUM_WINDOWS_FORMAL_ARGS);

/*
    declare optimized functions
*/

#if defined(_X86) || defined(_X64)

void load_4x4_windows_8u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void load_4x4_windows_8u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void load_4x4_windows_16u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void load_4x4_windows_16u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS);

#if NEW_SIMD_FUNC
void load_4x4_windows_10u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_int_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_int_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16_int_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x8_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_float_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_10u_avx2(SUM_WINDOWS_FORMAL_ARGS);
#endif

void sum_windows_8x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);

#elif defined(_ARM) || defined(_ARM64)

void load_4x4_windows_8u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void load_4x4_windows_16u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
#if NEW_SIMD_FUNC
void load_4x4_windows_10u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x8_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_int_10u_neon(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x8_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x8_float_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16_float_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x4_float_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x8_float_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_16x16_float_10u_neon(SUM_WINDOWS_FORMAL_ARGS);
#endif

#endif /* defined(_X86) || defined(_X64) */

#if defined(__cplusplus)
} // extern "C"
#endif /* defined(__cplusplus) */

#endif /* !defined(__SSIM_SSIM_INTERNAL_H) */
