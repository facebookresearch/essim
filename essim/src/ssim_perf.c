/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/essim.h>
#include <essim/inc/internal.h>

eSSIMResult ssim_compute_perf(SSIM_CTX *const ctx, const void *ref,
                              const ptrdiff_t refStride, const void *cmp,
                              const ptrdiff_t cmpStride, const uint32_t roiY,
                              const uint32_t roiHeight,
                              const uint32_t essim_mink_value) {
  const load_4x4_windows_proc_t load_4x4_windows_proc =
      ctx->params->load_4x4_windows_proc;
  const sum_windows_proc_t sum_windows_proc = ctx->params->sum_windows_proc;

  const uint32_t width = ctx->params->width;
  const uint32_t height = ctx->params->height;
  const uint32_t windowSize = ctx->params->windowSize;
  const uint32_t windowStride = ctx->params->windowStride;
  const uint32_t numWindows = GetNumWindows(width, windowSize, windowStride);
  const uint32_t num4x4Windows =
      GetNum4x4Windows(width, windowSize, windowStride);

  const uint32_t beginHeight =
      (roiY + windowStride - 1) / windowStride * windowStride;
  const uint32_t endHeight = roiY + roiHeight;

  const uint32_t numBuffers = windowSize / 4;

  SSIM_SRC src;
  src.refStride = refStride;
  src.cmpStride = cmpStride;

  for (uint32_t y = beginHeight; (y < endHeight) && (y + windowSize <= height);
       y += windowStride) {
    size_t rowIdx = (y / 4) % numBuffers;
    src.ref = AdvancePointer(ref, y * refStride);
    src.cmp = AdvancePointer(cmp, y * cmpStride);

    /* load 4x4 window row */
    for (uint32_t b = 0; b < numBuffers; ++b) {
      SSIM_4X4_WINDOW_ROW *const windowRow = ctx->windowRows + rowIdx;

      if (windowRow->y != y + 4 * b) {
        load_4x4_windows_proc(&windowRow->ptrs, num4x4Windows, &src);
        windowRow->y = y + 4 * b;
      }

      rowIdx = (rowIdx + 1) % numBuffers;
      src.ref = AdvancePointer(src.ref, 4 * refStride);
      src.cmp = AdvancePointer(src.cmp, 4 * cmpStride);
    }

    /* sum up windows */
#if INTR_PROFILING_PRINTS
    clock_t start=0, end=0;
    double cpu_time_used=0;
    start = clock();
#endif
    sum_windows_proc(&ctx->res, &ctx->windowRows[0].ptrs, numWindows,
                     windowSize, windowStride, ctx->params->bitDepthMinus8,
                     ctx->div_lookup_ptr, ctx->SSIMValRtShiftBits,
                     ctx->SSIMValRtShiftHalfRound, essim_mink_value);
#if INTR_PROFILING_PRINTS
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\t numWindows: %i \n",numWindows);
    printf("\t cpu_time_used/numWindows: %lf microsecs\n",(cpu_time_used/(double)numWindows)*1000000);
#endif
  }

  return SSIM_OK;

} /* eSSIMResult ssim_compute_perf(SSIM_CTX* const ctx, */
