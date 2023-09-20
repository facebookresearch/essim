/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/inc/memory.h>

#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define max(a, b) (((a) > (b)) ? (a) : (b))

void *ssim_alloc_aligned(const size_t size, const uint32_t log2Align) {
  ptrdiff_t align = (ptrdiff_t)1 << max(min(log2Align, 12), 6);

  /* allocated enough memory to save the real pointer and to do alignment */
  void *pAllocated = malloc(size + align + sizeof(void *));
  if (NULL == pAllocated) {
    return NULL;
  }

  /* align the pointer */
  void *p = (void *)(((ptrdiff_t)pAllocated + sizeof(void *) + (align - 1)) &
                     (-align));

  /* save the real pointer */
  void **pp = (void **)p;
  pp[-1] = pAllocated;

  return p;

} /* void *ssim_alloc_aligned(const size_t size, const uint32_t log2Align) */

void ssim_free_aligned(void *p) {
  if (NULL == p) {
    return;
  }

  void **pp = (void **)p;
  free(pp[-1]);

} /* void ssim_free_aligned(void *p) */
