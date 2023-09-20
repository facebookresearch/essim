/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(__SSIM_MEMORY_H)
#define __SSIM_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

void* ssim_alloc_aligned(const size_t size, const uint32_t log2Align);

void ssim_free_aligned(void* p);

#if defined(__cplusplus)
} // extern "C"
#endif /* defined(__cplusplus) */

#endif /* !defined(__SSIM_MEMORY_H) */
