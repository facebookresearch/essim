/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(ESSIM_CPU_H)
#define ESSIM_CPU_H

#include <essim/inc/xplatform.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

typedef enum eCPUType {

  cpuUnk = -1,
  cpuPlain = 0,

#if defined(_X86) || defined(_X64)

  cpu_sse = (1 << 0),
  cpu_sse2 = (1 << 1) | cpu_sse,
  cpu_sse3 = (1 << 2) | cpu_sse2,
  cpu_ssse3 = (1 << 3) | cpu_sse3,
  cpu_sse41 = (1 << 4) | cpu_ssse3,
  cpu_sse42 = (1 << 5) | cpu_sse41,
  cpu_avx = (1 << 6) | cpu_sse42,
  cpu_avx2 = (1 << 7) | cpu_avx,
  cpu_avx512f = (1 << 8) | cpu_avx2,
  cpu_avx512dq = (1 << 9) | cpu_avx512f,
  cpu_avx512bw = (1 << 10) | cpu_avx512f,
  cpu_avx512bwdq = cpu_avx512bw | cpu_avx512dq

#elif defined(__ARM_NEON)

  cpu_neon = (1 << 0),

#endif /* defined(_X86) || defined(_X64) */

} eCPUType;

eCPUType GetCpuType(void);

void SetCpuType(const eCPUType cpuType);

#if defined(__cplusplus)
} // extern "C"
#endif /* defined(__cplusplus) */

#endif /* !defined(ESSIM_CPU_H) */
