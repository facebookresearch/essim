/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/inc/cpu.h>
#include <essim/inc/internal.h>

#define IMPL_PROC_0(name, FORMAL_ARGS, ACTUAL_ARGS)                            \
  /* implement the caller */                                                   \
  void name FORMAL_ARGS { name##_c ACTUAL_ARGS; }

#define IMPL_PROC_1(name, FORMAL_ARGS, ACTUAL_ARGS, cpu0)                      \
  /* forward declaration of the init proc */                                   \
  static void name##_init FORMAL_ARGS;                                         \
  /* declare the function pointer */                                           \
  static void(*p##name) FORMAL_ARGS = name##_init;                             \
  /* implement the dispatcher */                                               \
  static void name##_init FORMAL_ARGS {                                        \
    eCPUType cpuType = GetCpuType();                                           \
    if (cpu_##cpu0 == (cpu_##cpu0 & cpuType)) {                                \
      p##name = name##_##cpu0;                                                 \
    } else {                                                                   \
      p##name = name##_c;                                                      \
    }                                                                          \
    if(ENABLE_ONLY_C_PATH) { \
      p##name = name##_c; \
    } \
    p##name ACTUAL_ARGS;                                                       \
  }                                                                            \
  /* implement the caller */                                                   \
  void name FORMAL_ARGS { p##name ACTUAL_ARGS; }

#define IMPL_PROC_2(name, FORMAL_ARGS, ACTUAL_ARGS, cpu0, cpu1)                \
  /* forward declaration of the init proc */                                   \
  static void name##_init FORMAL_ARGS;                                         \
  /* declare the function pointer */                                           \
  static void(*p##name) FORMAL_ARGS = name##_init;                             \
  /* implement the dispatcher */                                               \
  static void name##_init FORMAL_ARGS {                                        \
    eCPUType cpuType = GetCpuType();                                           \
    if (cpu_##cpu1 == (cpu_##cpu1 & cpuType)) {                                \
      p##name = name##_##cpu1;                                                 \
    } else if (cpu_##cpu0 == (cpu_##cpu0 & cpuType)) {                         \
      p##name = name##_##cpu0;                                                 \
    } else {                                                                   \
      p##name = name##_c;                                                      \
    }                                                                          \
    if(ENABLE_ONLY_C_PATH) { \
      p##name = name##_c; \
    } \
    p##name ACTUAL_ARGS;                                                       \
  }                                                                            \
  /* implement the caller */                                                   \
  void name FORMAL_ARGS { p##name ACTUAL_ARGS; }

#if defined(_X86) || defined(_X64)

IMPL_PROC_2(load_4x4_windows_8u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS), ssse3, avx2)
IMPL_PROC_2(load_4x4_windows_16u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS), ssse3, avx2)

IMPL_PROC_0(sum_windows_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
#if NEW_SIMD_FUNC
IMPL_PROC_2(sum_windows_8x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), sse41, avx2)
IMPL_PROC_1(sum_windows_8x8_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_16x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_16x8_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_16x16_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)

IMPL_PROC_1(load_4x4_windows_10u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS), avx2)

IMPL_PROC_1(sum_windows_8x4_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_8x8_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_16x4_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_16x8_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
IMPL_PROC_1(sum_windows_16x16_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), avx2)
#endif
IMPL_PROC_1(sum_windows_12x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), sse41)
IMPL_PROC_0(sum_windows_int_16u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))

IMPL_PROC_0(sum_windows_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_2(sum_windows_8x4_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
           (SUM_WINDOWS_ACTUAL_ARGS), ssse3, avx2)
IMPL_PROC_2(sum_windows_12x4_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), ssse3, avx2)
IMPL_PROC_0(sum_windows_float_16u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))

#elif defined(_ARM) || defined(_ARM64)

IMPL_PROC_1(load_4x4_windows_8u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(load_4x4_windows_16u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS), neon)

IMPL_PROC_0(sum_windows_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_1(sum_windows_8x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
#if NEW_SIMD_FUNC
IMPL_PROC_1(sum_windows_8x8_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_16x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_16x8_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_16x16_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)

IMPL_PROC_1(load_4x4_windows_10u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS), neon)

IMPL_PROC_1(sum_windows_8x4_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_8x8_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_16x4_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_16x8_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_1(sum_windows_16x16_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS), neon)
#endif
IMPL_PROC_0(sum_windows_12x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_int_16u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))

IMPL_PROC_0(sum_windows_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_1(sum_windows_8x4_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
           (SUM_WINDOWS_ACTUAL_ARGS), neon)
IMPL_PROC_0(sum_windows_12x4_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_float_16u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))

#else

IMPL_PROC_0(load_4x4_windows_8u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(load_4x4_windows_16u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS))
#if NEW_10BIT_C_FUNC
IMPL_PROC_0(load_4x4_windows_10u, (LOAD_4x4_WINDOWS_FORMAL_ARGS),
            (LOAD_4x4_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_8x4_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_8x8_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_16x4_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_16x8_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_16x16_int_10u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
#endif

IMPL_PROC_0(sum_windows_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_8x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
#if NEW_SIMD_FUNC
IMPL_PROC_0(sum_windows_8x8_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_16x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_16x8_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_16x16_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
#endif
IMPL_PROC_0(sum_windows_12x4_int_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_int_16u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))

IMPL_PROC_0(sum_windows_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_8x4_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_12x4_float_8u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))
IMPL_PROC_0(sum_windows_float_16u, (SUM_WINDOWS_FORMAL_ARGS),
            (SUM_WINDOWS_ACTUAL_ARGS))

#endif
