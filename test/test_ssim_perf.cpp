/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_ssim.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

enum { WIDTH = 1920, HEIGHT = 1080, NUM_SRC_FRAMES = 16, TEST_ITER = 1 };

namespace {

template <typename data_t>
using cont_t =
    std::vector<std::unique_ptr<data_t, ssim::aligned_memory_deleter_t>>;

template <typename data_t>
std::tuple<cont_t<data_t>, ptrdiff_t>
Allocate(const uint32_t width, const uint32_t height, const size_t num) {
  cont_t<data_t> cont(num);

  const ptrdiff_t stride = (width * sizeof(data_t) + (ALIGN - 1)) & -ALIGN;
  const size_t size = stride * height;

  for (auto &item : cont) {
    item.reset((data_t *)ssim_alloc_aligned(size, LOG2_ALIGN));
  }

  return {std::move(cont), stride};

} // std::tuple<cont_t<data_t>, ptrdiff_t> Allocate(const uint32_t width,

template <typename data_t, typename proc_t> class Load4x4WindowsCaller {
public:
  Load4x4WindowsCaller(proc_t proc, const std::string desc)
      : proc_(proc), desc_(desc) {
    // allocate the source
    std::tie(ref_, refStride_) =
        Allocate<data_t>(WIDTH, HEIGHT, NUM_SRC_FRAMES);
    std::tie(cmp_, cmpStride_) =
        Allocate<data_t>(WIDTH, HEIGHT, NUM_SRC_FRAMES);

    // allocate the destination buffer
    std::tie(dst_, dstStride_) = Allocate<data_t>(WIDTH, 4, 1);
  }

  inline const std::string &Desc() const { return desc_; }

  size_t operator()() {
    SSIM_4X4_WINDOW_BUFFER buf = {dst_[0].get(), dstStride_};
    const uint32_t num4x4Windows = WIDTH / 4;

    for (size_t i = 0; i < NUM_SRC_FRAMES; ++i) {
      SSIM_SRC s = {ref_[i].get(), refStride_, cmp_[i].get(), cmpStride_};

      for (uint32_t y = 0; y + 4 < HEIGHT; y += 4) {
        proc_(&buf, num4x4Windows, &s);
        s.ref = (const uint8_t *)s.ref + 4 * refStride_;
        s.cmp = (const uint8_t *)s.cmp + 4 * cmpStride_;
      }
    }

    return NUM_SRC_FRAMES;
  }

private:
  const proc_t proc_;
  const std::string desc_;

  cont_t<data_t> ref_;
  ptrdiff_t refStride_ = 0;
  cont_t<data_t> cmp_;
  ptrdiff_t cmpStride_ = 0;
  cont_t<data_t> dst_;
  ptrdiff_t dstStride_ = 0;
}; // namespace

using timer_t = std::chrono::high_resolution_clock;
using time_t = std::chrono::duration<double, std::milli>;

template <typename caller_t> void TestFunction(caller_t &caller) {
  std::cout << caller.Desc() << std::endl;

  for (uint32_t i = 0; i < TEST_ITER; ++i) {
    std::chrono::duration<double, std::milli> totalTime(0);
    uint32_t totalIter = 0;

    do {
      const auto begin = timer_t::now();
      totalIter += caller();
      const auto end = timer_t::now();

      totalTime += (end - begin);

    } while (std::chrono::seconds(1) > totalTime);

    auto perf = totalTime / totalIter;
    std::cout << perf.count() << std::endl;
  }

} // void TestFunction(proc_t proc, const std::string &desc, caller_t caller)

template <typename data_t, typename proc_t>
inline void TestLoad4x4Window(proc_t proc, const std::string desc) {
  Load4x4WindowsCaller<data_t, proc_t> caller(proc, desc);

  TestFunction(caller);
}

} // namespace

int main(int argc, char **argv) {
  UNUSED(argc);
  UNUSED(argv);

  TestLoad4x4Window<uint8_t>(load_4x4_windows_8u_c, "load_4x4_windows_8u_c");
#if defined(_X86) || defined(_X64)
  if (ssim::CheckSIMD(cpu_ssse3)) {
    TestLoad4x4Window<uint8_t>(load_4x4_windows_8u_ssse3,
                               "load_4x4_windows_8u_ssse3");
  }
  if (ssim::CheckSIMD(cpu_avx2)) {
    TestLoad4x4Window<uint8_t>(load_4x4_windows_8u_avx2,
                               "load_4x4_windows_8u_avx2");
  }
#elif defined(_ARM) || defined(_ARM64)
  if (ssim::CheckSIMD(cpu_neon)) {
    TestLoad4x4Window<uint8_t>(load_4x4_windows_8u_neon,
                               "load_4x4_windows_8u_neon");
  }
#endif // defined(_X86) || defined(_X64)
  TestLoad4x4Window<uint8_t>(load_4x4_windows_8u, "load_4x4_windows_8u");

  std::cout << std::endl;

  TestLoad4x4Window<uint16_t>(load_4x4_windows_16u_c, "load_4x4_windows_16u_c");
#if defined(_X86) || defined(_X64)
  if (ssim::CheckSIMD(cpu_ssse3)) {
    TestLoad4x4Window<uint16_t>(load_4x4_windows_16u_ssse3,
                                "load_4x4_windows_16u_ssse3");
  }
  if (ssim::CheckSIMD(cpu_avx2)) {
    TestLoad4x4Window<uint16_t>(load_4x4_windows_16u_avx2,
                                "load_4x4_windows_16u_avx2");
  }
#elif defined(_ARM) || defined(_ARM64)
  if (ssim::CheckSIMD(cpu_neon)) {
    TestLoad4x4Window<uint16_t>(load_4x4_windows_16u_neon,
                                "load_4x4_windows_16u_neon");
  }
#endif // defined(_X86) || defined(_X64)
  TestLoad4x4Window<uint16_t>(load_4x4_windows_16u, "load_4x4_windows_16u");

} // int main(int argc, char** argv)
