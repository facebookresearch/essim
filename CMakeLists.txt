cmake_minimum_required(VERSION 3.1)
project(essim VERSION 0.0.1 DESCRIPTION "efficiently compute enhanced SSIM (eSSIM)")
set(CMAKE_CXX_STANDARD 17)
include(GNUInstallDirs)

option(BUILD_TESTS "build test code" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(ClangFormat)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}" "-checks=-*,modernize-*")

set(HEADER_FILES essim/essim.h)

add_library(essim
  "essim/src/ssim_memory.c"
  "essim/src/ssim_internal.disp.c"
  "essim/src/ssim_ssim.c"
  "essim/src/ssim_perf.c"
  "essim/src/ssim_cpu.c"
  "essim/src/ssim_internal.c"
  "essim/src/ssim_prec.c"
  "essim/src/ssim_internal.arm.neon.c"
  "essim/src/ssim_internal.x86.avx2.c"
  "essim/src/ssim_internal.x86.sse41.c"
  "essim/src/ssim_internal.x86.ssse3.c"
  )
target_clangformat_setup(essim)
set_target_properties(essim PROPERTIES PUBLIC_HEADER ${HEADER_FILES})
set_target_properties(essim PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")

target_include_directories(essim PRIVATE
  ${CMAKE_SOURCE_DIR}
  )

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set_target_properties(essim PROPERTIES COMPILE_FLAGS "-fexceptions -Wall -Wpedantic -O3")
else()
  set_property(SOURCE "essim/src/ssim_internal.x86.sse41.c" "essim/src/ssim_internal.x86.ssse3.c"  PROPERTY COMPILE_FLAGS "-mssse3 -msse4.1")
  set_property(SOURCE "essim/src/ssim_internal.x86.avx2.c" PROPERTY COMPILE_FLAGS "-mavx2")
  set_target_properties(essim PROPERTIES COMPILE_FLAGS "-fexceptions -fpermissive -Wall -Wpedantic -mxsave -O3")
endif()

if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  enable_testing()
  add_executable(tests
    test/test_ssim_ssim.cpp
    test/test_ssim_memory.cpp
    test/test_ssim_internal.cpp
    test/test_ssim.h
    essim/essim.h)
  target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(tests essim gtest_main)
  gtest_discover_tests(tests)
endif()

configure_file(essim.pc.in essim.pc @ONLY)

# build settings :  make release, make debug

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Creating the executable in the debug mode.")

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Creating the executable in the release mode.")

install(TARGETS essim
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_BINARY_DIR}/essim.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
