/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "essim.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>

void print_help() {
  printf("\n");
  printf(" -r : Reference yuv file path \n");
  printf(" -d : Distorted yuv file path \n");
  printf(" -w : Width of yuv. Max Width <= 7680, to avoid huge precision loss "
         "in eSSIM score \n");
  printf(" -h : Height of yuv. Max Height <= 4320, to avoid huge precision "
         "loss in eSSIM score \n");
  printf(" -o : CSV file name, use - to print to stdout \n");
  printf(" -bd : Bit-depth. Default Bit-depth is 8 \n");
  printf(" -wsize : Window size for eSSIM (can be 8 or 16). Default is 8. \n");
  printf(" -wstride : Window stride for eSSIM (can be 4, 8, 16). Default is 4. "
         "\n");
  printf(" -mink : SSIM Minkowski pooling (can be 3 or 4). Default is 3. \n");
  printf(" -mode : Can be 0 -> SSIM_MODE_REF, 1 -> SSIM_MODE_PERF_INT, 2 -> "
         "SSIM_MODE_PERF_FLOAT. Default is 1 \n");
  printf("\n Example cmd : \t");
  printf(" -r /mnt/d/InpYuvPath/xyz.yuv -d /mnt/d/ReconYuvPath/abc.yuv -w 1280 "
         "-h 720 -bd 10 -wsize 16 -wstride 8 -mink 3\n");
  printf("\n");
}

int main(int argc, char **argv) {
  uint32_t i = 1;
  std::string InpYuvPath = "NULL", ReconYuvPath = "NULL";
  uint32_t Width = 0, Height = 0, WSize = 8, WStride = 4;
  uint32_t Mode = 1, BitDepth = 8, essim_mink_value = 3;
  eSSIMMode ModeEnum;

  std::string output;

  /*Read cmd line args*/
  if (argc <= 2) {
    print_help();
    return 0;
  } else {
    while ((i < (uint32_t)argc) && (strcmp(argv[i], "") != 0)) {
      if (strcmp(argv[i], "-r") == 0) {
        InpYuvPath = argv[++i];
        std::cerr << "Reference: " << InpYuvPath << std::endl;
      } else if (strcmp(argv[i], "-d") == 0) {
        ReconYuvPath = argv[++i];
        std::cerr << "Distorted: " << ReconYuvPath << std::endl;
      } else if (strcmp(argv[i], "-w") == 0) {
        Width = (uint32_t)atoi(argv[++i]);
        std::cerr << "Width: " << Width << std::endl;
      } else if (strcmp(argv[i], "-h") == 0) {
        Height = (uint32_t)atoi(argv[++i]);
        std::cerr << "Height: " << Height << std::endl;
      } else if (strcmp(argv[i], "-wsize") == 0) {
        WSize = (uint32_t)atoi(argv[++i]);
        std::cerr << "WSize: " << WSize << std::endl;
      } else if (strcmp(argv[i], "-wstride") == 0) {
        WStride = atoi(argv[++i]);
        std::cerr << "WStride: " << WStride << std::endl;
      } else if (strcmp(argv[i], "-mode") == 0) {
        Mode = atoi(argv[++i]);
        if (Mode != 0 && Mode != 1 && Mode != 2) {
          Mode = 1;
          std::cerr << "Invalid mode. Using default eSSIMMode i.e 1 "
                       "(SSIM_MODE_PERF_INT)"
                    << std::endl;
        }
      } else if (strcmp(argv[i], "-bd") == 0) {
        BitDepth = atoi(argv[++i]);
        std::cerr << "BitDepth: " << BitDepth << std::endl;
      } else if (strcmp(argv[i], "-mink") == 0) {
        essim_mink_value = atoi(argv[++i]);
        std::cerr << "ESSIM Minkowski Pooling: " << essim_mink_value
                  << std::endl;
      } else if (strcmp(argv[i], "-o") == 0) {
        output = argv[++i];
        std::cerr << "Output: " << output << std::endl;
      } else {
        std::cerr << "Unknow argument: " << argv[i] << std::endl;
        return 0;
      }
      i++;
    }
    if ((WSize != 8) && (WSize != 16)) {
      WSize = 16;
      std::cout << "Considering default WSize i.e 16" << std::endl;
    }
    if ((WStride > WSize) ||
        ((WStride != 4) && (WStride != 8) && (WStride != 16))) {
      WStride = WSize;
      std::cout << "Considering default WStride as WSize" << std::endl;
    }
  }

  std::cerr << std::endl;

  if (Mode == 0)
    ModeEnum = SSIM_MODE_REF;
  else if (Mode == 2)
    ModeEnum = SSIM_MODE_PERF_FLOAT;
  else
    ModeEnum = SSIM_MODE_PERF_INT;

  uint32_t BitDepthMinus8 = BitDepth - 8;

  std::fstream InpYuvfp, ReconYuvfp;

  InpYuvfp.open(InpYuvPath, std::ios::in | std::ios::binary);
  ReconYuvfp.open(ReconYuvPath, std::ios::in | std::ios::binary);

  if (!InpYuvfp) {
    std::cerr << "Input Yuv file is empty (or)"
              << "doesn't found in the specified path" << std::endl;
    InpYuvfp.close();
    return 0;
  }
  if (!ReconYuvfp) {
    std::cerr << "Recon Yuv file is empty (or)"
              << "doesn't found in the specified path" << std::endl;
    ReconYuvfp.close();
    InpYuvfp.close();
    return 0;
  }

  /*Get yuv name from recon path, so we can generate csv file,
    based on that name*/
  size_t pos = ReconYuvPath.find_last_of("/\\");
  std::string YuvFileName;

  if (pos != std::string::npos) {
      // Extract the substring after the last '/' or '\'
      YuvFileName = ReconYuvPath.substr(pos + 1);
  } else {
      YuvFileName = ReconYuvPath;
  }

  // Remove the last 4 characters (assuming it's an extension)
  if (YuvFileName.size() >= 4) {
      YuvFileName.resize(YuvFileName.size() - 4);
  }

  std::unique_ptr<std::ostream, void (*)(std::ostream *)> outStream(
      nullptr, [](std::ostream *) { /* do nothing */ });

  /*Creating o/p file pointers*/
  if (output == "-") {
    outStream.reset(&std::cout);
  } else {
    outStream.reset(new std::ofstream(output));
  }

  /*Writing to the file*/
  if (!output.empty())
    (*outStream) << "Frame, eSSIM, SSIM" << std::endl;

  /*Calculating total num of frames and Frame, Y-Plane & UV-Plane sizes*/
  InpYuvfp.seekg(0, std::ios::end);
  uint64_t InpYuvFileSize = InpYuvfp.tellg();
  InpYuvfp.seekg(std::ios::beg);

  ReconYuvfp.seekg(0, std::ios::end);
  uint64_t ReconYuvFileSize = ReconYuvfp.tellg();
  ReconYuvfp.seekg(std::ios::beg);

  uint64_t TotalNumOfFrames = 0, FileSize = 0, FrNum = 0;
  uint64_t FrSize = 0, YPlaneSize = 0, UVPlaneSize = 0;

  /*For 8 bit-depth DataTypeSize = sizeof(uint8_t)=1*/
  uint32_t DataTypeSize = 1;
  if (BitDepth > 8) {
    /*For > 8 bit-depth DataTypeSize = sizeof(uint16_t)=2*/
    DataTypeSize = 2;
  }

  FileSize = std::min(InpYuvFileSize, ReconYuvFileSize);
  if (InpYuvFileSize != ReconYuvFileSize) {
    std::cerr << "Inp yuv & Recon yuv file sizes are not same" << std::endl;
  }

  TotalNumOfFrames = (FileSize * 2) / (DataTypeSize * Width * Height * 3);
  FrSize = (DataTypeSize * Width * Height * 3) / 2;
  YPlaneSize = DataTypeSize * Width * Height;
  UVPlaneSize = FrSize - YPlaneSize;

  uint8_t *InpYuvBuff = NULL, *ReconYuvBuff = NULL;
  uint16_t *InpYuvBuffHbd = NULL, *ReconYuvBuffHbd = NULL;
  ptrdiff_t stride = (sizeof(uint8_t) * Width) & -6 /*LOG2_ALIGN*/;
  if (BitDepth == 8) {
    InpYuvBuff = new uint8_t[YPlaneSize];
    ReconYuvBuff = new uint8_t[YPlaneSize];
  } else {
    InpYuvBuffHbd = new uint16_t[YPlaneSize];
    ReconYuvBuffHbd = new uint16_t[YPlaneSize];
  }

  float SSIMScore, ESSIMScore;
  double TotalSSIMScore = 0.0, TotalESSIMScore = 0.0;

  for (FrNum = 0; FrNum < TotalNumOfFrames; FrNum++) {
    SSIMScore = 0.0;
    ESSIMScore = 0.0;

    if (BitDepth == 8) {
      memset(InpYuvBuff, 0, YPlaneSize);
      memset(ReconYuvBuff, 0, YPlaneSize);

      InpYuvfp.read((char *)InpYuvBuff, YPlaneSize);
      ReconYuvfp.read((char *)ReconYuvBuff, YPlaneSize);
      ssim_compute_8u(&SSIMScore, &ESSIMScore, InpYuvBuff, stride, ReconYuvBuff,
                      stride, Width, Height, WSize, WStride, 1, ModeEnum,
                      SSIM_SPATIAL_POOLING_BOTH, essim_mink_value);
    } else {
      memset(InpYuvBuffHbd, 0, YPlaneSize);
      memset(ReconYuvBuffHbd, 0, YPlaneSize);

      InpYuvfp.read((char *)InpYuvBuffHbd, YPlaneSize);
      ReconYuvfp.read((char *)ReconYuvBuffHbd, YPlaneSize);
      ssim_compute_16u(&SSIMScore, &ESSIMScore, InpYuvBuffHbd, stride,
                       ReconYuvBuffHbd, stride, Width, Height, BitDepthMinus8,
                       WSize, WStride, 1, ModeEnum, SSIM_SPATIAL_POOLING_BOTH,
                       essim_mink_value);
    }

    TotalESSIMScore += ESSIMScore;
    TotalSSIMScore += SSIMScore;

    InpYuvfp.seekg(UVPlaneSize, std::ios::cur);
    ReconYuvfp.seekg(UVPlaneSize, std::ios::cur);

    /*Writing to a csv file*/
    if (!output.empty()) {
      (*outStream) << FrNum << ",";
      (*outStream) << ESSIMScore << ",";
      (*outStream) << SSIMScore << std::endl;
    }
  }

  double finalSSIM = TotalSSIMScore / TotalNumOfFrames;
  double finalESSIM = TotalESSIMScore / TotalNumOfFrames;

  std::cerr << std::endl;
  std::cerr << "Average eSSIM: " << std::fixed << std::setprecision(4)
            << finalESSIM << std::endl;
  std::cerr << "Average SSIM: " << std::fixed << std::setprecision(4)
            << finalSSIM << std::endl;

  if (BitDepth == 8) {
    delete[] InpYuvBuff;
    delete[] ReconYuvBuff;
  } else {
    delete[] InpYuvBuffHbd;
    delete[] ReconYuvBuffHbd;
  }
  InpYuvfp.close();
  ReconYuvfp.close();
  return 0;
}