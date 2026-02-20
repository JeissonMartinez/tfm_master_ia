# =============================================================================
# Capture_imagen — ESP-IDF Project for Dataset Image Capture
# Target: ESP32-S3 (Freenove CAM Board, WROOM N16R8 + OV5640)
# Purpose: Capture JPEG images to SD card via web interface
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# C++ standard (consistent with deployment firmware)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

# Extra component directories
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/components")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(capture_imagen)
