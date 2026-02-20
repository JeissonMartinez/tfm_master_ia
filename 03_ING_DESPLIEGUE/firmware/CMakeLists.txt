# =============================================================================
# TFM TinyML Deployment - ESP-IDF Project
# Target: ESP32-S3 (Freenove CAM Board, WROOM N16R8 + OV5640)
# Runtime: ESP-DL 3.x with .espdl quantized models
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# C++ standard: GNU++23 (validated in pilot project)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)

# Extra component directories
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/components")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tfm_tinyml_deploy)
