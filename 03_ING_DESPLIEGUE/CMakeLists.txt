# ─────────────────────────────────────────────────────────────────────────────
# TFM — Firmware de detección de objetos para ESP32-S3 (TinyML)
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.16)

# Componentes extra (esp32-camera, esp-tflite-micro, esp-dl)
set(EXTRA_COMPONENT_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/components
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tfm_tinyml_detector)
