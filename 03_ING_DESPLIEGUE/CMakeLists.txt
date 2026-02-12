# ─────────────────────────────────────────────────────────────────────────────
# TFM — Firmware de detección de objetos para ESP32-S3 (TinyML)
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.16)

# Los componentes externos (esp32-camera, esp-tflite-micro, esp-dl)
# se gestionan automáticamente via main/idf_component.yml (ESP Component Manager).

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tfm_tinyml_detector)
