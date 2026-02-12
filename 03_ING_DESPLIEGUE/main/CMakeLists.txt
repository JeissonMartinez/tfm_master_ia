# ─────────────────────────────────────────────────────────────────────────────
# TFM TinyML Detector — main component
# ─────────────────────────────────────────────────────────────────────────────

# Fuentes C++
set(SOURCES
    "main.cpp"
    "camera.cpp"
    "image_proc.cpp"
    "tflite_engine.cpp"
    "espdl_engine.cpp"
    "postprocess.cpp"
    "metrics.cpp"
    "wifi_manager.cpp"
    "web_server.cpp"
)

idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS "." "models/tflite"
    REQUIRES
        nvs_flash
        esp_wifi
        esp_event
        esp_netif
        esp_http_server
        esp_timer
        esp_psram
        driver
        esp32-camera
    PRIV_REQUIRES
        json
    # EMBED_FILES para modelos ESP-DL (descomentar cuando estén generados)
    # EMBED_FILES
    #     "models/espdl/mobilenetv2_ssdlite_v1.espdl"
    #     "models/espdl/yolo11n_v1.espdl"
    #     "models/espdl/yolo26n_v1.espdl"
)

# C++23
set_source_files_properties(${SOURCES}
    PROPERTIES COMPILE_FLAGS "-std=gnu++2b"
)
