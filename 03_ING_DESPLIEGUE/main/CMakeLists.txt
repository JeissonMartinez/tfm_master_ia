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
    # Modelos ESP-DL se cargan desde particiones flash independientes
    # (no EMBED_FILES).  Usar scripts/flash_models.sh para flashear.
)

# C++23
set_source_files_properties(${SOURCES}
    PROPERTIES COMPILE_FLAGS "-std=gnu++2b"
)
