idf_component_register(
    SRCS "camera_handler.cpp"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_timer app_config espressif__esp32-camera
)
