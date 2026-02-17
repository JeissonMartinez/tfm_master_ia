idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        app_config
        nvs_flash
        camera_handler
        sd_storage
        capture_server
)
