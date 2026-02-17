idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        app_config
        nvs_flash
        camera_handler
        image_proc
        inference
        postprocess
        metrics
        network
        dashboard
)
