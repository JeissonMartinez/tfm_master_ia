idf_component_register(
    SRCS
        "postprocess_common.cpp"
        "postprocess_ssd.cpp"
        "postprocess_yolo.cpp"
    INCLUDE_DIRS "include"
    REQUIRES inference app_config
)
