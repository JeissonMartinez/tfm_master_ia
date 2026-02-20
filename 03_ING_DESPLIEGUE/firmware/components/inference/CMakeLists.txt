idf_component_register(
    SRCS "inference_engine.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_partition app_config
    PRIV_REQUIRES espressif__esp-dl
)
