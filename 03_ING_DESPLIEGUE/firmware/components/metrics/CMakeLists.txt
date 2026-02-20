idf_component_register(
    SRCS "metrics.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_timer
    PRIV_REQUIRES driver postprocess
)
