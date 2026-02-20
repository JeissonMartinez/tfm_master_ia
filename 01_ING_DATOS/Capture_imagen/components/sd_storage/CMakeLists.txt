idf_component_register(
    SRCS "sd_storage.cpp"
    INCLUDE_DIRS "include"
    REQUIRES app_config nvs_flash fatfs vfs
    PRIV_REQUIRES sdmmc esp_driver_sdmmc
)
