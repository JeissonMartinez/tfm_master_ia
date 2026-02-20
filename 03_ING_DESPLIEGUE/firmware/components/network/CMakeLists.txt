idf_component_register(
    SRCS "network.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_wifi esp_http_server esp_netif esp_event app_config metrics postprocess
    PRIV_REQUIRES nvs_flash image_proc esp_timer
)
