idf_component_register(
    SRCS "capture_server.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_wifi esp_http_server esp_netif esp_event app_config
             camera_handler sd_storage espressif__mdns
    PRIV_REQUIRES nvs_flash esp_timer
    EMBED_FILES "web/index.html"
)
