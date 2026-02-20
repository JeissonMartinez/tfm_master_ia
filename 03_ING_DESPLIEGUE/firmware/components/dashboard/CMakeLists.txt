idf_component_register(
    SRCS "dashboard.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_http_server
    PRIV_REQUIRES network
    EMBED_FILES "web/index.html"
)
