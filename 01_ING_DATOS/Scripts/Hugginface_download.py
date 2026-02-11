from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ayanzadeh93/obstacle_avoidance_BLV",
    repo_type="dataset",
    local_dir="/Users/admin/Documents/TFM_UNIR/01_ING_DATOS/Datasets_options/Huggingface_Dataset12",
    allow_patterns="images/*"
)