from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id="BAAI/bge-large-zh-v1.5",
    cache_dir="/Users/v_changzhitao/Desktop/project/kg/Models"
)

# /Users/v_changzhitao/Desktop/project/kg/Models/BAAI/bge-large-zh-v1___5