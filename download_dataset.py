from huggingface_hub import snapshot_download
import os

def download_correct_dataset(dataset_name, local_dir):
    """下载正确结构的数据集"""
    
    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 使用 huggingface_hub 下载
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ 数据集下载成功: {local_dir}")
        
        # # 验证结构
        # if check_dataset_structure(local_dir):
        #     print("数据集结构正确")
        #     return True
        # else:
        #     print("数据集结构可能有问题")
        #     return False
            
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

# 下载示例
download_correct_dataset("unsloth/OpenMathReasoning-mini", "./correct_openmath")