import os
import io
from PIL import Image
import pandas as pd

def extract_parquet_data(parquet_path, output_dir):
    """
    从parquet文件提取数据并按指定格式保存
    
    Args:
        parquet_path: parquet文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取parquet文件
    df = pd.read_parquet(parquet_path)
    
    # 遍历每一行数据
    for idx, row in enumerate(df.itertuples(), 1):
        # 创建图片对象
        img0 = Image.open(io.BytesIO(row.jpg_0)).convert('RGB')
        img1 = Image.open(io.BytesIO(row.jpg_1)).convert('RGB')
        
        # 根据label_0决定正负样本
        if row.label_0 == 1:
            positive_img = img0
            negative_img = img1
        else:
            positive_img = img1
            negative_img = img0
            
        # 保存图片
        positive_path = os.path.join(output_dir, f"{idx}_positive.jpg")
        negative_path = os.path.join(output_dir, f"{idx}_negative.jpg")
        positive_img.save(positive_path)
        negative_img.save(negative_path)
        
        # 保存caption
        caption_path = os.path.join(output_dir, f"{idx}.txt")
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(row.caption)
        
        # 打印进度
        if idx % 100 == 0:
            print(f"已处理 {idx} 个样本")

if __name__ == "__main__":
    # 使用示例
    parquet_path = r"C:\Users\Crystal427\Downloads\pickascore\validation_unique-00000-of-00001-33ead111845fc9c4.parquet"  # 替换为你的parquet文件路径
    output_dir = r"C:\Users\Crystal427\Downloads\pickascore\extracted_data"  # 替换为你想要的输出目录
    
    extract_parquet_data(parquet_path, output_dir)
    print("提取完成！")