import os
import json
import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import io
import re
from datetime import datetime

def read_image_to_bytes(image_path, target_size=None):
    """读取图片并转换为bytes，可选择调整分辨率
    
    Args:
        image_path: 图片路径
        target_size: 目标分辨率 (width, height)，如果提供则调整图片大小
    """
    img = Image.open(image_path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 将图片转换为bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    return img_byte_arr.getvalue()

def get_year_from_weibo_date(date_str):
    """从微博日期字符串中获取年份"""
    try:
        return int(date_str.split('-')[0])
    except:
        return None

def get_year_from_date(date_str):
    """从普通日期字符串中获取年份"""
    try:
        return int(date_str.split('-')[0])
    except:
        return None

def generate_caption(artist_folder, filename, results_json, src_img_path, features_threshold=0.5):
    """生成图片的caption"""
    # 生成final_artist_tag
    final_artist_tag = os.path.basename(artist_folder).replace('_', ' ') + ', '
    
    # 生成final_character_tag
    final_character_tag = ''
    if results_json and filename in results_json and 'character' in results_json[filename] and results_json[filename]['character']:
        final_character_tag = ', '.join(results_json[filename]['character'].keys()).replace('_', ' ') + ', '
    
    # 生成features_tag
    features_tag = set()
    if results_json and filename in results_json and 'features' in results_json[filename]:
        features_tag.update(k.replace('_', ' ') for k, v in results_json[filename]['features'].items() if v > features_threshold)
    
    final_features_tag = ', '.join(sorted(features_tag))
    
    # 生成final_rating_tag
    final_rating_tag = ''
    if results_json and filename in results_json:
        if results_json[filename].get('is_AI'):
            final_rating_tag += 'ai-generated, '
        
        scores_by_class = results_json[filename].get('scores_by_class', {})
        if scores_by_class:
            max_class = max(scores_by_class, key=scores_by_class.get)
            if max_class != 'masterpiece':
                final_rating_tag += f'{max_class} quality, '
            elif max_class == 'masterpiece':
                final_rating_tag += 'masterpiece, '
        
        rating = results_json[filename].get('rating', {})
        if rating:
            max_rating = max(rating, key=rating.get)
            if max_rating == 'general':
                max_rating = 'safe'
            elif max_rating == 'questionable':
                max_rating = 'nsfw'
            final_rating_tag += max_rating + ', '
        
        # 获取图片分辨率
        with Image.open(src_img_path) as img:
            width, height = img.size
            if width * height <= 589824:
                final_rating_tag += 'lowres, '
            elif width * height >= 1638400:
                final_rating_tag += 'absurdres, '
    
    # 生成additional_tags
    additional_tags = ''
    if results_json and filename in results_json and 'additional_tags' in results_json[filename]:
        additional_tags = results_json[filename]['additional_tags'].replace('_', ' ')
    
    # 组合最终的caption
    prefix_tags = filter(bool, [
        final_artist_tag.strip(', '),
        final_character_tag.strip(', ')
    ])
    prefix = ", ".join(prefix_tags)

    suffix_tags = filter(bool, [
        final_features_tag,
        final_rating_tag.strip(', '),
        additional_tags.strip(', ')
    ])
    suffix = ", ".join(suffix_tags)

    return f"{prefix}, |||{suffix}"

def generate_parquet(base_dir, output_path, chunk_size=300*1024*1024):
    """生成parquet文件
    
    Args:
        base_dir: 包含多个artist文件夹的基础目录
        output_path: 输出parquet文件的路径
        chunk_size: 每个parquet文件的最大大小(bytes)
    """
    data = {
        'jpg_0': [],
        'jpg_1': [], 
        'caption': [],
        'label_0': [],
        'label_1': []
    }
    
    current_size = 0
    file_count = 0
    
    # 首先计算总数据大小来估算chunks数量
    total_size = 0
    for artist in os.listdir(base_dir):
        artist_dir = os.path.join(base_dir, artist)
        if not os.path.isdir(artist_dir):
            continue
            
        orig_dir = os.path.join(artist_dir, 'OriginalPic')
        if not os.path.exists(orig_dir):
            continue
            
        for img_name in os.listdir(orig_dir):
            if not img_name.endswith(('.jpg', '.png')):
                continue
                
            img_path = os.path.join(orig_dir, img_name)
            total_size += os.path.getsize(img_path) * 4  # 原图 + 2个DPO图片 + 额外开销
            
    estimated_chunks = max(1, total_size // chunk_size + 1)
    
    # 遍历artist目录
    for artist in os.listdir(base_dir):
        artist_dir = os.path.join(base_dir, artist)
        if not os.path.isdir(artist_dir):
            continue
            
        # 读取results.json
        results_path = os.path.join(artist_dir, 'results.json')
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found, skipping {artist}")
            continue
            
        with open(results_path, 'r', encoding='utf-8') as f:
            results_json = json.load(f)
            
        orig_dir = os.path.join(artist_dir, 'OriginalPic')
        dpo_dir = os.path.join(artist_dir, 'DPO_generated')
        
        if not os.path.exists(orig_dir) or not os.path.exists(dpo_dir):
            print(f"Warning: Missing OriginalPic or DPO_generated folder in {artist}")
            continue
            
        # 处理每张原始图片
        for img_name in os.listdir(orig_dir):
            if not img_name.endswith(('.jpg', '.png')):
                continue
                
            if img_name not in results_json:
                print(f"Warning: {img_name} not found in results.json")
                continue
                
            # 查找对应的DPO生成图片
            base_name = os.path.splitext(img_name)[0]
            dpo_images = []
            for i in range(1, 5):
                dpo_path = os.path.join(dpo_dir, f"{base_name}_DPO{i}.png")
                if os.path.exists(dpo_path):
                    dpo_images.append(dpo_path)
                    
            if len(dpo_images) < 2:
                print(f"Warning: Not enough DPO images for {img_name}")
                continue
            
            # 获取DPO图片的分辨率
            with Image.open(dpo_images[0]) as dpo_img:
                target_size = dpo_img.size
            
            # 读取并调整原图大小
            orig_img_path = os.path.join(orig_dir, img_name)
            jpg_0 = read_image_to_bytes(orig_img_path, target_size)
            
            # 生成caption
            caption = generate_caption(artist_dir, img_name, results_json, orig_img_path)
            
            # 随机选择2个DPO图片
            selected_dpo = random.sample(dpo_images, 2)
            
            # 添加数据
            for dpo_path in selected_dpo:
                jpg_1 = read_image_to_bytes(dpo_path)  # DPO图片不需要调整大小
                data['jpg_0'].append(jpg_0)
                data['jpg_1'].append(jpg_1)
                data['caption'].append(caption)
                data['label_0'].append(1)
                data['label_1'].append(0)
                
                current_size += len(jpg_0) + len(jpg_1)
                
                # 检查是否需要保存当前chunk
                if current_size >= chunk_size:
                    save_parquet(data, output_path, file_count, estimated_chunks)
                    data = {k: [] for k in data}
                    current_size = 0
                    file_count += 1
    
    # 保存最后的数据
    if any(len(v) > 0 for v in data.values()):
        save_parquet(data, output_path, file_count, estimated_chunks)

def save_parquet(data, output_path, file_count, total_chunks):
    """保存数据为parquet文件
    
    Args:
        data: 要保存的数据字典
        output_path: 输出路径
        file_count: 当前文件编号
        total_chunks: 总文件数(用于格式化文件名)
    """
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    
    # 格式化文件名: train-{5位数字}-of-{6位数字}
    output_file = f"train-{file_count:05d}-of-{total_chunks:06d}.parquet"
    pq.write_table(table, output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    base_dir = "path/to/base/dir"  # 修改为实际路径
    output_path = "path/to/output/dpo_dataset"  # 修改为实际输出路径
    generate_parquet(base_dir, output_path)