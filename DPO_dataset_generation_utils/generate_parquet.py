import os
import json
import random
from PIL import Image
from datasets import Dataset, Features, Image as ImageFeature, Value

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

def generate_dataset(base_dir):
    """生成数据集字典列表"""
    data = {
        'jpg_0': [],
        'jpg_1': [], 
        'caption': [],
        'label_0': [],
        'label_1': []
    }
    
    # ... existing code ...
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
            
            # 获取原图路径
            orig_img_path = os.path.join(orig_dir, img_name)
            
            # 生成caption
            caption = generate_caption(artist_dir, img_name, results_json, orig_img_path)
            
            # 随机选择2个DPO图片
            selected_dpo = random.sample(dpo_images, 2)
            
            # 添加数据
            for dpo_path in selected_dpo:
                data['jpg_0'].append(orig_img_path)
                data['jpg_1'].append(dpo_path)
                data['caption'].append(caption)
                data['label_0'].append(1)
                data['label_1'].append(0)
    
    return data

def save_dataset(data, output_path, num_shards=10):
    """保存数据集为parquet文件"""
    # 定义数据集特征
    features = Features({
        'jpg_0': ImageFeature(),
        'jpg_1': ImageFeature(),
        'caption': Value('string'),
        'label_0': Value('int64'),
        'label_1': Value('int64')
    })
    
    # 创建Dataset对象
    dataset = Dataset.from_dict(data, features=features)
    
    # 保存为parquet文件，自动分片
    dataset.save_to_disk(
        output_path,
        num_shards=num_shards,
        max_shard_size="500MB"  # 可选：设置每个分片的最大大小
    )
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    base_dir = "path/to/base/dir"  # 修改为实际路径
    output_path = "path/to/output/dpo_dataset"  # 修改为实际输出路径
    
    # 生成数据集
    data = generate_dataset(base_dir)
    
    # 保存数据集
    save_dataset(data, output_path)