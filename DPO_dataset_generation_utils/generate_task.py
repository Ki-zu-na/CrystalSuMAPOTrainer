import os
import json
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

def read_image_dimensions(image_path):
    """读取图片尺寸"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"读取图片 {image_path} 失败: {str(e)}")
        return None

def generate_caption(artist_name, filename, results_json, features_threshold=0.5):
    """生成图片的caption，复用原有的generate_caption逻辑但去掉分辨率相关部分"""
    # 生成final_artist_tag
    final_artist_tag = artist_name.replace('_', ' ') + ', '
    
    # 生成final_character_tag
    final_character_tag = ''
    if 'character' in results_json and results_json['character']:
        final_character_tag = ', '.join(results_json['character'].keys()).replace('_', ' ') + ', '
    
    # 生成features_tag
    features_tag = set()
    if 'features' in results_json:
        features_tag.update(k.replace('_', ' ') for k, v in results_json['features'].items() if v > features_threshold)
    
    final_features_tag = ', '.join(sorted(features_tag))
    
    # 生成final_rating_tag
    final_rating_tag = ''
    if results_json.get('is_AI'):
        final_rating_tag += 'ai-generated, '
    
    rating = results_json.get('rating', {})
    if rating:
        max_rating = max(rating, key=rating.get)
        if max_rating == 'general':
            max_rating = 'safe'
        elif max_rating == 'questionable':
            max_rating = 'nsfw'
        final_rating_tag += max_rating + ', '
    
    # 生成additional_tags
    additional_tags = ''
    if 'additional_tags' in results_json:
        additional_tags = results_json['additional_tags'].replace('_', ' ')
    
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

def process_dataset(dataset_path: str, artist_limit: int, output_json: str):
    """处理数据集并生成JSON文件"""
    dataset_path = Path(dataset_path)
    result_data = {}
    processed_count = 0
    
    print(f"开始处理数据集，将处理 {artist_limit} 个艺术家的数据...")
    
    # 获取所有艺术家目录
    artist_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for artist_dir in tqdm(artist_dirs, desc="处理艺术家"):
        # 检查是否已处理
        if (artist_dir / "process.mark").exists():
            continue
            
        # 检查必要文件夹和文件是否存在
        orig_pic_dir = artist_dir / "OriginalPic"
        results_json_path = artist_dir / "results.json"
        
        if not (orig_pic_dir.exists() and results_json_path.exists()):
            continue
            
        try:
            # 读取results.json
            with open(results_json_path, 'r', encoding='utf-8') as f:
                results_json = json.load(f)
                
            artist_data = {}
            
            # 处理每张图片
            for img_path in orig_pic_dir.glob('*'):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                    
                img_name = img_path.name
                if img_name not in results_json:
                    continue
                    
                # 读取图片尺寸
                dimensions = read_image_dimensions(img_path)
                if dimensions is None:
                    continue
                    
                width, height = dimensions
                
                # 生成caption
                tag = generate_caption(artist_dir.name, img_name, results_json[img_name])
                
                # 保存数据
                artist_data[img_name] = {
                    "tag": tag,
                    "width": width,
                    "height": height
                }
            
            if artist_data:
                result_data[artist_dir.name] = artist_data
                
                # 创建处理标记
                with open(artist_dir / "process.mark", 'w') as f:
                    f.write(str(datetime.now()))
                
                processed_count += 1
                
                # 保存当前进度
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                if processed_count >= artist_limit:
                    break
                    
        except Exception as e:
            print(f"处理艺术家 {artist_dir.name} 时出错: {str(e)}")
            continue
    
    print(f"处理完成，共处理了 {processed_count} 个艺术家的数据")
    return result_data

def main():
    parser = argparse.ArgumentParser(description='处理数据集并生成JSON文件')
    parser.add_argument('dataset_path', type=str, help='数据集根目录路径')
    parser.add_argument('artist_limit', type=int, help='要处理的艺术家数量')
    parser.add_argument('--output', type=str, default='dataset_tags.json', help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_path, args.artist_limit, args.output)

if __name__ == "__main__":
    from datetime import datetime
    main()