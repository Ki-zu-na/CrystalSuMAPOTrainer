import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict
from imgutils.metrics import lpips_clustering
from PIL import Image
from imgutils.validate import is_monochrome
import argparse

def load_json_data(json_path: str) -> Dict:
    """Load and parse results.json file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_bad_image(image_scores, real_threshold=0.5, aesthetic_threshold=4.0):
    """
    判断图片是否为不合格图片
    
    Args:
        image_scores: 图片的各项评分
        real_threshold: 真实照片分数阈值
        aesthetic_threshold: 美学分数阈值
    """
    imgscore = image_scores["imgscore"]
    anime_real_score = image_scores["anime_real_score"]
    aesthetic_score = image_scores["aesthetic_score"]
    features = image_scores["features"]
    
    bad_imgscore_types = ["not_painting", "3d"]
    
    # 检查是否为单色图片 (monochrome值大于0.8)
    is_mono = features.get("monochrome", 0) > 0.8
    
    return (
        max(imgscore, key=imgscore.get) in bad_imgscore_types or
        anime_real_score["real"] > real_threshold or
        aesthetic_score < aesthetic_threshold or
        is_mono
    )
    
def count_images_in_dirs(artist_path: Path) -> int:
    """统计指定目录下的图片总数"""
    search_dirs = ['2020s', '2022s', 'new', 'unknown', 'undefined']
    total_images = 0
    
    for dir_name in search_dirs:
        dir_path = artist_path / dir_name
        if dir_path.exists():
            # 统计常见图片格式文件
            total_images += len(list(dir_path.glob('*.jpg')))
            total_images += len(list(dir_path.glob('*.png')))
            total_images += len(list(dir_path.glob('*.jpeg')))
            total_images += len(list(dir_path.glob('*.webp')))
    return total_images

def find_image_path(artist_path: Path, image_name: str) -> Path:
    """Find the actual path of an image in the artist's directory structure"""
    # Define allowed image extensions
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    
    # Check if the image has allowed extension
    if not any(image_name.lower().endswith(ext) for ext in allowed_extensions):
        return None
        
    search_dirs = ['2020s', '2022s', 'new', 'unknown', 'undefined']
    for dir_name in search_dirs:
        img_path = artist_path / dir_name / image_name
        if img_path.exists():
            return img_path
    return None

def process_artist_folder(artist_path: Path, output_path: Path, target_count: int = 40, max_retry: int = 10) -> bool:
    """Process single artist folder and select images"""
    # Check if results.json exists
    if not (artist_path / "results.json").exists():
        print(f"Skipping {artist_path.name}: results.json not found")
        return False
    
    # Load results.json
    results_json = load_json_data(artist_path / "results.json")
    
    # Separate good and bad images
    good_images = []
    bad_images = []
    
    for img_name, scores in results_json.items():
        if not is_bad_image(scores, 0.5, 4.0):
            good_images.append((img_name, scores['aesthetic_score']))
        else:
            bad_images.append((img_name, scores))
    
    # Sort bad images by aesthetic score for potential later use
    bad_images.sort(key=lambda x: x[1]['aesthetic_score'], reverse=True)
    
    selected_images = []
    
    # Try to get initial 40 images from good images
    if len(good_images) >= target_count:
        candidates = random.sample(good_images, target_count)
        
        while len(selected_images) < target_count:
            # 获取候选图片路径并过滤掉None值
            candidate_paths = [find_image_path(artist_path, img[0]) for img in candidates]
            valid_paths = [(path, img) for path, img in zip(candidate_paths, candidates) if path is not None]
            
            if not valid_paths:  # 如果没有有效路径，跳出循环
                break
                
            # 分离路径和原始候选数据
            paths_for_clustering = [p[0] for p in valid_paths]
            valid_candidates = [p[1] for p in valid_paths]
            
            # 运行聚类
            clusters = lpips_clustering(paths_for_clustering)
            
            # 处理聚类结果
            used_clusters = set()
            for i, cluster in enumerate(clusters):
                if cluster == -1 or cluster not in used_clusters:
                    selected_images.append(valid_candidates[i])
                    if cluster != -1:
                        used_clusters.add(cluster)
            
            # If we need more images, get new candidates from good_images, excluding already selected ones
            if len(selected_images) < target_count:
                remaining_count = target_count - len(selected_images)
                new_candidates = random.sample([img for img in good_images 
                                             if img not in selected_images], 
                                             min(100, remaining_count))
                candidates = selected_images + new_candidates
    
    # 如果选中的图片少于30张，从bad_images中补充
    if len(selected_images) < 30:
        # 筛选符合条件的bad_images
        filtered_bad_images = []
        for img_name, scores in bad_images:
            features = scores["features"]
            imgscore = scores["imgscore"]
            
            # 检查条件：illustration分数最高且monochrome条件符合要求
            is_illustration_highest = max(imgscore, key=imgscore.get) == "illustration"
            mono_score = features.get("monochrome", 0)
            meets_mono_condition = "monochrome" not in features or mono_score < 0.4
            
            if is_illustration_highest and meets_mono_condition:
                filtered_bad_images.append((img_name, scores["aesthetic_score"]))
        
        # 按aesthetic_score排序
        filtered_bad_images.sort(key=lambda x: x[1], reverse=True)
        
        # 补充需要的数量
        remaining_count = target_count - len(selected_images)
        selected_images.extend(filtered_bad_images[:remaining_count])

    # Copy selected images and create new results.json
    if len(selected_images) > 0:
        artist_output_dir = output_path / artist_path.name
        original_pic_dir = artist_output_dir / "OriginalPic"
        os.makedirs(original_pic_dir, exist_ok=True)
        
        # 创建新的results.json数据
        selected_results = {}
        
        for img_name, _ in selected_images:
            # 复制图片到 OriginalPic 目录
            src_path = find_image_path(artist_path, img_name)
            if src_path:
                shutil.copy2(src_path, original_pic_dir / img_name)
                # 保存对应的JSON数据
                selected_results[img_name] = results_json[img_name]
        
        # 保存新的results.json到artist根目录
        with open(artist_output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(selected_results, f, ensure_ascii=False, indent=4)
            
        return len(selected_images) == target_count
    
    return False

def fix_missing_results(dataset_path: Path, source_dataset_path: Path):
    """修复数据集中缺失的 results.json 条目"""
    print("开始修复缺失的 results.json 条目...")
    
    for artist_dir in dataset_path.iterdir():
        if not artist_dir.is_dir():
            continue
            
        results_path = artist_dir / "results.json"
        original_pic_dir = artist_dir / "OriginalPic"
        
        if not results_path.exists() or not original_pic_dir.exists():
            print(f"跳过 {artist_dir.name}: 缺少必要文件")
            continue
            
        # 读取当前的 results.json
        with open(results_path, 'r', encoding='utf-8') as f:
            current_results = json.load(f)
            
        # 获取源数据集中对应艺术家的 results.json
        source_artist_dir = source_dataset_path / artist_dir.name
        source_results_path = source_artist_dir / "results.json"
        
        if not source_results_path.exists():
            print(f"跳过 {artist_dir.name}: 源数据集中未找到 results.json")
            continue
            
        with open(source_results_path, 'r', encoding='utf-8') as f:
            source_results = json.load(f)
            
        # 检查每个图片是否有对应的 results 条目
        modified = False
        for img_path in original_pic_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']:
                img_name = img_path.name
                
                if img_name not in current_results and img_name in source_results:
                    print(f"为 {artist_dir.name}/{img_name} 添加缺失的 results 条目")
                    current_results[img_name] = source_results[img_name]
                    modified = True
                    
        # 如果有修改，保存更新后的 results.json
        if modified:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=4)
            print(f"已更新 {artist_dir.name} 的 results.json")

def main():
    parser = argparse.ArgumentParser(description='处理数据集图片选择和标签修复')
    parser.add_argument('--mode', choices=['select', 'fixtagger'], required=True,
                      help='运行模式：select-选择图片，fixtagger-修复标签')
    
    args = parser.parse_args()
    
    # 直接指定路径
    dataset_path = Path(r"G:\Dataset_selected_MAPO")
    source_dataset_path = Path(r"G:\SDXL_large_Modified")
    output_path = Path(r"G:\Dataset_selected_MAPO")
    
    if args.mode == 'select':
        failed_artists = []
        for artist_dir in source_dataset_path.iterdir():
            if artist_dir.is_dir():
                success = process_artist_folder(artist_dir, output_path)
                if not success:
                    failed_artists.append(artist_dir.name)
        
        if failed_artists:
            with open("failed_artists.txt", "w") as f:
                f.write("\n".join(failed_artists))
                
    elif args.mode == 'fixtagger':
        fix_missing_results(dataset_path, source_dataset_path)

if __name__ == "__main__":
    main()
