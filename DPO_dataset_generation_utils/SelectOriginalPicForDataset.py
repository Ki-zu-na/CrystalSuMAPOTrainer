import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict
from imgutils.metrics import lpips_clustering
from PIL import Image
from imgutils.validate import is_monochrome

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

def process_artist_folder(artist_path: Path, output_path: Path, target_count: int = 40) -> bool:
    """处理单个艺术家文件夹并选择图片"""
    # 首先检查results.json是否存在
    if not (artist_path / "results.json").exists():
        print(f"Skipping {artist_path.name}: results.json not found")
        return False
        
    # 检查图片总数
    total_images = count_images_in_dirs(artist_path)
    if total_images < 20:
        print(f"Skipping {artist_path.name}: only {total_images} images found")
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
        os.makedirs(artist_output_dir, exist_ok=True)
        
        # 创建新的results.json数据
        selected_results = {}
        
        for img_name, _ in selected_images:
            # 复制图片
            src_path = find_image_path(artist_path, img_name)
            if src_path:
                shutil.copy2(src_path, artist_output_dir / img_name)
                # 保存对应的JSON数据
                selected_results[img_name] = results_json[img_name]
        
        # 保存新的results.json
        with open(artist_output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(selected_results, f, ensure_ascii=False, indent=4)
            
        return len(selected_images) == target_count
    
    return False

def find_image_path(artist_path: Path, image_name: str) -> Path:
    """在艺术家目录下查找图片文件"""
    search_dirs = ['2020s', '2022s', 'new', 'unknown', 'undefined']
    
    for dir_name in search_dirs:
        img_path = artist_path / dir_name / image_name
        if img_path.exists():
            return img_path
            
    return None

def main():
    # Set up paths
    dataset_path = Path(r"G:\SDXL_large_Modified")
    output_path = Path(r"G:\Dataset_selected_MAPO")
    failed_artists = []
    
    # Process each artist
    for artist_dir in dataset_path.iterdir():
        if artist_dir.is_dir():
            success = process_artist_folder(artist_dir, output_path)
            if not success:
                failed_artists.append(artist_dir.name)
    
    # Log failed artists
    if failed_artists:
        with open("failed_artists.txt", "w") as f:
            f.write("\n".join(failed_artists))

if __name__ == "__main__":
    main()
