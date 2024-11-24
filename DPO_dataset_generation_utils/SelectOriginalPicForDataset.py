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

def find_image_path(artist_path: Path, image_name: str) -> Path:
    """
    在艺术家目录的各个子文件夹中查找图片
    
    Args:
        artist_path: 艺术家目录路径
        image_name: 图片文件名（可能包含hash部分）
    """
    # 只在这四个文件夹中查找
    image_folders = ['2022s', 'new', 'unknown', 'undefined']
    
    # 从文件名中提取danbooru ID部分
    if image_name.startswith('danbooru_'):
        try:
            # 提取ID部分 (danbooru_7149401_...)
            danbooru_id = image_name.split('_')[1]
            # 构建搜索模式
            search_pattern = f"danbooru_{danbooru_id}_*"
            
            # 在指定文件夹中搜索匹配的文件
            for folder in image_folders:
                folder_path = artist_path / folder
                if folder_path.exists():
                    matches = list(folder_path.glob(search_pattern))
                    if matches:
                        return matches[0]  # 返回第一个匹配的文件
        except Exception as e:
            print(f"处理文件名时出错: {image_name}, 错误: {str(e)}")
    
    # 如果上述方法失败，尝试直接匹配完整文件名
    for folder in image_folders:
        folder_path = artist_path / folder
        if folder_path.exists():
            image_path = folder_path / image_name
            if image_path.exists():
                return image_path
                
    return None

def get_image_dimensions(image_path: Path) -> tuple:
    """
    获取图片的尺寸
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height)
    except Exception as e:
        print(f"读取图片 {image_path} 尺寸时出错: {str(e)}")
        return (0, 0)

def is_bad_image(image_scores, image_path: Path, real_threshold=0.5, aesthetic_threshold=4.0, min_side_length=512):
    """判断图片是否为不合格图片"""
    # 检查文件名是否以 'arg' 开头
    if image_path.stem.lower().startswith('arg'):
        return True
        
    imgscore = image_scores["imgscore"]
    anime_real_score = image_scores["anime_real_score"]
    aesthetic_score = image_scores["aesthetic_score"]
    features = image_scores.get("features", {})
    
    width, height = get_image_dimensions(image_path)
    min_side = min(width, height)
    
    bad_imgscore_types = ["not_painting", "3d"]
    is_mono = features.get("monochrome", 0) > 0.8
    comic_score = imgscore.get("comic", 0)
    is_comic_high = comic_score > 0.4
    
    # 检查 multiple_views 特征
    multiple_views_score = features.get("multiple_views", 0)
    is_multiple_views_high = multiple_views_score > 0.4
    
    return (
        max(imgscore, key=imgscore.get) in bad_imgscore_types or
        anime_real_score["real"] > real_threshold or
        aesthetic_score < aesthetic_threshold or
        is_mono or
        is_comic_high or
        is_multiple_views_high or  # 添加 multiple_views 检查
        min_side < min_side_length
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

def process_artist_folder(artist_path: Path, output_path: Path, target_count: int = 40, max_retry: int = 10, min_side_length: int = 512) -> bool:
    """Process single artist folder and select images"""
    if (artist_path / "no.dpo").exists():
        print(f"跳过 {artist_path.name}: 发现 no.dpo 文件")
        return False
        
    if not (artist_path / "results.json").exists():
        print(f"跳过 {artist_path.name}: results.json 未找到")
        return False
    
    results_json = load_json_data(artist_path / "results.json")
    print(f"\n处理艺术家文件夹: {artist_path.name}")
    
    # 1. 首先筛选分辨率符合要求的图片
    resolution_filtered = []
    for img_name, scores in results_json.items():
        img_path = find_image_path(artist_path, img_name)
        if img_path is None:
            print(f"找不到图片: {img_name}")
            continue
            
        width, height = get_image_dimensions(img_path)
        if min(width, height) >= min_side_length:
            resolution_filtered.append((img_name, scores, img_path))
    
    print(f"分辨率符合要求的图片数量: {len(resolution_filtered)}")
    
    # 2. 剔除特征不符合要求的图片
    feature_filtered = []
    for img_name, scores, img_path in resolution_filtered:
        features = scores.get("features", {})
        imgscore = scores["imgscore"]
        
        # 检查特征值
        is_mono = features.get("monochrome", 0) > 0.8
        comic_score = imgscore.get("comic", 0) > 0.4
        multiple_views = features.get("multiple_views", 0) > 0.4
        
        if not (is_mono or comic_score or multiple_views):
            feature_filtered.append((img_name, scores, img_path))
    
    print(f"特征符合要求的图片数量: {len(feature_filtered)}")
    
    # 3. 对剩余图片进行LPIPS聚类
    if feature_filtered:
        retry_count = 0
        current_candidates = feature_filtered.copy()
        selected_images = []
        
        while retry_count < max_retry:
            # 如果当前候选数量不足target_count，直接使用所有候选
            if len(current_candidates) <= target_count:
                paths_for_clustering = [str(p[2]) for p in current_candidates]
                try:
                    clusters = lpips_clustering(paths_for_clustering)
                    
                    # 处理聚类结果：每个聚类只保留一张图片
                    used_clusters = set()  # 使用set而不是dict
                    final_candidates = []
                    
                    # 处理非噪声点和噪声点
                    for i, cluster in enumerate(clusters):
                        if cluster == -1 or cluster not in used_clusters:
                            final_candidates.append(current_candidates[i])
                            if cluster != -1:
                                used_clusters.add(cluster)
                    
                    selected_images = final_candidates
                    break
                    
                except Exception as e:
                    print(f"聚类过程发生错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # 随机选择target_count张图片进行聚类
            current_batch = random.sample(current_candidates, target_count)
            paths_for_clustering = [str(p[2]) for p in current_batch]
            
            try:
                clusters = lpips_clustering(paths_for_clustering)
                
                # 处理聚类结果：每个聚类只保留一张图片
                used_clusters = set()  # 使用set而不是dict
                kept_images = []  # 直接保存图片而不是索引
                
                # 处理非噪声点和噪声点
                for i, cluster in enumerate(clusters):
                    if cluster == -1 or cluster not in used_clusters:
                        kept_images.append(current_batch[i])
                        if cluster != -1:
                            used_clusters.add(cluster)
                
                # 如果保留的图片数量等于target_count，说明没有重复，可以直接使用
                if len(kept_images) == target_count:
                    selected_images = kept_images
                    break
                
                # 否则，更新候选列表，移除被聚类的图片
                current_candidates = [img for img in current_candidates if img not in current_batch]
                
                retry_count += 1
                print(f"第 {retry_count} 次尝试，剩余候选图片: {len(current_candidates)}")
                
            except Exception as e:
                print(f"聚类过程发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        if selected_images:
            # 保存结果
            artist_output_dir = output_path / artist_path.name
            original_pic_dir = artist_output_dir / "OriginalPic"
            os.makedirs(original_pic_dir, exist_ok=True)
            
            selected_results = {}
            for img_name, scores, _ in selected_images:
                src_path = find_image_path(artist_path, img_name)
                if src_path:
                    shutil.copy2(src_path, original_pic_dir / img_name)
                    selected_results[img_name] = results_json[img_name]
            
            with open(artist_output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(selected_results, f, ensure_ascii=False, indent=4)
            
            return True
    
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
    parser = argparse.ArgumentParser(description='select original DPO pic for dataset')
    parser.add_argument('--mode', choices=['select', 'fixtagger'], required=True,
                      help='Operation mode: select - select images, fixtagger - fix tags')
    parser.add_argument('--min-side', type=int, default=1279,
                      help='Minimum side length requirement for images (default: 1300 pixels)')
    
    args = parser.parse_args()
    
    # 直接指定路径
    output_path = Path(r"F:\Dataset_selected_MAPO_3")
    source_dataset_path = Path(r"g:\DPO_TESTSET1")

    
    if args.mode == 'select':
        failed_artists = []
        for artist_dir in source_dataset_path.iterdir():
            if artist_dir.is_dir():
                success = process_artist_folder(artist_dir, output_path, min_side_length=args.min_side)
                if not success:
                    failed_artists.append(artist_dir.name)
        
        if failed_artists:
            with open("failed_artists.txt", "w") as f:
                f.write("\n".join(failed_artists))
                
    elif args.mode == 'fixtagger':
        fix_missing_results(output_path, source_dataset_path)

if __name__ == "__main__":
    main()
