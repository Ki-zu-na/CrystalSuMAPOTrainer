import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict
from imgutils.metrics import lpips_clustering
from PIL import Image

def load_json_data(json_path: str) -> Dict:
    """Load and parse results.json file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_bad_image(image_scores, real_threshold=0.5, aesthetic_threshold=4.0, monochrome_threshold=0.8):
    imgscore = image_scores["imgscore"]
    anime_real_score = image_scores["anime_real_score"]
    aesthetic_score = image_scores["aesthetic_score"]
    scores_by_class = image_scores["scores_by_class"]
    anime_style_age_score = image_scores["anime_style_age_score"]
    monochrome_score = image_scores["monochrome_score"]
    bad_anime_style_ages = ["1970s-", "1980s", "1990s", "2000s"]
    bad_imgscore_types = ["not_painting", "3d"]
    
    return (
        max(imgscore, key=imgscore.get) in bad_imgscore_types or
        anime_real_score["real"] > real_threshold or
        aesthetic_score < aesthetic_threshold or
        
        max(anime_style_age_score, key=anime_style_age_score.get) in bad_anime_style_ages or
        monochrome_score > monochrome_threshold
    )
    
def process_artist_folder(artist_path: Path, output_path: Path, target_count: int = 40) -> bool:
    """Process single artist folder and select images"""
    # Load results.json
    results_json = load_json_data(artist_path / "results.json")
    
    # Separate good and bad images
    good_images = []
    bad_images = []
    
    for img_name, scores in results_json.items():
        if not is_bad_image(scores, 0.5, 4.0, 0.8):
            good_images.append((img_name, scores['aesthetic_score']))
        else:
            bad_images.append((img_name, scores['aesthetic_score']))
    
    # Sort bad images by aesthetic score for potential later use
    bad_images.sort(key=lambda x: x[1], reverse=True)
    
    selected_images = []
    
    # Try to get initial 40 images from good images
    if len(good_images) >= target_count:
        candidates = random.sample(good_images, target_count)
        
        while len(selected_images) < target_count:
            # Get full paths for current candidates
            candidate_paths = [find_image_path(artist_path, img[0]) for img in candidates]
            
            # Run clustering
            clusters = lpips_clustering(candidate_paths)
            
            # Process clustering results
            used_clusters = set()
            for i, cluster in enumerate(clusters):
                if cluster == -1:  # Not in any cluster
                    selected_images.append(candidates[i])
                elif cluster not in used_clusters:
                    selected_images.append(candidates[i])
                    used_clusters.add(cluster)
            
            # If we need more images
            if len(selected_images) < target_count:
                remaining_count = target_count - len(selected_images)
                # Get new candidates from good_images, excluding already selected ones
                new_candidates = random.sample([img for img in good_images 
                                             if img not in selected_images], 
                                             min(100, remaining_count))
                candidates = selected_images + new_candidates
    
    # If we still don't have enough images, use bad images
    if len(selected_images) < target_count:
        remaining_count = target_count - len(selected_images)
        selected_images.extend(bad_images[:remaining_count])
    
    # Copy selected images to output directory
    if len(selected_images) == target_count:
        os.makedirs(output_path / artist_path.name, exist_ok=True)
        for img_name, _ in selected_images:
            src_path = find_image_path(artist_path, img_name)
            if src_path:
                shutil.copy2(src_path, output_path / artist_path.name / img_name)
        return True
    
    return False

def find_image_path(artist_path: Path, image_name: str) -> Path:
    """Find the actual path of an image in the artist's directory structure"""
    search_dirs = ['2020s', '2022s', 'new', 'unknown', 'undefined']
    for dir_name in search_dirs:
        img_path = artist_path / dir_name / image_name
        if img_path.exists():
            return img_path
    return None

def main():
    # Set up paths
    dataset_path = Path("Dataset")
    output_path = Path("Dataset_selected")
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
