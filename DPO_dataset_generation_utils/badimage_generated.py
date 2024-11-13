import os
import json
import random
import time
from typing import Dict, List, Tuple
import webuiapi
from PIL import Image
import sdeval
from sdeval.corrupt import AICorruptMetrics

class SDWebUIGenerator:
    def __init__(self, host, port, model, batch_size=5, width=832, height=1218, max_retries=3, retry_delay=5):
        self.host = host
        self.port = port
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api = self._connect_with_retry()
        
        self.negative_prompt = "lowres,bad hands,worst quality,watermark,censored,jpeg artifacts"
        self.cfg_scale = 4.5
        self.steps = 35
        self.sampler_name = 'DPM++ 2M'
        self.scheduler = 'SGM Uniform'
        self.width = width
        self.height = height
        self.seed = random.randint(1, 1000000)
        self.batch_size = batch_size
        self.set_model(model)

    def _connect_with_retry(self):
        while True:
            try:
                api = webuiapi.WebUIApi(host=self.host, port=self.port, use_https=True)
                api.util_get_model_names()
                print("Successfully connected to the SD Web UI API")
                return api
            except Exception as e:
                print(f"Connection attempt failed: {str(e)}")
                print("Retrying in 1 second...")
                time.sleep(1)

    def set_model(self, model):
        self.api.util_set_model(model)
        print("Model set to:" + model)

    def generate(self, prompt, group, original_size, batch_size=None, width=None, height=None):
        batch_size = batch_size or self.batch_size
        width, height = self._adjust_size(original_size, width, height)
        negative_prompt = self._get_negative_prompt(group)

        result = self.api.txt2img(
            prompt=prompt,
            steps=self.steps,
            negative_prompt=negative_prompt,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            scheduler=self.scheduler,
            width=width,
            height=height,
            seed=self.seed,
            batch_size=batch_size
        )
        return result.images

    def _adjust_size(self, original_size, width, height):
        orig_width, orig_height = original_size
        if orig_height > orig_width * 1.15:
            return 832, 1218
        elif orig_width > orig_height * 1.15:
            return 1218, 832
        return width or self.width, height or self.height

    def _get_negative_prompt(self, group):
        if group == 'new':
            return 'lowres,(bad),extra digits,2girls,bad hands,error,text,fewer,extra,missing,worst quality,jpeg artifacts,(low, old, early,mid),'
        return self.negative_prompt

class ImageGenerator:
    def __init__(self, dataset_path: str, output_path: str, sd_generator: SDWebUIGenerator):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.sd_generator = sd_generator
        self.ai_corrupt_metrics = AICorruptMetrics()
        self.data_file = 'generation_data.json'
        self.data = self._load_data()
        os.makedirs(self.output_path, exist_ok=True)

    def _load_data(self) -> Dict:
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            print("Existing generation_data.json found. Loading data...")
            return data
        return {
            'progress': {},
            'tags': {'new': [], 'mid': [], 'old': []}
        }

    def _save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def collect_data(self):
        if self.data['progress']:
            print("Data already collected. Skipping collection phase.")
            return

        for artist in os.listdir(self.dataset_path):
            artist_path = os.path.join(self.dataset_path, artist)
            if os.path.isdir(artist_path):
                self._process_artist(artist, artist_path)
        self._save_data()
        print("Data collection completed. generation_data.json has been created/updated.")

    # _process_artist, _process_image_data, _classify_tag, _find_image_path methods remain unchanged
    def _process_artist(self, artist: str, artist_path: str):
        final_json_path = os.path.join(artist_path, 'final.json')
        if not os.path.exists(final_json_path):
            print(f"final.json not found for artist: {artist}")
            return

        with open(final_json_path, 'r',encoding='utf-8') as f:
            final_data = json.load(f)

        for image, data in final_data.items():
            if artist not in self.data['progress']:
                self.data['progress'][artist] = {}
            if image not in self.data['progress'][artist]:
                self._process_image_data(artist, image, data)

    def _process_image_data(self, artist: str, image: str, data: Dict):
        finaltag_dan = data['finaltag_dan']
        
        # 只处理包含 "1girl" 标签的数据
        if "1girl" not in finaltag_dan.lower():
            return

        # 替换质量标签
        quality_tags = ["great quality", "good quality", "normal quality", "low quality", "worst quality"]
        for tag in quality_tags:
            finaltag_dan = finaltag_dan.replace(tag, "best quality")

        # 替换或添加分辨率标签
        if "lowres" in finaltag_dan:
            finaltag_dan = finaltag_dan.replace("lowres", "absurdres")
        elif "absurdres" not in finaltag_dan:
            finaltag_dan += ", absurdres"

        group = self._classify_tag(finaltag_dan)
        if finaltag_dan not in self.data['tags'][group]:
            self.data['tags'][group].append(finaltag_dan)

        image_path = self._find_image_path(artist, image)
        if not image_path:
            print(f"Image not found: {image} for artist: {artist}")
            return

        original_size = Image.open(image_path).size

        self.data['progress'][artist][image] = {
            'finaltag_dan': finaltag_dan,
            'group': group,
            'original_size': original_size,
            'generated': False
        }

    def _classify_tag(self, tag: str) -> str:
        if any(keyword in tag.lower() for keyword in ['newest', 'recent']):
            return 'new'
        elif 'mid' in tag.lower():
            return 'mid'
        elif any(keyword in tag.lower() for keyword in ['early', 'old']):
            return 'old'
        return 'new'  # Default to 'new' if no classification found
    def _find_image_path(self, artist: str, image: str) -> str:
        for subdir in ['2010s', '2017s', '2020s', '2022s', 'new', 'unknown', 'undefined']:
            path = os.path.join(self.dataset_path, artist, subdir, image)
            if os.path.exists(path):
                return path
        return None

    def generate_images(self, count: int):
        all_tags = []
        for artist, images in self.data['progress'].items():
            for image, image_data in images.items():
                if not image_data['generated']:
                    all_tags.append((artist, image, image_data['finaltag_dan'], image_data['group'], image_data['original_size']))

        if not all_tags:
            print("All images have been generated. No more tags to process.")
            return

        if len(all_tags) < count:
            print(f"Warning: Only {len(all_tags)} tags available. Generating {len(all_tags)} images instead of {count}.")
            count = len(all_tags)

        selected_tags = random.sample(all_tags, count)

        new_count = mid_count = old_count = 0
        for _, _, _, group, _ in selected_tags:
            if group == 'new':
                new_count += 1
            elif group == 'mid':
                mid_count += 1
            else:
                old_count += 1

        print(f"Selected tags: New: {new_count}, Mid: {mid_count}, Old: {old_count}")

        for artist, image, tag, group, original_size in selected_tags:
            generated_images = self.sd_generator.generate(tag, group, original_size)
            best_images = self._evaluate_images(generated_images)
            self._save_best_images(artist, best_images)
            
            # Update the generation status
            self.data['progress'][artist][image]['generated'] = True
            self._save_data()

    def _evaluate_images(self, images: List[Image.Image]) -> List[Tuple[Image.Image, float]]:
        scores = []
        for img in images:
            try:
                score = self.ai_corrupt_metrics.score(img)
                if isinstance(score, dict):
                    score = score.get('score', 0.0)
                elif not isinstance(score, (int, float)):
                    print(f"Unexpected score type: {type(score)}. Using 0.0 as default.")
                    score = 0.0
            except Exception as e:
                print(f"Error evaluating image: {e}")
                score = 0.0
            scores.append((img, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:2]

    def _save_best_images(self, artist: str, best_images: List[Tuple[Image.Image, float]]):
        timestamp = int(time.time())
        for i, (img, score) in enumerate(best_images):
            filename = f"{artist}_{timestamp}_{i+1}.png"
            filepath = os.path.join(self.output_path, filename)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            try:
                img.save(filepath)
                print(f"Saved image: {filepath} with score: {score}")
            except Exception as e:
                print(f"Error saving image {filepath}: {e}")

if __name__ == "__main__":
    dataset_path = r"F:\SDXL_large_Modified_tagged"
    output_path = r"F:\SDXL_test"
    
    sd_generator = SDWebUIGenerator(
        host="officially-paradise-august-develop.trycloudflare.com", 
        port=443, 
        model="13.5-18e"
    )
    
    generator = ImageGenerator(dataset_path, output_path, sd_generator)

    # 第一阶段：如果需要，收集数据并生成或更新 generation_data.json
    generator.collect_data()

    # 第二阶段：根据 generation_data.json 生成图像
    generator.generate_images(5000)  # 生成 100 张图像