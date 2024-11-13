class ImageGenerator:
    def __init__(self, dataset_path: str, sd_generator: SDWebUIGenerator):
        self.dataset_path = dataset_path
        self.sd_generator = sd_generator
        
    def process_dataset(self):
        # 遍历所有艺术家文件夹
        for artist in os.listdir(self.dataset_path):
            artist_path = os.path.join(self.dataset_path, artist)
            if not os.path.isdir(artist_path):
                continue
                
            # 检查必要的文件夹和文件
            original_pic_path = os.path.join(artist_path, 'OriginalPic')
            results_path = os.path.join(artist_path, 'results.json')
            output_path = os.path.join(artist_path, 'DPO_generated')
            
            if not os.path.exists(original_pic_path) or not os.path.exists(results_path):
                continue
                
            os.makedirs(output_path, exist_ok=True)
            
            # 读取 results.json
            with open(results_path, 'r', encoding='utf-8') as f:
                results_json = json.load(f)
                
            # 处理每张原始图片
            for img_name in os.listdir(original_pic_path):
                img_path = os.path.join(original_pic_path, img_name)
                if not os.path.isfile(img_path):
                    continue
                    
                # 获取图片信息
                if img_name not in results_json:
                    continue
                    
                img_data = results_json[img_name]
                finaltag_dan = self._generate_tags(img_data, artist)
                
                # 获取原始图片尺寸
                with Image.open(img_path) as img:
                    original_size = img.size
                
                # 生成图片
                generated_images = self.sd_generator.generate(
                    prompt=finaltag_dan,
                    group='new',  # 默认使用 new 组的负面提示词
                    original_size=original_size
                )
                
                # 保存生成的图片
                for i, gen_img in enumerate(generated_images, 1):
                    output_name = f"{os.path.splitext(img_name)[0]}_DPO{i}.png"
                    output_path = os.path.join(output_path, output_name)
                    gen_img.save(output_path)
                    print(f"Saved generated image: {output_path}")

    def _generate_tags(self, img_data: dict, artist: str) -> str:
        # 处理艺术家标签
        final_artist_tag = artist.replace('_', ' ') + ', '
        
        # 处理特征标签
        features = img_data.get('features', {})
        features_tag = [k.replace('_', ' ') for k, v in features.items() if v > 0.5]
        final_features_tag = ', '.join(features_tag)
        
        # 处理评分标签
        rating_tag = ''
        scores_by_class = img_data.get('scores_by_class', {})
        if scores_by_class:
            max_class = max(scores_by_class, key=scores_by_class.get)
            if max_class != 'masterpiece':
                rating_tag += f'{max_class} quality, '
            else:
                rating_tag += 'masterpiece, '
        
        # 组合最终标签
        finaltag_dan = f"{final_artist_tag}|||{final_features_tag}, {rating_tag}".strip(', ')
        return finaltag_dan

if __name__ == "__main__":
    dataset_path = r"path/to/dataset"
    
    sd_generator = SDWebUIGenerator(
        host="your-host",
        port=443,
        model="your-model"
    )
    
    generator = ImageGenerator(dataset_path, sd_generator)
    generator.process_dataset() 