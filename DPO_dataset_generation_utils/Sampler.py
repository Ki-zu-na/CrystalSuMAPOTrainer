import json
import os
import math
from pathlib import Path
import webuiapi
import time
import random
from tqdm import tqdm
from PIL import Image

class SDXLBatchGenerator:
    def __init__(
        self,
        model_name: str,
        output_dir: str = "temp",
        batch_size: int = 3,
        host: str = "localhost",
        port: int = 7860
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.api = self._connect_with_retry(host, port)
        
        # 设置基本参数
        self.negative_prompt = "lowres,(bad),extra digits,2girls,bad hands,error,text,fewer,extra,missing,worst quality,jpeg artifacts,(low, old, early,mid)"
        self.cfg_scale = 4.0
        self.steps = 30
        self.sampler_name = 'DPM++ 2M SDE'
        self.scheduler = 'SGM Uniform'
        
        # 设置模型
        self.set_model(model_name)
        
    def _connect_with_retry(self, host, port):
        while True:
            try:
                api = webuiapi.WebUIApi(host=host, port=port, use_https=False)
                api.util_get_model_names()
                print("成功连接到SD WebUI API")
                return api
            except Exception as e:
                print(f"连接失败: {str(e)}")
                print("1秒后重试...")
                time.sleep(1)
                
    def set_model(self, model_name):
        self.api.util_set_model(model_name)
        print(f"模型已设置为: {model_name}")
        
    def adjust_resolution(self, width: int, height: int) -> tuple:
        """调整分辨率，确保是8的倍数"""
        # 确保不超过1536的限制
        max_area = 1536 * 1536
        area = width * height
        
        if area > max_area:
            scale = math.sqrt(max_area / area)
            width = int(width * scale)
            height = int(height * scale)
            
        # 调整为8的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        return width, height
        
    def generate_images(self, json_path: str):
        """从JSON文件生成图像"""
        # 加载JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        
        # 遍历艺术家
        for artist_name, pics in tqdm(data.items(), desc="处理艺术家"):
            artist_dir = self.output_dir / artist_name
            artist_dir.mkdir(exist_ok=True)
            
            # 遍历图片
            for pic_name, pic_info in tqdm(pics.items(), desc=f"处理 {artist_name}"):
                # 检查是否已经生成
                if pic_info.get("generated", False):
                    continue
                    
                try:
                    # 调整分辨率
                    width, height = self.adjust_resolution(
                        pic_info["width"],
                        pic_info["height"]
                    )
                    
                    # 生成图像
                    result = self.api.txt2img(
                        prompt=pic_info["tag"],
                        negative_prompt=self.negative_prompt,
                        steps=self.steps,
                        cfg_scale=self.cfg_scale,
                        width=width,
                        height=height,
                        batch_size=self.batch_size,
                        sampler_name=self.sampler_name,
                        scheduler=self.scheduler,
                        seed=random.randint(1, 1000000)
                    )
                    
                    # 保存图像
                    base_name = Path(pic_name).stem
                    for idx, image in enumerate(result.images, 1):
                        output_path = artist_dir / f"{base_name}_DPO{idx}.webp"
                        image.save(output_path, format="WEBP", quality=90)
                    
                    # 更新JSON
                    pic_info["generated"] = True
                    
                    # 保存进度
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    print(f"处理 {pic_name} (艺术家: {artist_name}) 时出错: {str(e)}")
                    continue

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--json_path", type=str, required=True, help="数据集JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="temp", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=3, help="批量大小")
    parser.add_argument("--host", type=str, default="localhost", help="WebUI主机地址")
    parser.add_argument("--port", type=int, default=7860, help="WebUI端口")
    
    args = parser.parse_args()
    
    generator = SDXLBatchGenerator(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        host=args.host,
        port=args.port
    )
    
    generator.generate_images(args.json_path)

if __name__ == "__main__":
    main()