import contextlib
import io
import json
import random
import math
import cv2

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate.logging import get_logger
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import PretrainedConfig

from diffusers import DiffusionPipeline, UNet2DConditionModel


logger = get_logger(__name__)

with open("validation_prompts.json", "r") as f:
    validation_prompt_file = json.load(f)

VALIDATION_PROMPTS = validation_prompt_file["VALIDATION_PROMPTS"]


# Loading baseline model
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Logging validations during training
def log_validation(args, unet, vae, accelerator, weight_dtype, epoch, is_final_validation=False):
    logger.info(f"Running validation... \n Generating images with prompts:\n" f" {VALIDATION_PROMPTS}.")

    if is_final_validation:
        if args.mixed_precision == "fp16":
            vae.to(weight_dtype)

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    if not is_final_validation:
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        if args.lora_rank is not None:
            pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")
        else:
            unet = UNet2DConditionModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
            pipeline.unet = unet

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    images = []
    context = contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()

    guidance_scale = 5.0
    num_inference_steps = 35
    for prompt in VALIDATION_PROMPTS:
        with context:
            image = pipeline(
                prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
            ).images[0]
            images.append(image)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}") for i, image in enumerate(images)
                    ]
                }
            )

    # Also log images without the LoRA params for comparison.
    if is_final_validation:
        if args.lora_rank is not None:
            pipeline.disable_lora()
        else:
            del pipeline
            # We reinitialize the pipeline here with the pre-trained UNet.
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            ).to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        no_lora_images = [
            pipeline(
                prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
            ).images[0]
            for prompt in VALIDATION_PROMPTS
        ]

        tracker_key = "test_without_lora" if args.lora_rank is not None else "test_without_aligned_unet"
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in no_lora_images])
                tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        tracker_key: [
                            wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                            for i, image in enumerate(no_lora_images)
                        ]
                    }
                )


def process_input_ids(input_ids, tokenizer, max_length):
    """Process input ids for long sequences"""
    if max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        for i in range(
            1,
            max_length - tokenizer.model_max_length + 2,
            tokenizer.model_max_length - 2,
        ):
            ids_chunk = (
                input_ids[0].unsqueeze(0),  # BOS
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),  # PAD or EOS
            )
            ids_chunk = torch.cat(ids_chunk)

            if ids_chunk[-2] not in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                ids_chunk[-1] = tokenizer.eos_token_id
            if ids_chunk[1] == tokenizer.pad_token_id:
                ids_chunk[1] = tokenizer.eos_token_id

            iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)
    return input_ids

def tokenize_captions(tokenizers, examples, max_length=255):
    captions = []
    for caption in examples["caption"]:
        # Split caption by ||| delimiter
        parts = caption.split("|||")
        if len(parts) == 2:
            first_words = parts[0].strip().split(",")
            second_words = parts[1].strip().split(",")
            
            random.shuffle(first_words)
            random.shuffle(second_words)
            
            shuffled_caption = ",".join(first_words + second_words)
            captions.append(shuffled_caption)
        else:
            captions.append(caption)

    # Process tokens for both tokenizers
    def get_tokens(tokenizer, texts):
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
        return tokens

    tokens_one = get_tokens(tokenizers[0], captions)
    tokens_two = get_tokens(tokenizers[1], captions)
    
    # Process long sequences if needed
    tokens_one = torch.stack([process_input_ids(inp, tokenizers[0], max_length) for inp in tokens_one])
    tokens_two = torch.stack([process_input_ids(inp, tokenizers[1], max_length) for inp in tokens_two])

    return tokens_one, tokens_two

@torch.no_grad()
def encode_prompt(text_encoders, text_input_ids_list, max_length=255):
    prompt_embeds_list = []
    device = next(text_encoders[0].parameters()).device

    for i, text_encoder in enumerate(text_encoders):
        text_input_ids = text_input_ids_list[i]
        
        # Get batch size and reshape input ids
        b_size = text_input_ids.size()[0]
        text_input_ids = text_input_ids.reshape((-1, text_encoder.config.max_position_embeddings))
        text_input_ids = text_input_ids.to(device)

        # Get embeddings
        enc_out = text_encoder(text_input_ids, output_hidden_states=True, return_dict=True)
        
        # Handle different encoder outputs
        if i == len(text_encoders) - 1:  # Second text encoder
            hidden_states = enc_out["hidden_states"][-2]
            pooled_prompt_embeds = enc_out["text_embeds"]
        else:  # First text encoder
            hidden_states = enc_out["hidden_states"][11]

        # Reshape hidden states
        hidden_states = hidden_states.reshape((b_size, -1, hidden_states.shape[-1]))

        # Process long sequences
        if max_length is not None:
            states_list = [hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for j in range(1, max_length, text_encoder.config.max_position_embeddings):
                chunk = hidden_states[:, j : j + text_encoder.config.max_position_embeddings - 2]
                states_list.append(chunk)
            states_list.append(hidden_states[:, -1].unsqueeze(1))  # <EOS>
            hidden_states = torch.cat(states_list, dim=1)

            if i == len(text_encoders) - 1:
                n_chunks = max_length // 75
                pooled_prompt_embeds = pooled_prompt_embeds[::n_chunks]

        prompt_embeds_list.append(hidden_states)

    # Concatenate embeddings from both encoders
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    
    return prompt_embeds, pooled_prompt_embeds


def get_wandb_url():
    wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb.run.url}).
"""
    return wandb_info


def get_dataset_preprocessor(args, tokenizer_one, tokenizer_two):

    
    # Preprocessing the datasets.
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    normalize = transforms.Normalize([0.5], [0.5])

    target_area = args.resolution * args.resolution
    divisible = args.divisible
    batch_size = args.preprocess_batch_size

    def calculate_adaptive_size(height, width, target_area, divisible):
        img_area = height * width
        
        if img_area > target_area:
            scale_factor = math.sqrt(target_area / img_area)
            width = math.floor(width * scale_factor / divisible) * divisible
            height = math.floor(height * scale_factor / divisible) * divisible
        
        width = width - width % divisible
        height = height - height % divisible
        
        return height, width

    def process_image_batch(image_bytes_batch, target_sizes=None):
        processed_images = []
        valid_indices = []
        
        for idx, im_bytes in enumerate(image_bytes_batch):
            try:
                # 使用cv2读取图像
                nparr = np.frombuffer(im_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image")
                
                # BGR转RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if target_sizes is not None:
                    # 确保尺寸符合divisible
                    target_h, target_w = target_sizes[idx]
                    target_w = target_w - target_w % divisible
                    target_h = target_h - target_h % divisible
                    # 使用LANCZOS4进行缩放
                    image = cv2.resize(image, (target_w, target_h), 
                                     interpolation=cv2.INTER_LANCZOS4)
                
                # 转换为tensor
                img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                processed_images.append(img_tensor)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
                
        return processed_images, valid_indices

    def preprocess_train(examples):
        num_examples = len(examples["jpg_0"])
        all_pixel_values = []
        all_original_sizes = []
        all_crop_top_lefts = []
        valid_indices = set(range(num_examples))

        for i in range(0, num_examples, batch_size):
            batch_slice = slice(i, min(i + batch_size, num_examples))
            
            # 处理第一组图像
            images_0, valid_idx_0 = process_image_batch(examples["jpg_0"][batch_slice])
            if not images_0:
                continue
                
            original_sizes = [(img.shape[1], img.shape[2]) for img in images_0]
            adaptive_sizes = [
                calculate_adaptive_size(h, w, target_area, divisible) 
                for h, w in original_sizes
            ]

            # 处理第二组图像
            images_1, valid_idx_1 = process_image_batch(
                examples["jpg_1"][batch_slice],
                target_sizes=original_sizes
            )

            batch_valid_indices = set(valid_idx_0) & set(valid_idx_1)
            valid_indices &= {idx + i for idx in batch_valid_indices}

            for idx in batch_valid_indices:
                img_0 = images_0[valid_idx_0.index(idx)]
                img_1 = images_1[valid_idx_1.index(idx)]
                
                target_h, target_w = adaptive_sizes[valid_idx_0.index(idx)]
                
                # 如果需要进一步调整大小
                if img_0.shape[1] > target_h or img_0.shape[2] > target_w:
                    # 转换回numpy进行resize
                    img_0_np = img_0.numpy().transpose(1, 2, 0)
                    img_1_np = img_1.numpy().transpose(1, 2, 0)
                    
                    img_0_np = cv2.resize(img_0_np, (target_w, target_h), 
                                        interpolation=cv2.INTER_LANCZOS4)
                    img_1_np = cv2.resize(img_1_np, (target_w, target_h), 
                                        interpolation=cv2.INTER_LANCZOS4)
                    
                    # 转回tensor
                    img_0 = torch.from_numpy(img_0_np.transpose(2, 0, 1)).float()
                    img_1 = torch.from_numpy(img_1_np.transpose(2, 0, 1)).float()

                # 处理标签噪声
                label_0 = examples["label_0"][idx + i]
                if args.label_noise_prob is not None and random.random() < args.label_noise_prob:
                    label_0 = 1 - label_0

                im_pair = (img_0, img_1) if label_0 == 1 else (img_1, img_0)
                combined_im = torch.cat(im_pair, dim=0)

                # 裁剪
                if not args.random_crop:
                    y1 = max(0, int(round((combined_im.shape[1] - target_h) / 2.0)))
                    x1 = max(0, int(round((combined_im.shape[2] - target_w) / 2.0)))
                else:
                    y1, x1, h, w = transforms.RandomCrop.get_params(
                        combined_im, (target_h, target_w)
                    )

                combined_im = crop(combined_im, y1, x1, target_h, target_w)
                crop_top_left = (y1, x1)

                # 翻转
                if not args.no_hflip and random.random() < 0.5:
                    combined_im = train_flip(combined_im)

                combined_im = normalize(combined_im)
                
                all_pixel_values.append(combined_im)
                all_original_sizes.append(original_sizes[valid_idx_0.index(idx)])
                all_crop_top_lefts.append(crop_top_left)

        # 只保留有效样本
        valid_indices = sorted(list(valid_indices))
        examples = {k: [v[i] for i in valid_indices] for k, v in examples.items()}
        
        examples["pixel_values"] = all_pixel_values
        examples["original_sizes"] = all_original_sizes
        examples["crop_top_lefts"] = all_crop_top_lefts

        try:
            tokens_one, tokens_two = tokenize_captions([tokenizer_one, tokenizer_two], examples)
            examples["input_ids_one"] = tokens_one
            examples["input_ids_two"] = tokens_two
        except Exception as e:
            print(f"Error in tokenization: {e}")
            return None

        return examples

    return preprocess_train


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }


def compute_time_ids(args, accelerator, weight_dtype, original_size, crops_coords_top_left):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = (args.resolution, args.resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    return add_time_ids


def compute_loss(args, noise_scheduler, model_pred, target):
    model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
    model_losses_w, model_losses_l = model_losses.chunk(2)
    log_odds = (args.snr_value * model_losses_w) / (torch.exp(args.snr_value * model_losses_w) - 1) - (
        args.snr_value * model_losses_l
    ) / (torch.exp(args.snr_value * model_losses_l) - 1)

    # Ratio loss.
    # By multiplying T to the inner term, we try to maximize the margin throughout the overall denoising process.
    ratio = F.logsigmoid(log_odds * noise_scheduler.config.num_train_timesteps)
    ratio_losses = args.beta_mapo * ratio

    # Full ORPO loss
    loss = model_losses_w.mean() - ratio_losses.mean()
    return loss, model_losses_w, model_losses_l, ratio_losses
