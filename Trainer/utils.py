import contextlib
import io
import json
import random

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
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_train(examples):
        all_pixel_values = []
        images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples["jpg_0"]]
        original_sizes = [(image.height, image.width) for image in images]
        crop_top_lefts = []

        for col_name in ["jpg_0", "jpg_1"]:
            images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            if col_name == "jpg_1":
                # Need to bring down the image to the same resolution.
                # This seems like the simplest reasonable approach.
                # "::-1" because PIL resize takes (width, height).
                images = [image.resize(original_sizes[i][::-1]) for i, image in enumerate(images)]
            pixel_values = [to_tensor(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            # Label noise.
            if args.label_noise_prob is not None and random.random() < args.label_noise_prob:
                label_0 = 1 - label_0

            if label_0 == 0:
                im_tup = im_tup[::-1]

            combined_im = torch.cat(im_tup, dim=0)  # no batch dim

            # Resize.
            combined_im = train_resize(combined_im)

            # Flipping.
            if not args.no_hflip and random.random() < 0.5:
                combined_im = train_flip(combined_im)

            # Cropping.
            if not args.random_crop:
                y1 = max(0, int(round((combined_im.shape[1] - args.resolution) / 2.0)))
                x1 = max(0, int(round((combined_im.shape[2] - args.resolution) / 2.0)))
                combined_im = train_crop(combined_im)
            else:
                y1, x1, h, w = train_crop.get_params(combined_im, (args.resolution, args.resolution))
                combined_im = crop(combined_im, y1, x1, h, w)

            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            combined_im = normalize(combined_im)
            combined_pixel_values.append(combined_im)

        examples["pixel_values"] = combined_pixel_values
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        tokens_one, tokens_two = tokenize_captions([tokenizer_one, tokenizer_two], examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
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


class ORLossTracker:
    def __init__(self, window_size=100, threshold=1e-6):
        self.window_size = window_size
        self.threshold = threshold
        self.or_losses = []
        self.beta_history = []
        
    def update(self, or_loss):
        """更新OR loss历史"""
        self.or_losses.append(or_loss)
        if len(self.or_losses) > self.window_size:
            self.or_losses.pop(0)
            
    def get_loss_trend(self):
        """计算OR loss的趋势"""
        if len(self.or_losses) < 2:
            return 0
        
        mid_point = len(self.or_losses) // 2
        recent_avg = sum(self.or_losses[mid_point:]) / len(self.or_losses[mid_point:])
        previous_avg = sum(self.or_losses[:mid_point]) / len(self.or_losses[:mid_point])
        
        return recent_avg - previous_avg
    
    def should_adjust_beta(self):
        """判断是否需要调整beta"""
        if len(self.or_losses) < self.window_size:
            return False
        return abs(self.get_loss_trend()) > self.threshold

def compute_loss(args, noise_scheduler, model_pred, target, or_tracker=None, current_step=None):
    model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
    model_losses_w, model_losses_l = model_losses.chunk(2)
    
    # 计算基础的log_odds
    log_odds = (args.snr_value * model_losses_w) / (torch.exp(args.snr_value * model_losses_w) - 1) - (
        args.snr_value * model_losses_l
    ) / (torch.exp(args.snr_value * model_losses_l) - 1)

    ratio = F.logsigmoid(log_odds * noise_scheduler.config.num_train_timesteps)
    
    # 动态调整beta_mapo
    current_beta = args.beta_mapo
    if or_tracker is not None:
        or_loss = -ratio.mean().detach().item()
        or_tracker.update(or_loss)
        
        if or_tracker.should_adjust_beta():
            loss_trend = or_tracker.get_loss_trend()
            
            # 调整beta的变化幅度为更小的值
            if loss_trend > 0:  # OR loss在增加
                # 减小beta，但保持在合理范围内
                current_beta = max(args.beta_mapo * 0.95, args.beta_mapo * 0.5)  # 更温和的减少
            else:  # OR loss在减少
                # 增加beta，但保持在合理范围内
                current_beta = min(args.beta_mapo * 1.05, args.beta_mapo * 1.5)  # 更温和的增加
                
        or_tracker.beta_history.append(current_beta)
    
    ratio_losses = current_beta * ratio
    loss = model_losses_w.mean() - ratio_losses.mean()
    
    return loss, model_losses_w, model_losses_l, ratio_losses, current_beta
