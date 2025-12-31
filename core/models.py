import numpy as np
import random
import trimesh
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPImageProcessor

import kiui
from core.options import Options
from diffusers import StableDiffusionPipeline, DDIMScheduler
from accelerate import Accelerator
accelerator = Accelerator()

class LMM(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        if opt.cond_mode == 'text':
            pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base')
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            if opt.freeze_encoder:
                self.text_encoder = self.text_encoder.eval().half()
                self.text_encoder.requires_grad_(False)
            self.proj_cond = nn.Linear(1024, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)
        elif opt.cond_mode == 'image':
            self.normalize_image = partial(TF.normalize, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # ref: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json#L6
            self.image_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            if opt.freeze_encoder:
                self.image_encoder = self.image_encoder.eval().half()
                self.image_encoder.requires_grad_(False)
            self.proj_cond = nn.Linear(1280, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)
        elif opt.cond_mode == 'image+text':
            # assert not opt.freeze_encoder
            pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base')
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.normalize_image = partial(TF.normalize, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # ref: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json#L6
            self.image_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            if opt.freeze_encoder:
                self.text_encoder = self.text_encoder.eval().half()
                self.text_encoder.requires_grad_(False)
                self.image_encoder = self.image_encoder.eval().half()
                self.image_encoder.requires_grad_(False)
            self.proj_cond_text = nn.Linear(1024, opt.hidden_dim)
            self.proj_cond_image = nn.Linear(1280, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)
        
        elif opt.cond_mode == 'image+depth':
            self.normalize_image = partial(TF.normalize, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # ref: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json#L6
            self.image_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            self.depth_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            if opt.freeze_encoder:
                self.image_encoder = self.image_encoder.eval().half()
                self.image_encoder.requires_grad_(False)
                self.depth_encoder = self.depth_encoder.eval().half()
                self.depth_encoder.requires_grad_(False)
            self.proj_cond_image = nn.Linear(1280, opt.hidden_dim)
            self.proj_cond_depth = nn.Linear(1280, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)
            
        elif opt.cond_mode == 'depth+image+text':
            pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base')
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.normalize_image = partial(TF.normalize, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # ref: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json#L6
            self.image_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            self.depth_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            if opt.freeze_encoder:
                self.text_encoder = self.text_encoder.eval().half()
                self.text_encoder.requires_grad_(False)
                self.image_encoder = self.image_encoder.eval().half()
                self.image_encoder.requires_grad_(False)
                self.depth_encoder = self.depth_encoder.eval().half()
                self.depth_encoder.requires_grad_(False)
            self.proj_cond_text = nn.Linear(1024, opt.hidden_dim)
            self.proj_cond_image = nn.Linear(1280, opt.hidden_dim)
            self.proj_cond_depth = nn.Linear(1280, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)
            
        else:
            raise ValueError(f'Unknown cond_mode: {opt.cond_mode}')

        self.vocab_size = opt.discrete_bins + 4

        from core.transformer.modeling_opt import ShapeOPTConfig, ShapeOPT
        self.config = ShapeOPTConfig(
            vocab_size=self.vocab_size,
            hidden_dim=opt.hidden_dim,
            intermediate_dim=opt.hidden_dim * 4 if opt.intermediate_dim is None else opt.intermediate_dim,
            num_hidden_layers=opt.num_layers,
            num_attention_heads=opt.num_heads,
            max_position_embeddings=opt.max_seq_length + opt.num_cond_tokens + 10, # pos embedding size
            num_cond_tokens=opt.num_cond_tokens,
        )
        self.mesh_decoder = ShapeOPT(self.config)

        if opt.checkpointing:
            self.mesh_decoder.model.gradient_checkpointing_enable()
    
    def encode_cond(self, conds):

        results = {}

        grad_ctx = torch.no_grad if self.opt.freeze_encoder else nullcontext
        
        if self.opt.cond_mode == 'text':
            with grad_ctx():
                if isinstance(conds, list):  # 如果是文本列表
                    inputs = self.tokenizer(
                        conds,
                        padding="max_length",
                        truncation_strategy='longest_first',
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).to(device=self.text_encoder.device)
                cond_embeds = self.text_encoder(**inputs).last_hidden_state # [B, 77, 768]
            cond_embeds = self.norm_cond(self.proj_cond(cond_embeds))
        elif self.opt.cond_mode == 'image':
            with grad_ctx():
                images_clip = self.normalize_image(conds)
                images_clip = F.interpolate(images_clip, (224, 224), mode='bilinear', align_corners=False)
                images_clip = images_clip.to(device=self.image_encoder.device)
                
                from PIL import Image
                images_array = images_clip.cpu()
                image = images_array[0]  # 获取第一个图片（如果是 batch）
                image = image.permute(1, 2, 0)  # 转换形状为 [H, W, C]
                image = image * 255  # 将像素值从 [0, 1] 转换为 [0, 255]
                image = image.numpy().astype('uint8')  # 转换为 NumPy 数组并确保数据类型是 uint8
                image_pil = Image.fromarray(image)
                image_pil.save("test.png")
                cond_embeds = self.image_encoder(images_clip).last_hidden_state # [B, 257, 1280]
            cond_embeds = self.norm_cond(self.proj_cond(cond_embeds))
        elif self.opt.cond_mode == 'image+text':
            texts, rgb_images = conds
            with grad_ctx():
                if isinstance(texts, list):  # 如果是文本列表
                    inputs = self.tokenizer(
                        texts,
                        padding="max_length",
                        truncation_strategy='longest_first',
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).to(device=self.text_encoder.device)
                cond_embeds_text = self.text_encoder(**inputs).last_hidden_state # [B, 77, 768]
                
                images_clip = self.normalize_image(rgb_images)
                images_clip = F.interpolate(images_clip, (224, 224), mode='bilinear', align_corners=False)
                images_clip = images_clip.to(device=self.image_encoder.device)
                cond_embeds_image = self.image_encoder(images_clip).last_hidden_state # [B, 257, 1280]
            
            cond_embeds_text = self.proj_cond_text(cond_embeds_text)
            cond_embeds_image = self.proj_cond_image(cond_embeds_image)
                
            cond_embeds = torch.cat((cond_embeds_text, cond_embeds_image), dim=1)
            cond_embeds = self.norm_cond(cond_embeds)
                            
        elif self.opt.cond_mode == 'image+depth':
            depths, rgb_images = conds
            with grad_ctx():
                images_clip = self.normalize_image(rgb_images)
                images_clip = F.interpolate(images_clip, (224, 224), mode='bilinear', align_corners=False)
                images_clip = images_clip.to(device=self.image_encoder.device)
                cond_embeds_image = self.image_encoder(images_clip).last_hidden_state # [B, 257, 1280]
                
                depth_clip = F.interpolate(depths, (224, 224), mode='bilinear', align_corners=False)
                depth_clip = depth_clip.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                depth_clip = depth_clip.to(device=self.depth_encoder.device)
                cond_embeds_depth = self.depth_encoder(depth_clip).last_hidden_state # [B, 257, 1280]
            
            cond_embeds_image = self.proj_cond_image(cond_embeds_image)
            cond_embeds_depth = self.proj_cond_depth(cond_embeds_depth)
                
            cond_embeds = torch.cat((cond_embeds_image, cond_embeds_depth), dim=1)
            cond_embeds = self.norm_cond(cond_embeds)
            
        elif self.opt.cond_mode == 'depth+image+text':
            texts, rgb_images, depths = conds
            with grad_ctx():
                if isinstance(texts, list):  # 如果是文本列表
                    inputs = self.tokenizer(
                        texts,
                        padding="max_length",
                        truncation_strategy='longest_first',
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).to(device=self.text_encoder.device)
                cond_embeds_text = self.text_encoder(**inputs).last_hidden_state # [B, 77, 768]
                
                images_clip = self.normalize_image(rgb_images)
                images_clip = F.interpolate(images_clip, (224, 224), mode='bilinear', align_corners=False)
                images_clip = images_clip.to(device=self.image_encoder.device)
                cond_embeds_image = self.image_encoder(images_clip).last_hidden_state # [B, 257, 1280]
                
                depth_clip = F.interpolate(depths, (224, 224), mode='bilinear', align_corners=False)
                depth_clip = depth_clip.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                depth_clip = depth_clip.to(device=self.depth_encoder.device)
                cond_embeds_depth = self.depth_encoder(depth_clip).last_hidden_state # [B, 257, 1280]
            
            cond_embeds_text = self.proj_cond_text(cond_embeds_text)
            cond_embeds_image = self.proj_cond_image(cond_embeds_image)
            cond_embeds_depth = self.proj_cond_depth(cond_embeds_depth)
                
            cond_embeds = torch.cat((cond_embeds_text, cond_embeds_image, cond_embeds_depth), dim=1)
            cond_embeds = self.norm_cond(cond_embeds)
                        
        elif self.opt.cond_mode == 'none': # will ignore conds
            cond_embeds = None

        results['cond_embeds'] = cond_embeds
        return results


    def forward(self, data, step_ratio=1):

        results = {}

        rgb_images = data['rgb'] # [batch_size, 3, height, width]
        depths = data['depth'] # [batch_size, 3, height, width]
        texts = data['text'] # [batch_size, max_text_length]
        tokens = data['tokens'] # tokens [B, 1+M+1], long
        labels = data['labels'] # labels [B, C+1+M+1], long
        masks = data['masks'] # attn masks [B, C+1+M+1], bool
        num_tokens = data['num_tokens'] # num_tokens [B], long

        B = tokens.shape[0]

        if self.opt.cond_mode == 'text':
            conds = texts
        elif self.opt.cond_mode == 'image':
            conds = rgb_images
        elif self.opt.cond_mode == 'image+text':
            conds = [texts, rgb_images]
        elif self.opt.cond_mode == 'image+depth':
            conds = [depths, rgb_images]
        elif self.opt.cond_mode == 'depth+image+text':
            conds = [texts, rgb_images, depths]  

        # encode conds
        results_cond = self.encode_cond(conds) # [B, N, C]
        cond_embeds = results_cond['cond_embeds']

        # encode tokens
        token_embeds = self.mesh_decoder.model.embd(tokens)

        # insert cond embeds
        if cond_embeds is not None:
            inputs_embeds = torch.cat((cond_embeds, token_embeds), dim=1)
        else:
            inputs_embeds = token_embeds
        
        # call decoder
        kwargs = {
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'attention_mask': masks,
            'num_tokens': num_tokens,
        }

        outputs = self.mesh_decoder(**kwargs)

        results['loss_ce'] = outputs.loss
        loss = outputs.loss
       
        results['loss'] = loss
        results['logits'] = outputs.logits # [B, 1+C+M+1, V]

        return results

    @torch.no_grad()
    def generate(
            self,
            conds,
            resume_ids=None,
            max_new_tokens=None,
            clean=True,
        ):
                        
            if self.opt.cond_mode == 'text':
                B = len(conds)
            elif self.opt.cond_mode == 'image':
                B = conds.shape[0]
            elif self.opt.cond_mode == 'image+text':
                B = conds[1].shape[0]
            elif self.opt.cond_mode == 'image+depth':
                B = conds[1].shape[0]
            elif self.opt.cond_mode == 'depth+image+text':
                B = conds[1].shape[0]
            assert B == 1, 'Batch size must be 1 for generation.'

            # encode input_embeds (only COND)
            results_cond = self.encode_cond(conds) # [B, N, C]
            cond_embeds = results_cond['cond_embeds']

            # BOS input_ids to start generation
            input_ids = torch.full((B, 1), self.opt.bos_token_id, dtype=torch.long, device=cond_embeds.device) # BOS token
            if resume_ids is not None:
                input_ids = torch.cat((input_ids, resume_ids), dim=1)

            tokens_embeds = self.mesh_decoder.model.embd(input_ids) # [B, 1, C]

            if cond_embeds is not None:
                inputs_embeds = torch.cat((cond_embeds, tokens_embeds), dim=1)
            else:
                inputs_embeds = tokens_embeds

            # constraint function
            tokenizer = None
            if tokenizer is None:
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    candidates = list(range(3, self.vocab_size))
                    # BOS is already provided as the first input token 
                    if input_ids.shape[0] % 10 == 1:
                        candidates.append(self.opt.eos_token_id) # EOS is only allowed at 1 + 10 * N position
                    return candidates
            
            max_new_tokens = 10 * self.opt.pose_length + 1

            num_tokens = torch.full((B,), 10 * self.opt.pose_length + 2 + self.opt.num_cond_tokens, dtype=torch.long, device=cond_embeds.device)

            kwargs = {
                # 'input_ids': input_ids,
                'inputs_embeds': inputs_embeds,
                'num_tokens': num_tokens,
                'pad_token_id': self.opt.pad_token_id,
                'bos_token_id': self.opt.bos_token_id,
                'eos_token_id': self.opt.eos_token_id,
                'max_new_tokens': max_new_tokens,
                'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn, # after converging we don't actually need this.
            }

            if self.opt.generate_mode == 'greedy':
                kwargs['num_beams'] = 1
            elif self.opt.generate_mode == 'sample':
                kwargs['do_sample'] = True
                kwargs['top_k'] = 10

            output_ids = self.mesh_decoder.generate(**kwargs) # [B, 1+C+M+1]

            # batch detokenize to meshes
            meshes = []
            all_tokens = []
            for b in range(B):
                tokens = output_ids[b].detach().cpu().numpy() - 3  # remove offset
                assert np.all(tokens >= 0) and np.all(tokens <= self.opt.discrete_bins)
                if resume_ids is not None:
                    tokens = np.concatenate((resume_ids[b].detach().cpu().numpy(), tokens), axis=0)
                all_tokens.append(tokens)

            return all_tokens
