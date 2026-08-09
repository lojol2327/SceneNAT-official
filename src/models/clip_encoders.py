from typing import *
from PIL.Image import Image as PILImage

import torch
from torch import nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn.functional as F


class CLIPTextEncoder(nn.Module):
    def __init__(self,
        name="openai/clip-vit-base-patch32",
        max_length=77, device="cpu"
    ):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(name).to(device).eval()

        self.text_emb_dim = self.text_encoder.config.hidden_size
        self.max_length = max_length
        self.device = device

        assert self.max_length == self.tokenizer.model_max_length

    @torch.no_grad()
    def forward(self, prompt: Union[str, List[str]], norm=True, return_tokens=False):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        text_encoder_output = self.text_encoder(text_input_ids.to(self.device))

        text_last_hidden_state = text_encoder_output.last_hidden_state.float()  # (num_prompts, max_length, text_emb_dim)
        text_embeds = text_encoder_output.text_embeds.float()  # (num_prompts, text_emb_dim)
        if norm:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # L2 normalize

        if return_tokens:
            # Convert token IDs to words
            processed_words = []
            for batch_idx in range(text_input_ids.shape[0]):
                ids = text_input_ids[batch_idx].tolist()
                attention_mask = text_inputs.attention_mask[batch_idx]
                if (attention_mask == 0).any():
                    first_zero_idx = (attention_mask == 0).nonzero(as_tuple=True)[0][0].item()
                else:
                    first_zero_idx = len(attention_mask)
                
                words = []
                for idx in range(first_zero_idx):
                    word = self.tokenizer._convert_id_to_token(ids[idx])
                    if word == '<|startoftext|>':
                        words.append('START')
                    elif word == '<|endoftext|>':
                        words.append('END')
                    else:
                        words.append(word.replace('</w>', ''))
                processed_words.append(words)

            return text_last_hidden_state, text_embeds, processed_words

        return text_last_hidden_state, text_embeds

class CLIPImageEncoder(nn.Module):
    def __init__(self,
        name="openai/clip-vit-base-patch32",
        device="cpu"
    ):
        super().__init__()

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(name).to(device).eval()

        self.image_emb_dim = self.image_encoder.config.hidden_size
        self.device = device

    @torch.no_grad()
    def forward(self, image: Union[PILImage, List[PILImage]], norm=True):
        image = self.image_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_embeds = self.image_encoder(image).image_embeds.float()  # (num_images, image_emb_dim)
        if norm:
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds
