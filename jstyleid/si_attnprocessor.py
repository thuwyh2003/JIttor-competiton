from typing import Any, Optional
import jittor as jt
import torch
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.attention_processor import Attention, AttnProcessor

class SaveFeatureAttnProcessor(AttnProcessor):
    def __init__(self, is_sty_img=True):
        super().__init__()
        self.sty_img = is_sty_img
        if self.sty_img:
            self.ft_dict = [[],[]]
        else:
            self.ft_dict = []
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: jt.Var,
        encoder_hidden_states: Optional[jt.Var] = None,
        attention_mask: Optional[jt.Var] = None,
        temb: Optional[jt.Var] = None,
        scale: float = 1.0,
    ) -> jt.Var:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        
        if self.sty_img:
            self.ft_dict[0].append(key)
            self.ft_dict[1].append(value)
        else:
            self.ft_dict.append(query)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class StyleInjectAttnProcessor(AttnProcessor):
    def __init__(self, _sty_ft: list[list], _cnt_ft: list):
        super().__init__()
        self.sty_ft = _sty_ft
        self.cnt_ft = _cnt_ft
        self.gamma = 0.9
        self.T = 1.8
        self.reverse_step = 0
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: jt.Var,
        encoder_hidden_states: Optional[jt.Var] = None,
        attention_mask: Optional[jt.Var] = None,
        temb: Optional[jt.Var] = None,
        scale: float = 1.0,
    ) -> jt.Var:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.gamma * self.cnt_ft[len(self.cnt_ft) - 1 - self.reverse_step] + (1.0 - self.gamma) * attn.to_q(hidden_states, *args)
        query = query * self.T

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.sty_ft[0][len(self.sty_ft) - 1 - self.reverse_step]
        value = self.sty_ft[1][len(self.sty_ft) - 1- self.reverse_step]

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        self.reverse_step += 1

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
