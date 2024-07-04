import jittor as jt
from jittor import nn
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from si_attnprocessor import SaveFeatureAttnProcessor, StyleInjectAttnProcessor

jt.flags.use_cuda = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "stabilityai/stable-diffusion-2-1"

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder='scheduler',
    prediction_type='v_prediction'
)
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    safety_checker=None,
)

list_attn = ['up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
             'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
             'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor',
             'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor',
             'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor',
             'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor']

parser = ArgumentParser()
parser.add_argument('--sty_idx', default='00')
parser.add_argument('--sty_dir', default='./data/input/style')
parser.add_argument('--cnt_dir', default='./data/input/content')
parser.add_argument('--out_img_dir', default='./data/output')
parser.add_argument('--step_num', type=int, default=50)
arguments = parser.parse_args()

def load_img(path) -> jt.Var:
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded image of size ({x}, {y}) from {path}")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = jt.Var(image).unsqueeze(0).permute(0, 3, 1, 2)
    return 2.0 * image - 1.0

@torch.no_grad()
def ddim_inversion(img: jt.Var) -> jt.Var:
    timesteps = jt.Var(np.flip(pipeline.scheduler.timesteps))
    
    latents = 0.1875 * pipeline.vae.encode(img).latent_dist.sample()
    text_embeddings = pipeline._encode_prompt(prompt='', device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    
    for i in tqdm(range(0, arguments.step_num), desc='DDIM Inversion', total=arguments.step_num):
        t = timesteps[i]
        
        latent_model_input = jt.cat([latents] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        
        v_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        t_cur = max(1, t.item() - (1000 // arguments.step_num))
        t_nxt = t
        a_cur = pipeline.scheduler.alphas_cumprod[t_cur]
        a_nxt = pipeline.scheduler.alphas_cumprod[t_nxt]

        noise_pred = a_cur.sqrt() * v_pred.chunk(2)[0] + (1.0 - a_cur).sqrt() * latents
        latents = a_nxt.sqrt() * (latents - (1.0 - a_cur).sqrt() * noise_pred) / a_cur.sqrt() + (1.0 - a_nxt).sqrt() * noise_pred
        
    return latents

@torch.no_grad()
def ddim_reversion(start_latents: jt.Var, device=device) -> jt.Var:
    dict_attn = {}
    attn_processors = pipeline.unet.attn_processors
    j = 0
    for i in attn_processors.keys():
        if i in list_attn:
            dict_attn[i] = StyleInjectAttnProcessor(_sty_ft=sty_ft[j], _cnt_ft=cnt_ft[j])
            j += 1
        else:
            dict_attn[i] = AttnProcessor()
    pipeline.unet.set_attn_processor(dict_attn)
    
    timesteps = pipeline.scheduler.timesteps
    
    latents = start_latents.clone()
    text_embeddings = pipeline._encode_prompt(prompt='', device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    
    for i in tqdm(range(0, arguments.step_num), desc='DDIM Reversion', total=arguments.step_num):
        t = timesteps[i]
        
        latent_model_input = jt.cat([latents] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        
        v_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        prev_t = max(0, t.item() - (1000 // arguments.step_num))
        a_cur = pipeline.scheduler.alphas_cumprod[t.item()]
        a_pre = pipeline.scheduler.alphas_cumprod[prev_t]
        
        noise_pred = a_cur.sqrt() * v_pred.chunk(2)[0] + (1.0 - a_cur).sqrt() * latents
        pred_x0 = (latents - (1 - a_cur).sqrt() * noise_pred) / a_cur.sqrt()
        drc_xt = (1 - a_pre).sqrt() * noise_pred
        
        latents = a_pre.sqrt() * pred_x0 + drc_xt
        
    return latents

def adain(sty_latent: jt.Var, cnt_latent: jt.Var) -> jt.Var:
    sty_np = np.asarray(sty_latent)
    cnt_np = np.asarray(cnt_latent)

    sty_mean = sty_np.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)
    cnt_mean = cnt_np.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)
    sty_std = sty_np.std(axis=2, keepdims=True).mean(axis=3, keepdims=True)
    cnt_std = cnt_np.std(axis=2, keepdims=True).mean(axis=3, keepdims=True)

    sty_mean = jt.Var(sty_mean)
    cnt_mean = jt.Var(cnt_mean)
    sty_std = jt.Var(sty_std)
    cnt_std = jt.Var(cnt_std)

    return sty_std * (cnt_latent - cnt_mean) / cnt_std + sty_mean

def extract_features(img_dir: str, is_sty_img) -> list:
    dict_attn = {}
    for i in pipeline.unet.attn_processors.keys():
        if i in list_attn:
            dict_attn[i] = SaveFeatureAttnProcessor(is_sty_img=is_sty_img)
        else:
            dict_attn[i] = AttnProcessor()
    pipeline.unet.set_attn_processor(dict_attn)
    latent = ddim_inversion(img=load_img(img_dir))

    ft_lst = []
    attn_processors = pipeline.unet.attn_processors
    for i in attn_processors.keys():
        if i in list_attn:
            ft_lst.append(attn_processors[i].ft_dict)
  
    return [latent, ft_lst]

if __name__ == '__main__':
    sty_img_dir = f'{arguments.sty_dir}/{arguments.sty_idx}.png'
    cnt_imgs_dir = f'{arguments.cnt_dir}/{arguments.sty_idx}'
    out_imgs_dir = f'{arguments.out_img_dir}/{arguments.sty_idx}'
    
    if not os.path.exists(sty_img_dir):
        print(f'Style image {sty_img_dir} not found.')
        exit()
    if not os.path.exists(cnt_imgs_dir):
        print(f'Content images dictionary {cnt_imgs_dir} not found.')
        exit()
    if not os.path.isdir(cnt_imgs_dir):
        print(f'Path {cnt_imgs_dir} is not a dictionary.')
        exit()
    if not os.path.exists(out_imgs_dir):
        os.mkdir(out_imgs_dir)

    pipeline.scheduler.set_timesteps(arguments.step_num, device=device)
    timesteps = jt.Var(np.flip(pipeline.scheduler.timesteps))
    print(f'DDIM time steps: {timesteps.tolist()}')

    sty_latent, sty_ft = extract_features(img_dir=sty_img_dir, is_sty_img=True)
    for cnt_img_dir in os.listdir(cnt_imgs_dir):
        cnt_latent, cnt_ft = extract_features(img_dir=f'{cnt_imgs_dir}/{cnt_img_dir}', is_sty_img=False)
        
        init_noise = adain(sty_latent=sty_latent, cnt_latent=cnt_latent)
        
        styled_latent = ddim_reversion(start_latents=init_noise)
        
        styled_image = pipeline.decode_latents(styled_latent[-1].unsqueeze(0))
        pipeline.numpy_to_pil(styled_image)[0].save(f'{out_imgs_dir}/{cnt_img_dir}')
        print(f'Styled image saved to {out_imgs_dir}/{cnt_img_dir}.')