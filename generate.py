import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from transformers import pipeline, FSMTForConditionalGeneration, FSMTTokenizer
from PIL import Image
import torch
import numpy as np


# translation function (eng to ru)
def translation_wrapper(text: str):
    input_ids = translation_tokenizer.encode(text, return_tensors="pt")
    outputs = translation_model.generate(input_ids.to(device))
    decoded = translation_tokenizer.decode(outputs[0].float(), skip_special_tokens=True)
    return decoded

# prepare models:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translation_model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-ru", torch_dtype=torch.float16).half().to(device)
translation_tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-ru")

dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)

# pipeline utils:
realesrgan = get_realesrgan('x2', device=device)
clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=4)

text = 'a city skyline on a cloudy night'
translated = translation_wrapper(text)

seed_everything(42)
pil_images = []
scores = []
for top_k, top_p, images_num in [
    (2048, 0.995, 6),
]:
    _pil_images, _scores = generate_images(translated, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, bs=4, top_p=top_p)
    pil_images += _pil_images
    scores += _scores
#show(pil_images)

# cherry-pick images using ruCLIP
top_images, clip_scores = cherry_pick_by_ruclip(pil_images, translated, clip_predictor, count=6)
#show(top_images, 3)

# super resolution
sr_images = super_resolution(top_images, realesrgan)
show(sr_images, 3, save_dir="./pics")