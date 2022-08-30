import ruclip
import argparse
import random
import string
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from transformers import pipeline, FSMTForConditionalGeneration, FSMTTokenizer
from PIL import Image
import torch
import numpy as np


parser = argparse.ArgumentParser(description="This script generates image based on the prompt you give using ruDALL-E Malevich")
parser.add_argument(
    '--prompt', 
    type=str, 
    default="a city skyline on a cloudy night",
    help="Prompt to generate an image of"
)
parser.add_argument(
    '--output', 
    type=str, 
    default="./output",
    help="Path to save images to"
)
parser.add_argument(
    '--show', 
    action='store_true',
    help="Show saved images in a window"
)
args = parser.parse_args()

# translation function (eng to ru)
def translation_wrapper(text: str):
    input_ids = translation_tokenizer.encode(text, return_tensors="pt")
    outputs = translation_model.generate(input_ids.to(device))
    decoded = translation_tokenizer.decode(outputs[0].float(), skip_special_tokens=True)
    return decoded


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

print("[INFO]: Generating '{}'".format(args.prompt))

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

translated = translation_wrapper(args.prompt)

print("[INFO]: Translated '{}' to '{}'".format(args.prompt, translated))

seed_everything(42)
pil_images = []
scores = []
for top_k, top_p, images_num in [
    (2048, 0.995, 6),
]:
    _pil_images, _scores = generate_images(translated, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, bs=4, top_p=top_p)
    pil_images += _pil_images
    scores += _scores

# cherry-pick images using ruCLIP
top_images, clip_scores = cherry_pick_by_ruclip(pil_images, translated, clip_predictor, count=6)

# super resolution
sr_images = super_resolution(top_images, realesrgan)

if args.show:
    show(sr_images, 3, save_dir=args.output)
else:
    for i in sr_images:
        random_str = get_random_string(6)
        i.save("{}/image-{}.jpg".format(args.output, random_str))