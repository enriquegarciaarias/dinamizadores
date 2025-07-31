from sources.common.common import logger, processControl, log_
from sources.common.utils import huggingface_login
import torch

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

def processLLAVA(imageList, device="cuda" if torch.cuda.is_available() else "cpu"):
    # 📌 Cargar el modelo LLaVA
    huggingface_login()


    model_name = "liuhaotian/llava-v1.5-13b"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 📌 Generar captions para cada imagen
    captions = {}
    for img in imageList:
        image = Image.open(img["path"]).convert("RGB")
        prompt = "Describe this image."

        # ✅ Correct input processing
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=50)

        # ✅ Decode output correctly
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        captions[img["name"]] = caption
        print(f"📷 {img['name']} → {caption}")

    # 📌 Save captions to a file
    with open("generated_captions.txt", "w") as f:
        for name, caption in captions.items():
            f.write(f"{name}: {caption}\n")

    print("✅ Captions generated and saved in 'generated_captions.txt'")
