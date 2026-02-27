import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model Configuration
MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_ID} on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Use eager attention for CPU to avoid SDPA/Flash-Attention issues on older processors
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if DEVICE == "cpu":
    config.attn_implementation = "eager"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    config=config,
    trust_remote_code=True
).to(DEVICE)
print("Model loaded successfully!")

import re

def run_florence2(image, task_prompt="<OCR>", text_input=None):
    if text_input:
        # Refined Florence-2 VQA prompt format
        prompt = task_prompt + text_input
    else:
        prompt = task_prompt
        
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      do_sample=False,
      num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    
    result = parsed_answer[task_prompt]
    
    # Cleanup: Remove pesky <loc_> and <poly> tags and prompt echoing if it happens
    if isinstance(result, str):
        result = re.sub(r'<(loc_\d+|poly|/poly)>', '', result)
        result = result.replace(prompt, '').strip()
        # Remove Florence task prefixes if they leak
        result = re.sub(r'^(VQA|OCR)>?\s*', '', result, flags=re.IGNORECASE)
        
    return result

def fallback_parse(text):
    """Keyword-based fallback to extract sections from raw OCR text."""
    ingredients = "Section not detected. Check 'Raw OCR' below."
    nutrition = "Section not detected. Check 'Raw OCR' below."
    
    # 1. Clean the text slightly for better splitting
    clean_text = re.sub(r'\s+', ' ', text)
    
    # 2. Match Ingredients
    # Keywords: Ingredients, Składniki, Inhaltsstoffe, Zutaten, etc.
    ing_pattern = r'(INGREDIENTS?|SKŁADNIKI|ZUTATEN|INHALTSSTOFFE|KOMPOZYCJA)[:\s]+'
    ing_matches = re.split(ing_pattern, clean_text, flags=re.IGNORECASE)
    if len(ing_matches) >= 3:
        content = ing_matches[2]
        stop_keywords = r'NUTRITION|FACTS|WARTOŚĆ|ODŻYWCZA|STORAGE|WAŻNOŚĆ|BEST BEFORE|PRODUCED|NET|ALERGEN'
        ingredients = re.split(stop_keywords, content, flags=re.IGNORECASE)[0].strip()

    # 3. Match Nutrition
    # Keywords: Nutrition Information, Nutrition Facts, Nutrition, Wartość odżywcza, etc.
    nut_pattern = r'(NUTRITION(\s+INFORMATION|\s+FACTS)?|WARTOŚĆ\s+ODŻYWCZA|NÄHRWERTE|VALORES\s+NUTRICIONALES)[:\s]+'
    nut_matches = re.split(nut_pattern, clean_text, flags=re.IGNORECASE)
    if len(nut_matches) >= 3:
        nutrition = nut_matches[len(nut_matches)-1].strip()
        
    return ingredients, nutrition

def is_vqa_hallucination(text):
    """Heuristic to check if the VQA output is junk or hallucination."""
    if not text: return True
    # If it contains prompt fragments or common Florence junk
    junk = ['what are', 'vqa>', 'question:', '<loc_', 'not sure', 'i don\'t know', 'list the', 'extract the']
    low_text = text.lower()
    for j in junk:
        if j in low_text: return True
    # If it's just repeating the question or very short repetition
    if (('ingredients' in low_text or 'nutrition' in low_text) and len(text) < 40 and 
        'water' not in low_text and 'kcal' not in low_text and 'fat' not in low_text):
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image = Image.open(filepath).convert("RGB")
            
            # 1. OCR for general text (The most reliable source)
            full_text = run_florence2(image, task_prompt="<OCR>")
            if isinstance(full_text, list): full_text = " ".join(full_text)
            
            # 2. Try VQA for Ingredients
            ingredients_ai = run_florence2(image, task_prompt="<DocVQA>", text_input="What are the ingredients?")
            
            # 3. Try VQA for Nutrition
            nutrition_ai = run_florence2(image, task_prompt="<DocVQA>", text_input="List the nutrition facts.")
            
            # 4. Smart Merge / Fallback
            fb_ingredients, fb_nutrition = fallback_parse(full_text)
            
            # Decide what to show for ingredients
            if is_vqa_hallucination(ingredients_ai):
                ingredients = fb_ingredients
            else:
                ingredients = ingredients_ai

            # Decide what to show for nutrition
            if is_vqa_hallucination(nutrition_ai):
                nutrition = fb_nutrition
            else:
                nutrition = nutrition_ai

            # Final check: if both failed to find something specific, we at least show the fallback
            if ingredients == "Section not detected. Check 'Raw OCR' below." and fb_ingredients != "Section not detected. Check 'Raw OCR' below.":
                ingredients = fb_ingredients
            if nutrition == "Section not detected. Check 'Raw OCR' below." and fb_nutrition != "Section not detected. Check 'Raw OCR' below.":
                nutrition = fb_nutrition

            # Save the result
            result_filename = f"{os.path.splitext(filename)[0]}_structured.txt"
            result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
            with open(result_filepath, 'w', encoding='utf-8') as f:
                f.write(f"--- FULL OCR ---\n{full_text}\n\n")
                f.write(f"--- INGREDIENTS ---\n{ingredients}\n\n")
                f.write(f"--- NUTRITION ---\n{nutrition}\n")
                
            return jsonify({
                'success': True,
                'full_text': full_text,
                'ingredients': ingredients,
                'nutrition': nutrition,
                'saved_file': result_filepath
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
