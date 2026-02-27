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

def run_florence2(image_path, task_prompt="<OCR>"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
    
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      do_sample=False,
      num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer[task_prompt]

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
            # We use <OCR> to pull all text from the label
            extracted_text = run_florence2(filepath, task_prompt="<OCR>")
            
            # Format output (Florence sometimes returns a list or string depending on task)
            if isinstance(extracted_text, list):
               extracted_text = " ".join(extracted_text)
            
            # Save the result
            result_filename = f"{os.path.splitext(filename)[0]}_ingredients.txt"
            result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
            with open(result_filepath, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
                
            return jsonify({
                'success': True,
                'text': extracted_text,
                'saved_file': result_filepath
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
