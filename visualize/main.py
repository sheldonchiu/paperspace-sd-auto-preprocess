from flask import Flask, render_template, request, jsonify, send_from_directory
from math import ceil
import os
import json

app = Flask(__name__)

IMAGE_DIR = '/Users/sheldon/Documents/booru_data/0_filter'  # Replace with your path
EXCLUDE_FILE = 'exclude.json'
exclude_list = []
accepted_formats = ['.jpg', '.png', '.jpeg', '.gif']  # Add or remove formats as needed
IMAGES_PER_PAGE = 200

if os.path.isfile(EXCLUDE_FILE):
    with open(EXCLUDE_FILE, 'r') as f:
        exclude_list = json.load(f)

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1] in accepted_formats]
    # files = [f for f in files if f not in exclude_list]
    
    total_pages = ceil(len(files) / IMAGES_PER_PAGE)
    
    start = (page - 1) * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    files = files[start:end]
    
    return render_template('index.html', files=files, page=page, total_pages=total_pages, exclude_list=exclude_list)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/caption/<path:filename>')
def get_caption(filename):
    try:
        # Remove the original extension before adding .caption
        base_filename = os.path.splitext(filename)[0]
        with open(os.path.join(IMAGE_DIR, base_filename + '.caption'), 'r') as f:
            caption = f.read()
    except FileNotFoundError:
        caption = "No caption available for this image."
    return jsonify({'caption': caption})


@app.route('/exclude/<path:filepath>', methods=['POST'])
def exclude(filepath):
    if filepath not in exclude_list:
        exclude_list.append(filepath)
        with open(EXCLUDE_FILE, 'w') as f:
            json.dump(exclude_list, f)
    else:
        exclude_list.remove(filepath)
        with open(EXCLUDE_FILE, 'w') as f:
            json.dump(exclude_list, f)
    return '', 204


if __name__ == '__main__':
    app.run(debug=True)
