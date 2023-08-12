import os
import json
from math import ceil

import click
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

IMAGE_DIR = None
EXCLUDE_FILE = None
EXCLUDE_LIST = []
IMAGES_PER_PAGE = 200
accepted_formats = ['.jpg', '.png', '.jpeg', '.gif']  # Add or remove formats as needed

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1] in accepted_formats]
    
    total_pages = ceil(len(files) / IMAGES_PER_PAGE)
    
    start = (page - 1) * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    files = files[start:end]
    
    return render_template('index.html', files=files, page=page, total_pages=total_pages, exclude_list=EXCLUDE_LIST)

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
    if filepath not in EXCLUDE_LIST:
        EXCLUDE_LIST.append(filepath)
        with open(EXCLUDE_FILE, 'w') as f:
            json.dump(EXCLUDE_LIST, f)
    else:
        EXCLUDE_LIST.remove(filepath)
        with open(EXCLUDE_FILE, 'w') as f:
            json.dump(EXCLUDE_LIST, f)
    return '', 204

@click.command()
@click.option(
    '--image_dir',
    type=str,
    help='Path to the image directory',
    prompt='Enter the path to the image directory:'
)
@click.option(
    '--exclude-file',
    type=str,
    help='Path to exclude.json',
    prompt='Enter the path to exclude.json:'
)
@click.option(
    '--images-per-page',
    type=int,
    default=200,
    help='Number of images per page',
    prompt='Enter the number of images per page:'
)
@click.option(
    '--port',
    type=int,
    default=8000,
    help='Port number',
    prompt='Enter the port number:'
)
def main(image_dir, exclude_file, images_per_page, port):
    """
    Main function to run the application.
    
    Args:
        image_dir (str): Path to the image directory.
        exclude_file (str): Path to exclude.json file.
        images_per_page (int): Number of images per page.
    """
    global IMAGE_DIR, EXCLUDE_FILE, IMAGES_PER_PAGE, EXCLUDE_LIST
    IMAGE_DIR = image_dir
    EXCLUDE_FILE = exclude_file
    IMAGES_PER_PAGE = images_per_page
    
    if os.path.isfile(EXCLUDE_FILE):
        with open(EXCLUDE_FILE, 'r') as f:
            EXCLUDE_LIST = json.load(f)
    
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
