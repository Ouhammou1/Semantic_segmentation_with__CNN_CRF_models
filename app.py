# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, send_file
import os
import uuid
from inference import run_inference

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename    = uuid.uuid4().hex + '.jpg'
    input_path  = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(UPLOAD_FOLDER, 'seg_' + filename)
    file.save(input_path)

    use_crf = request.form.get('use_crf', 'true') == 'true'

    try:
        result_path, detected = run_inference(input_path, output_path, use_crf=use_crf)
        return jsonify({
            'result_url': '/result/' + os.path.basename(output_path),
            'classes': detected
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result/<filename>')
def result(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


if __name__ == '__main__':
    app.run(debug=True)# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, send_file
import os
import uuid
from inference import run_inference

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename    = uuid.uuid4().hex + '.jpg'
    input_path  = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(UPLOAD_FOLDER, 'seg_' + filename)
    file.save(input_path)

    use_crf = request.form.get('use_crf', 'true') == 'true'

    try:
        result_path, detected = run_inference(input_path, output_path, use_crf=use_crf)
        return jsonify({
            'result_url': '/result/' + os.path.basename(output_path),
            'classes': detected
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result/<filename>')
def result(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))


if __name__ == '__main__':
    app.run(debug=True)