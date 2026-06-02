# -*- coding: utf-8 -*-
"""
智能车辆检测分析系统 - Flask Web 应用 (v3)
后端一次性预生成 5 个图层视频，前端切换 video src 即可
"""

import os
import uuid
import shutil
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from src.config import CONFIG
from src.video_processor import VideoProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RES_FOLDER'] = os.path.join('static', 'res')
app.config['MAX_CONTENT_LENGTH'] = CONFIG['max_upload_mb'] * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RES_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html', config=CONFIG)


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video']
    if file.filename == '' or not file:
        return redirect(url_for('index'))

    uid = str(uuid.uuid4())[:8]
    res_dir = os.path.join(app.config['RES_FOLDER'], uid)
    os.makedirs(res_dir, exist_ok=True)

    # 保存上传文件
    input_path = os.path.join(res_dir, 'input.mp4')
    file.save(input_path)

    processor = VideoProcessor()
    try:
        total_frames, elapsed, summary, video_paths = processor.process_video(
            input_path, res_dir
        )
    except Exception as e:
        return render_template('result.html',
                               uid=uid, video_paths={}, summary={},
                               error=str(e), config=CONFIG)

    return render_template('result.html',
                           uid=uid,
                           video_paths=video_paths,
                           summary=summary,
                           error=None, config=CONFIG)


@app.route('/static/res/<uid>/<filename>')
def res_file(uid, filename):
    return send_from_directory(os.path.join(app.config['RES_FOLDER'], uid), filename)


@app.route('/reset/<uid>', methods=['POST'])
def reset(uid):
    """删除指定任务的资源目录"""
    res_dir = os.path.join(app.config['RES_FOLDER'], uid)
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
