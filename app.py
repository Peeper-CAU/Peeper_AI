from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from funcs import transcribe_audio, voicePhishingAnalysis, get_risk_level

import scipy.io.wavfile as wav
import numpy as np
import tempfile
import os


app = Flask(__name__)
CORS(app, resources={r'*': {'origins': ["https://peeper.dev-lr.com/"]}})

# 전역 변수 설정
calls_data = {}

@app.route('/wavAnalysis', methods=['POST'])
def wav_analysis():
    if 'file' not in request.files or 'uid' not in request.form:
        return jsonify({'error': 'No file part or UID part'}), 400

    file = request.files['file']
    uid = request.form['uid']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if uid not in calls_data:
        calls_data[uid] = {'file_counter': 0, 'temp_files': [], 'phishing_count': 0}

    call_info = calls_data[uid]
    call_info['file_counter'] += 1

    # 홀수 번째 요청: 파일 저장
    if call_info['file_counter'] % 2 != 0:
        temp_file_path = tempfile.mktemp(suffix='.wav')
        file.save(temp_file_path)
        call_info['temp_files'].append(temp_file_path)
        return jsonify({
            'message': 'File saved successfully', 
            'messageSending': False,
            'riskLevel': get_risk_level(call_info['phishing_count']),
            'part': call_info['file_counter'], 
            'uid': uid
            })

    # 짝수 번째 요청: 파일 결합 및 변환
    else:
        # 이전 파일과 현재 파일 결합
        fs1, audio_data1 = wav.read(call_info['temp_files'][-1])
        fs2, audio_data2 = wav.read(file)

        if fs1 != fs2:
            return jsonify({'error': 'Sampling rates do not match'}), 400

        combined_audio = np.concatenate((audio_data1, audio_data2))
        text = transcribe_audio(combined_audio, fs1)
        is_phishing = voicePhishingAnalysis(text)

        if is_phishing:
            call_info['phishing_count'] += 1

        result = {
            'messageSending': is_phishing,
            'riskLevel': get_risk_level(call_info['phishing_count']),
            'uid': uid
        }

        # 임시 파일 삭제
        os.remove(call_info['temp_files'].pop(-1))

        # 한글 텍스트를 UTF-8로 반환
        response = make_response(jsonify(result))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response


if __name__ == '__name__':
    app.run('0.0.0.0',port=5000,debug=True)