import streamlit as st
from openai import OpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import time

import threading
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import json


# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 불러오기
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# OpenAI API 키 설정
OpenAI.api_key = openai_api_key
genai.configure(api_key=gemini_api_key)

def save_prompt_performance(prompts, file_path='prompt_performance.json'):
    with open(file_path, 'w') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)

def load_prompt_performance(file_path='prompt_performance.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return []

def update_prompt_performance(new_prompt, f1_score, new_scores, file_path='prompt_performance.json'):
    prompts = load_prompt_performance(file_path)
    prompts.append({
        'prompt': new_prompt,
        'f1_score': f1_score,
        'scores': new_scores
    })
    save_prompt_performance(prompts, file_path)

def rank_prompts(prompts):
    sorted_prompts = sorted(prompts, key=lambda x: x['f1_score'], reverse=True)
    return sorted_prompts



def record_audio(duration, fs):
    # 마이크로부터 오디오를 캡처합니다.
    st.info("녹음 중... 음성을 입력하세요.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(duration):
        time.sleep(1)
        progress_bar.progress((i + 1) / duration)
        status_text.text(f"남은 시간: {duration - (i + 1)} 초")
    
    sd.wait()  # 녹음이 끝날 때까지 기다립니다.
    progress_bar.empty()
    status_text.empty()
    
    return recording


def voicePhishingAnalysis(dialog_text, model_name, result_list, index):
    model = genai.GenerativeModel(model_name)
    
    prompt = """
다음 문단은 전화 대화 상황입니다. 똑똑한 수사관의 관점에서 보이스 피싱인지 잡아내야 합니다.
대화에서 '개인 정보(주민번호 등)를 탈취하기 위해 지시를 하거나, 금전(돈, 계좌이체 등)을 요구하고 있는지를 검출해야 합니다.
반드시 '예'/'아니요'로만 답변해야합니다.
잘못하면 잘못 없는 사람이 범죄자가 될 수 있으니, 확실히 보이스피싱으로 의심되는 경우에만 신중하게 '예'라고 답변해야합니다.

[대화]"""
    response = model.generate_content(
        [prompt + dialog_text],
        safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        ,request_options= {"timeout": 60}
    )
    result_list[index] = response.text
    time.sleep(1)

def transcribe_audio(audio_data, fs):
    # 오디오 데이터를 파일로 저장합니다.
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        wav.write(tmpfile.name, fs, audio_data)
        tmpfile.seek(0)

        client = OpenAI(api_key = OpenAI.api_key)
        # 오디오 파일을 텍스트로 변환합니다.
        with open(tmpfile.name, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
    
    return transcription.text


def load_texts_from_excel(file_path, column_name='text'):
    # Excel 파일을 DataFrame으로 읽어오기
    df = pd.read_excel(file_path, engine='openpyxl')
    # 특정 열(column) 불러오기
    column_data = df[column_name]
    return column_data

def promptTestAnalysis(prompt, text, model_name):
    model = genai.GenerativeModel(model_name)
    
    full_prompt = f"""
    {prompt}
    [대화]
    {text}
    """
    
    response = model.generate_content(
        [full_prompt],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    return response.text

def classify_text(text):
    # 텍스트를 판별하는 함수
    temp = text
    if len(temp) > 20:
        temp = temp[:20]
    
    if "예" in temp:
        return "보이스 피싱 의심됨"
    else:
        return "보이스 피싱 아님"

# 사이드바에 페이지 선택을 위한 메뉴 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Voice Phishing Detector", "Prompt Performance Test", "Prompt Ranking Board"])

if page == "Voice Phishing Detector":
    st.title("실시간 오디오 보이스 피싱 판별기")

    col11, col22 = st.columns(2)

    with col11:
        # 모델 선택
        model_name_left = st.selectbox(
            "왼쪽 모델 선택",
            ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', 'gemini-1.0-pro-latest'],
            key="left_model"
        )

    with col22:
        model_name_right = st.selectbox(
            "오른쪽 모델 선택",
            ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', 'gemini-1.0-pro-latest'],
            key="right_model"
        )

    # '녹음 시간' 슬라이더를 상단에 위치
    duration = st.slider("녹음 시간 (초)", min_value=1, max_value=10, value=5)
    fs = 44100  # 샘플링 주파수

    audio_data = None
    upload = st.file_uploader("음성 파일 업로드", type=["wav"])

    if upload is not None:
        # 업로드된 오디오 파일 처리
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(upload.read())
            tmpfile.seek(0)
            fs, audio_data = wav.read(tmpfile.name)

    if st.button("녹음 시작"):
        with st.spinner('녹음 중...'):
            audio_data = record_audio(duration, fs)

    if audio_data is not None:
        with st.spinner('오디오 처리 중...'):
            start = time.time()
            
            # 오디오 데이터 텍스트 변환
            transcription_text = transcribe_audio(audio_data, fs)
            st.write("변환된 텍스트:", transcription_text)

            # 두 개의 모델을 동시에 테스트
            results = [None, None]
            thread_left = threading.Thread(target=voicePhishingAnalysis, args=(transcription_text, model_name_left, results, 0))
            thread_right = threading.Thread(target=voicePhishingAnalysis, args=(transcription_text, model_name_right, results, 1))

            thread_left.start()
            thread_right.start()

            thread_left.join()
            thread_right.join()

            gemini_response_left = results[0]
            gemini_response_right = results[1]

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"왼쪽 모델 ({model_name_left})")
                st.write(f"제미니 답변:", gemini_response_left)
                result_left = classify_text(gemini_response_left)
                st.write("판별 결과:", result_left)
            
            with col2:
                st.write(f"오른쪽 모델 ({model_name_right})")
                st.write(f"제미니 답변:", gemini_response_right)
                result_right = classify_text(gemini_response_right)
                st.write("판별 결과:", result_right)
            
            end = time.time()
            st.write(f"처리 시간: {end - start:.2f} 초")

elif page == "Prompt Performance Test":
    st.title("프롬프트 성능 테스트")

    # 프롬프트 입력 받기
    user_prompt = st.text_area("프롬프트 입력", value="", height=200)

    # 모델 선택
    model_name = st.selectbox(
        "모델 선택",
        ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', 'gemini-1.0-pro-latest']
    )

    # Excel 파일 업로드
    uploaded_file = st.file_uploader("Excel 파일 업로드", type=["xlsx"])

    if uploaded_file is not None and user_prompt:
        # Excel 파일에서 텍스트 불러오기
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        column_data = df['text']
        labels = df['label']

        results = []
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        with st.spinner('프롬프트 성능 테스트 중...'):
            progress_bar = st.progress(0)
            total_texts = len(column_data)
            
            for i, text in enumerate(column_data):
                response = promptTestAnalysis(user_prompt, text, model_name)
                classified_result = classify_text(response)
                predicted_label = 1 if classified_result == "보이스 피싱 의심됨" else 0
                actual_label = labels[i]
                
                results.append({
                    '텍스트': text,
                    '응답': response,
                    '예측': predicted_label,
                    '실제': actual_label
                })

                # 성능 평가를 위한 카운팅
                if predicted_label == 1:
                    if actual_label == 1:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    if actual_label == 0:
                        true_negative += 1
                    else:
                        false_negative += 1

                progress_bar.progress((i + 1) / total_texts)

        progress_bar.empty()

        # 성능 지표 계산
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        accuracy = (true_positive + true_negative) / total_texts if total_texts > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 0일 때와 1일 때의 정답률 계산
        accuracy_0 = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
        accuracy_1 = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

        # 결과를 데이터프레임으로 변환하여 표시
        results_df = pd.DataFrame(results)
        st.write("결과:")
        st.dataframe(results_df)

        # 성능 지표 표시
        st.write("성능 지표:")
        metrics_df = pd.DataFrame({
            "지표": ["Precision", "Recall", "Accuracy", "F1 Score", "True Positives", "False Positives", "True Negatives", "False Negatives"],
            "값": [precision, recall, accuracy, f1_score, true_positive, false_positive, true_negative, false_negative]
        })
        st.table(metrics_df)

        # 그래프 그리기
        accuracy_data = {
            '정답률': [accuracy_0, accuracy_1]
        }
        accuracy_df = pd.DataFrame(accuracy_data, index=['정답이 0일 때', '정답이 1일 때'])
        st.bar_chart(accuracy_df)

        # 프롬프트 성능 저장
        new_scores = {
            'file': uploaded_file.name,
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'accuracy': round(accuracy, 2),
            'total_texts': total_texts,
            'true_positive': true_positive,
            'false_positive': false_positive,
            'true_negative': true_negative,
            'false_negative': false_negative,
            'model_name': model_name
        }
        update_prompt_performance(user_prompt, f1_score, new_scores)

# 랭킹 보드 페이지 추가
elif page == "Prompt Ranking Board":
    st.title("프롬프트 랭킹 보드")

    # 프롬프트 성능 불러오기
    prompts = load_prompt_performance()

    if prompts:
        ranked_prompts = rank_prompts(prompts)

        # 랭킹 보드 표시
        st.write("랭킹 보드:")
        rank_df = pd.DataFrame(ranked_prompts)
        rank_df['prompt'] = '생략'

        st.table(rank_df)

        # 순위에 따라 프롬프트 선택
        prompt_options = [str(num)+': '+item['prompt'] for num, item in enumerate(ranked_prompts)]
        selected_prompt = st.selectbox("프롬프트 선택", prompt_options)
        st.text_area("선택된 프롬프트", value=selected_prompt, height=200)
    else:
        st.write("저장된 프롬프트 성능 데이터가 없습니다.")
