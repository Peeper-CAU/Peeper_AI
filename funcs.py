from openai import OpenAI
import scipy.io.wavfile as wav
import tempfile

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import json
 


def transcribe_audio(audio_data, fs):
    
    # .env 파일 로드
    load_dotenv()
    # 환경 변수에서 API 키 불러오기
    openai_api_key = os.getenv('OPENAI_API_KEY')
    OpenAI.api_key = openai_api_key
    
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
            
    if "시청" in transcription.text and "감사" in transcription.text:
        return "대화 없음"
    
    return transcription.text

def get_risk_level(count):
    if count == 1:
        return "1단계 의심"
    elif count == 2:
        return "2단계 주의"
    elif count == 3:
        return "3단계 경계"
    elif count >= 4:
        return "4단계 심각"
    else:
        return "일반 전화"


def voicePhishingAnalysis(dialog_text, model_name = 'gemini-1.5-pro-latest'):
    
    load_dotenv()
    # 환경 변수에서 API 키 불러오기
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    # OpenAI API 키 설정
    genai.configure(api_key=gemini_api_key)
    
    model = genai.GenerativeModel(model_name,
                                  generation_config={"response_mime_type": "application/json"})
    
    prompt = f"""
다음 문단은 전화 대화 상황입니다. 똑똑한 수사관의 관점에서 보이스 피싱인지 잡아내야 합니다.
대화에서 '개인 정보(주민번호 등)를 탈취하기 위해 지시를 하거나, 금전(돈, 계좌이체 등)을 요구하고 있는지를 검출해야 합니다.
잘못하면 잘못 없는 사람이 범죄자가 될 수 있으니, 확실히 보이스피싱으로 의심되는 경우에만 신중하게 'true'라고 답변해야합니다.

[대화]
{dialog_text}

Using this JSON schema:
    response = {{"SuspectedVoicephishing": True/False}}
Return a `response`
"""
    response = model.generate_content(
        [prompt],
        safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        ,request_options= {"timeout": 10}
    )
    
    return json.loads(response.text)['SuspectedVoicephishing']
