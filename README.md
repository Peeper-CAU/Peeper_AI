# Peeper_AI

1. 환경 설정
먼저 Conda 환경을 생성하고 활성화한 후 필요한 패키지를 설치합니다.

```
# Conda 환경 생성
conda create -n peeper python=3.10

# Conda 환경 활성화
conda activate peeper

# 필요한 패키지 설치
pip install -r requirements.txt
```

2. Flask 서버 실행
Flask 서버를 실행하여 음성 파일 분석 API를 사용할 수 있습니다.

```
# Flask 서버 실행
flask run
```

3. 데모 페이지 실행
Streamlit을 사용하여 데모 페이지를 실행하고 웹 인터페이스를 통해 API를 테스트할 수 있습니다.

```
# Streamlit 데모 페이지 실행
streamlit run streamlit_app.py
```
