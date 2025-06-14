# 1. Python 3.11.3 베이스 이미지 사용
FROM python:3.11.3

# 2. 서울 시간대 설정
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone

# 3. 작업 디렉토리 생성
WORKDIR /app

# 4. requirements.txt 먼저 복사하고 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 앱 전체 복사
COPY . .

# 6. GCS에서 TFLite 모델 자동 다운로드 (main.py 내부에서 처리 가정)

# 7. FastAPI 앱 실행 (uvicorn)
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

