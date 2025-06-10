# 1. Python 3.11 베이스 이미지 사용
FROM python:3.11.3

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 앱 소스 복사
COPY . .

# 5. 모델 파일은 필요 시에만 수동으로 서버에 배포
# (혹은 CI/CD 파이프라인에서 모델만 따로 받아오기)

# 6. Uvicorn 실행 명령어 (main.py 안에 FastAPI 인스턴스가 `app`일 때)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
