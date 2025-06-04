from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io
from rembg import remove  # rembg 라이브러리 임포트

app = FastAPI()

# 1. TFLite 모델 로드
model_path = "final_model.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 📌 2. train_generator.class_indices 사용
# 실제 학습 시 사용한 class_indices로 업데이트해야 합니다.
# 예시로 아래는 학습 시 사용한 class_indices와 같은 형태로 정의한 것입니다.
# 이 부분은 실제 모델 학습 시 train_generator에서 class_indices를 가져와야 합니다.

class_indices = {
    '가지볶음': 0, '간장게장': 1, '갈비구이': 2, '갈비찜': 3, '갈비탕': 4, '갈치구이': 5, '갈치조림': 6, '감자전': 7, '감자조림': 8, '감자채볶음': 9, '감자탕': 10, '갓김치': 11, '건새우볶음': 12, '경단': 13, '계란국': 14, '계란말이': 15, '계란찜': 16, '계란후라이': 17, '고등어구이': 18, '고등어조림': 19, '고사리나물': 20, '고추장진미채볶음': 21, '고추튀김': 22, '곰탕_설렁탕': 23, '곱창구이': 24, '곱창전골': 25, '과메기': 26, '김밥': 27, '김치볶음밥': 28, '김치전': 29, '김치찌개': 30, '김치찜': 31, '깍두기': 32, '깻잎장아찌': 33, '꼬막찜': 34, '꽁치조림': 35, '꽈리고추무침': 36, '꿀떡': 37, '나박김치': 38, '누룽지': 39, '닭갈비': 40, '닭계장': 41, '닭볶음탕': 42, '더덕구이': 43, '도라지무침': 44, '도토리묵': 45, '동그랑땡': 46, '동태찌개': 47, '된장찌개': 48, '두부김치': 49, '두부조림': 50, '땅콩조림': 51, '떡갈비': 52, '떡국_만두국': 53, '떡꼬치': 54, '떡볶이': 55, '라면': 56, '라볶이': 57, '막국수': 58, '만두': 59, '매운탕': 60, '멍게': 61, '메추리알장조림': 62, '멸치볶음': 63, '무국': 64, '무생채': 65, '물냉면': 66, '물회': 67, '미역국': 68, '미역줄기볶음': 69, '배추김치': 70, '백김치': 71, '보쌈': 72, '부추김치': 73, '북엇국': 74, '불고기': 75, '비빔냉면': 76, '비빔밥': 77, '산낙지': 78, '삼겹살': 79, '삼계탕': 80, '새우볶음밥': 81, '새우튀김': 82, '생선전': 83, '소세지볶음': 84, '송편': 85, '수육': 86, '수정과': 87, '수제비': 88, '숙주나물': 89, '순대': 90, '순두부찌개': 91, '시금치나물': 92, '시래기국': 93, '식혜': 94, '알밥': 95, '애호박볶음': 96, '약과': 97, '약식': 98, '양념게장': 99, '양념치킨': 100, '어묵볶음': 101, '연근조림': 102, '열무국수': 103, '열무김치': 104, '오이소박이': 105, '오징어채볶음': 106, '오징어튀김': 107, '우엉조림': 108, '유부초밥': 109, '육개장': 110, '육회': 111, '잔치국수': 112, '잡곡밥': 113, '잡채': 114, '장어구이': 115, '장조림': 116, '전복죽': 117, '젓갈': 118, '제육볶음': 119, '조개구이': 120, '조기구이': 121, '족발': 122, '주꾸미볶음': 123, '주먹밥': 124, '짜장면': 125, '짬뽕': 126, '쫄면': 127, '찜닭': 128, '총각김치': 129, '추어탕': 130, '칼국수': 131, '코다리조림': 132, '콩국수': 133, '콩나물국': 134, '콩나물무침': 135, '콩자반': 136, '파김치': 137, '파전': 138, '편육': 139, '피자': 140, '한과': 141, '해물찜': 142, '호박전': 143, '호박죽': 144, '홍어무침': 145, '황태구이': 146, '회무침': 147, '후라이드치킨': 148, '훈제오리': 149
}


classes = list(class_indices.keys())

# 3. 이미지 전처리 함수
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # 모델 입력 크기 맞춤
    image = np.array(image, dtype=np.float32) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# 4. FastAPI 예측 엔드포인트
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 1) 업로드된 이미지 읽기
    input_bytes = await file.read()
    image = Image.open(io.BytesIO(input_bytes)).convert("RGB")

    # 2) rembg로 배경 제거 (배경 투명 처리)
    img_no_bg = remove(image)

    # 3) rembg 처리된 이미지를 RGB 모드로 변환 (rembg는 RGBA 반환)
    img_no_bg = img_no_bg.convert("RGB")

    # 4) 이미지 전처리
    input_data = preprocess_image(img_no_bg)

    # 5) 모델 예측 실행
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 6) 결과 처리
    predicted_index = np.argmax(output_data)
    predicted_class = classes[predicted_index]
    confidence = float(output_data[0][predicted_index])
    predicted_id = class_indices[predicted_class]

    return {"id": predicted_id, "class": predicted_class, "confidence": confidence}
