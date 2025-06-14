from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io

app = FastAPI()

# 📌 1. .tflite 모델 로드
model_path = "final_model.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ 디버깅용 입력 텐서 shape 출력
print("👉 expected input shape from model:", input_details[0]['shape'])  # 예: (1, 3, 299, 299)

# 📌 2. train_generator.class_indices 사용
# 실제 학습 시 사용한 class_indices로 업데이트해야 합니다.
# 예시로 아래는 학습 시 사용한 class_indices와 같은 형태로 정의한 것입니다.
# 이 부분은 실제 모델 학습 시 train_generator에서 class_indices를 가져와야 합니다.

class_indices = {
'가지볶음': 0,

'간장게장': 1,

'갈비탕': 2,

'갈치구이': 3,

'감자조림': 4,

'감자채볶음': 5,

'감자탕': 6,

'갓김치': 7,

'건새우볶음': 8,

'경단': 9,

'계란국': 10,

'계란말이': 11,

'계란찜': 12,

'고등어구이': 13,

'고사리나물': 14,

'고추튀김': 15,

'곰탕_설렁탕': 16,

'곱창구이': 17,

'과메기': 18,

'김밥': 19,

'김치볶음밥': 20,

'김치전': 21,

'김치찌개': 22,

'깍두기': 23,

'깻잎장아찌': 24,

'꼬막찜': 25,

'꽈리고추무침': 26,

'꿀떡': 27,

'나박김치': 28,

'누룽지': 29,

'닭갈비': 30,

'도토리묵': 31,

'동그랑땡': 32,

'된장찌개': 33,

'두부김치': 34,

'두부조림': 35,

'땅콩조림': 36,

'떡갈비': 37,

'떡국_만두국': 38,

'떡꼬치': 39,

'떡볶이': 40,

'라면': 41,

'라볶이': 42,

'막국수': 43,

'만두': 44,

'멍게': 45,

'메추리알장조림': 46,

'멸치볶음': 47,

'무국': 48,

'무생채': 49,

'물냉면': 50,

'물회': 51,

'미역국': 52,

'미역줄기볶음': 53,

'배추김치': 54,

'백김치': 55,

'보쌈': 56,

'부추김치': 57,

'불고기': 58,

'비빔냉면': 59,

'비빔밥': 60,

'산낙지': 61,

'삼겹살': 62,

'삼계탕': 63,

'새우볶음밥': 64,

'새우튀김': 65,

'생선조림': 66,

'소세지볶음': 67,

'송편': 68,

'수정과': 69,

'숙주나물': 70,

'순대': 71,

'순두부찌개': 72,

'시금치나물': 73,

'시래기국': 74,

'식혜': 75,

'애호박볶음': 76,

'약과': 77,

'약식': 78,

'양념게장': 79,

'양념치킨': 80,

'어묵볶음': 81,

'연근조림': 82,

'열무국수': 83,

'열무김치': 84,

'오이소박이': 85,

'오징어채볶음': 86,

'우엉조림': 87,

'유부초밥': 88,

'육개장': 89,

'육회': 90,

'잔치국수': 91,

'잡곡밥': 92,

'잡채': 93,

'장어구이': 94,

'장조림': 95,

'전복죽': 96,

'제육볶음': 97,

'조개구이': 98,

'조기구이': 99,

'족발': 100,

'주꾸미볶음': 101,

'짜장면': 102,

'짬뽕': 103,

'쫄면': 104,

'찜닭': 105,

'총각김치': 106,

'추어탕': 107,

'칼국수': 108,

'콩국수': 109,

'콩나물국': 110,

'콩나물무침': 111,

'콩자반': 112,

'파김치': 113,

'파전': 114,

'피자': 115,

'한과': 116,

'해물찜': 117,

'호박전': 118,

'호박죽': 119,

'황태구이': 120,

'후라이드치킨': 121,

'훈제오리': 122}


classes = list(class_indices.keys())

# 📌 3. 이미지 전처리 함수
def preprocess_image(image: Image.Image):
    # ✅ 모델이 기대하는 크기 추출
    expected_shape = input_details[0]['shape']
    if len(expected_shape) == 4:
        if expected_shape[1] == 3:
            input_height = expected_shape[2]
            input_width = expected_shape[3]
        else:
            input_height = expected_shape[1]
            input_width = expected_shape[2]
    else:
        input_height = 224
        input_width = 224



    # 이미지 전처리
    image = image.resize((input_width, input_height))
    image = np.array(image, dtype=np.float32) / 255.0

    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    # 채널 순서 변경 (HWC → CHW)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

 

    return image.astype(np.float32)

# 📌 4. 예측 API
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data))
    predicted_class = classes[predicted_index]
    confidence = float(output_data[0][predicted_index])
    predicted_id = class_indices[predicted_class]

    return {
        "id": predicted_id,
        "class": predicted_class,
        "confidence": confidence
    }