import cv2
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

print("모델 다운로드 및 로딩 중... (약 10초 소요)")

# 1. 모델 로드 (이전과 동일, VRAM 최적화)
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"},
    torch_dtype=torch.float16
)

# 2. 웹캠(카메라) 연결
# 0은 기본 카메라를 의미합니다. 노트북 웹캠이나 USB 카메라가 연결되어 있어야 합니다.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 찾을 수 없습니다. 카메라 연결 상태를 확인해 주세요.")
    exit()

print("\n카메라 실행 완료!")
print("=========================================")
print(" [Spacebar] : 현재 화면 분석하기")
print(" [q] 키     : 프로그램 종료")
print("=========================================")

while True:
    # 카메라에서 프레임(이미지) 읽어오기
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 불러올 수 없습니다.")
        break

    # 화면에 현재 카메라 모습 띄우기
    cv2.imshow('Moondream2 Camera Test', frame)

    # 키보드 입력 대기 (1ms)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # 'q'를 누르면 종료
        break
    
    elif key == ord(' '):  # '스페이스바'를 누르면 분석 시작
        print("\n[AI 분석")
        
        
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_frame)
        
        # 캡션 생성 (상황 설명)
        caption = model.caption(pil_image, length="normal")["caption"]
        print(f"상황 설명: {caption}")
        
        print("\nVisual query: 'what is ?'")
        print(model.query(pil_image, "what is ?")["answer"])

        # 객체 탐지 예시 
        objects = model.detect(pil_image, "person")["objects"]
        print(f"탐지된 사람 수: {len(objects)}개")
        print("-----------------------------------------")

# 종료
cap.release()
cv2.destroyAllWindows()
