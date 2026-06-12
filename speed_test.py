import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

print("모델 및 토크나이저 로딩 중...")
model_id = "vikhyatk/moondream2"
revision = "2025-06-21"

# 1. 모델과 토크나이저 불러오기
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=True,
    device_map={"": "cuda"},
    torch_dtype=torch.float16
)
# 글자를 토큰 개수로 세기 위해 토크나이저가 필요합니다!
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open("/home/airlab/moondream_test/cat_mm.jpg")

print("\n================== 토큰 생성 속도 측정 ==================")

# 타이머 시작 전 GPU 동기화
torch.cuda.synchronize()
start_time = time.perf_counter()

first_token_time = None
generated_text = ""

print("AI 캡션 스트리밍: ", end="", flush=True)

# stream=True로 텍스트를 한 조각씩 받아옵니다.
for chunk in model.caption(image, length="normal", stream=True)["caption"]:
    # 첫 글자(토큰)가 튀어나온 순간의 시간을 기록합니다.
    if first_token_time is None:
        torch.cuda.synchronize()
        first_token_time = time.perf_counter()

    print(chunk, end="", flush=True)
    generated_text += chunk

# 모든 문장이 완성된 시점 기록
torch.cuda.synchronize()
end_time = time.perf_counter()

# ================= 결과 분석 =================

# 1. 생성된 문장을 다시 토큰으로 변환하여 정확한 개수를 셉니다.
token_count = len(tokenizer.encode(generated_text))

# 2. 첫 토큰까지 걸린 시간 (이미지 분석 시간 포함)
ttft = first_token_time - start_time

# 3. 첫 토큰 이후 나머지 문장을 완성하는 데 걸린 순수 디코딩 시간
decoding_time = end_time - first_token_time

# 4. 초당 토큰 생성 속도 (Tokens per Second)
if decoding_time > 0 and token_count > 1:
    tps = (token_count - 1) / decoding_time
else:
    tps = 0

print("\n\n================== 측정 결과 리포트 ==================")
print(f"총 생성된 토큰 수: {token_count} 개")
print(f"전체 소요 시간: {end_time - start_time:.4f} 초")
print("-" * 50)
print(f"⏱️ 첫 토큰 생성 시간 (TTFT): {ttft:.4f} 초  <-- 이미지를 이해하는 데 걸린 병목 구간")
print(f"⏱️ 순수 문장 생성 시간: {decoding_time:.4f} 초")
print(f"🚀 초당 토큰 생성 속도: {tps:.2f} tokens/sec")
print("======================================================")