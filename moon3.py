import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"},
    torch_dtype=torch.float16  # VRAM 최적화 (OOM 방지)
)

image = Image.open("/home/airlab/moondream_test/cat_mm.jpg")

print("\n================== 추론 시간 측정 시작 ==================")

# 1. Short Caption 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nShort caption:")
print(model.caption(image, length="short")["caption"])

torch.cuda.synchronize()
end = time.perf_counter()
print(f"⏱️ [Short caption] 소요 시간: {end - start:.4f} 초")
print("-" * 50)


# 2. Normal Caption (스트리밍 방식) 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nNormal caption (스트리밍):")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    # Streaming generation example, supported for caption() and detect()
    print(t, end="", flush=True)

torch.cuda.synchronize()
end = time.perf_counter()
print(f"\n⏱️ [Normal caption - 스트리밍] 소요 시간: {end - start:.4f} 초")
print("-" * 50)


# 3. Normal Caption (일반 출력 방식) 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nNormal caption (전체 출력):")
print(model.caption(image, length="normal")["caption"])

torch.cuda.synchronize()
end = time.perf_counter()
print(f"⏱️ [Normal caption - 전체 출력] 소요 시간: {end - start:.4f} 초")
print("-" * 50)


# 4. Visual Querying 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nVisual query: 'How many animal are in the image?'")
print(model.query(image, "How many animal are in the image?")["answer"])

torch.cuda.synchronize()
end = time.perf_counter()
print(f"⏱️ [Visual query] 소요 시간: {end - start:.4f} 초")
print("-" * 50)


# 5. Object Detection ('dog') 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nObject detection: 'dog'")
objects = model.detect(image, "dog")["objects"]
print(f"Found {len(objects)} dog(s)")

torch.cuda.synchronize()
end = time.perf_counter()
print(f"⏱️ [Detection - 'dog'] 소요 시간: {end - start:.4f} 초")
print("-" * 50)


# 6. Object Detection ('cat') 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nObject detection: 'cat'")
objects = model.detect(image, "cat")["objects"]
print(f"Found {len(objects)} cat(s)")

torch.cuda.synchronize()
end = time.perf_counter()
print(f"⏱️ [Detection - 'cat'] 소요 시간: {end - start:.4f} 초")
print("-" * 50)


# 7. Pointing 측정
torch.cuda.synchronize()
start = time.perf_counter()

print("\nPointing: 'animal'")
points = model.point(image, "animal")["points"]
print(f"Found {len(points)} animal(s)")

torch.cuda.synchronize()
end = time.perf_counter()
print(f"⏱️ [Pointing] 소요 시간: {end - start:.4f} 초")
print("=========================================================")