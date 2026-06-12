from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
)

image = Image.open("/home/airlab/moondream_test/cat_mm.jpg")

# Captioning
print("\nShort caption:")
print(model.caption(image, length="short")["caption"])

print("\nNormal caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    # Streaming generation example, supported for caption() and detect()
    print(t, end="", flush=True)
print(model.caption(image, length="normal"))

# Visual Querying
print("\nVisual query: 'How many animal are in the image?'")
print(model.query(image, "How many animal are in the image?")["answer"])

# Object Detection
print("\nObject detection: 'dog'")
objects = model.detect(image, "dog")["objects"]
print(f"Found {len(objects)} dog(s)")

print("\nObject detection: 'cat'")
objects = model.detect(image, "cat")["objects"]
print(f"Found {len(objects)} cat(s)")

# Pointing
print("\nPointing: 'animal'")
points = model.point(image, "animal")["points"]
print(f"Found {len(points)} animal(s)")
