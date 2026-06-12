import moondream as md
from PIL import Image

# Initialize with local inference (NVIDIA GPU or Apple Silicon)
model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI3MGMzNDgxNC03NDY1LTQwOWQtOTRiYS02MTA1N2VhNzQyMDYiLCJvcmdfaWQiOiJ4VXhGSWdJMVpYVjl2RDFBVExGS2U4QXR4VGJaT21IRiIsImlhdCI6MTc4MTAwNjM3NiwidmVyIjoxfQ.DV09lYDkJh4x7GggQpveQFgdLaDvvQ1cGFx3w4jzOhQ", 
              local=True)

# Load an image
image = Image.open("/home/airlab/moondream_test/cat_mm.jpg")

# Generate a caption
caption = model.caption(image)["caption"]
print("Caption:", caption)

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Detect objects
objects = model.detect(image, "person")["objects"]
for obj in objects:
    print(f"Bounds: ({obj['x_min']}, {obj['y_min']}) to ({obj['x_max']}, {obj['y_max']})")

# Locate objects
points = model.point(image, "person")["points"]
for point in points:
    print(f"Center: ({point['x']}, {point['y']})")