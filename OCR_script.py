import pytesseract
from PIL import Image
from pathlib import Path

out = []
for img in sorted(Path(".").glob("mma_ai-*.png")):
    text = pytesseract.image_to_string(Image.open(img))
    out.append(text)

Path("mma_ai.md").write_text("\n\n".join(out))
