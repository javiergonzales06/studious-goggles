import fitz  # PyMuPDF
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "your-translation-model"
SOURCE_PDF = "input.pdf"
OUTPUT_PDF = "translated.pdf"

BATCH_SIZE = 3
MAX_WORKERS = 2


def translate_pages_text(page_texts, page_indices):
    joined_text = "\n\n----- PAGE SPLIT -----\n\n".join(page_texts)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate all English text into Turkish, "
                    "keeping the same structure and order for each page. "
                    "Separate translated pages using the same delimiter: "
                    "----- PAGE SPLIT -----"
                ),
            },
            {"role": "user", "content": joined_text},
        ],
        "temperature": 0.0,
    }

    response = requests.post(VLLM_ENDPOINT, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()
    translated = result["choices"][0]["message"]["content"].strip()
    pages = [p.strip() for p in translated.split("----- PAGE SPLIT -----")]
    return dict(zip(page_indices, pages))


def process_page_translation(out_doc, src_page, translated_text):
    """
    Yeni PDF dokümanına (out_doc) çevrilmiş sayfayı ekler.
    """
    rect = src_page.rect
    new_page = out_doc.new_page(width=rect.width, height=rect.height)

    # Görselleri taşı
    for img_info in src_page.get_images(full=True):
        xref = img_info[0]
        base_image = src_page.parent.extract_image(xref)
        img_bytes = base_image["image"]
        pix = fitz.Pixmap(img_bytes)
        for rect in src_page.get_image_rects(xref):
            new_page.insert_image(rect, pixmap=pix)

    # Metin bloklarını al
    text_blocks = src_page.get_text("blocks")
    block_texts = [b[4] for b in text_blocks if b[4].strip()]
    translated_blocks = translated_text.split("\n\n")
    if len(translated_blocks) != len(block_texts):
        translated_blocks = [translated_text] * len(block_texts)

    for block, translated in zip(text_blocks, translated_blocks):
        x0, y0, x1, y1, *_ = block
        new_page.insert_textbox(
            fitz.Rect(x0, y0, x1, y1),
            translated,
            fontsize=12,
            fontname="DejaVuSans",
            color=(0, 0, 0),
        )


# === Ana Süreç ===
src_doc = fitz.open(SOURCE_PDF)
out_doc = fitz.open()

page_texts = []
page_indices = list(range(len(src_doc)))
for i in page_indices:
    page = src_doc.load_page(i)
    blocks = page.get_text("blocks")
    text = "\n".join(b[4] for b in blocks if b[4].strip())
    page_texts.append(text)

# Sayfaları batch'lere böl
batches = [
    (page_indices[i:i+BATCH_SIZE], page_texts[i:i+BATCH_SIZE])
    for i in range(0, len(page_indices), BATCH_SIZE)
]

translated_pages = {}

# Paralel çeviri
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(translate_pages_text, texts, indices): indices
        for indices, texts in batches
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Çeviri ilerlemesi"):
        try:
            result = future.result()
            translated_pages.update(result)
        except Exception as e:
            print(f"⚠️ Batch çeviri hatası: {e}")

# Çevrilmiş sayfaları yeniden oluştur
for page_idx in tqdm(page_indices, desc="Sayfalar yeniden oluşturuluyor"):
    page = src_doc.load_page(page_idx)
    translated_text = translated_pages.get(page_idx, "")
    process_page_translation(out_doc, page, translated_text)

# PDF kaydet
out_doc.save(OUTPUT_PDF)
out_doc.close()
src_doc.close()

print(f"\n✅ Tüm PDF başarıyla çevrildi: {OUTPUT_PDF}")
