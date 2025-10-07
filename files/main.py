import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io, os
import pytesseract

def extract_pdf(pdf_path, image_dir="images", ocr=True, image_quality=70):
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    full_flow = []

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        flow = []
        blocks = page.get_text("dict")["blocks"]

        # 1️⃣ Metin ve Görselleri çıkar
        for block in blocks:
            if "lines" in block:
                # Metin bloğu
                text = " ".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
                if text:
                    flow.append({
                        "type": "text",
                        "page": page_number,
                        "bbox": block["bbox"],
                        "content": text,
                        "source": "pdf"
                    })
            elif "image" in block:
                # Görsel bloğu
                xref = block["image"]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                img = Image.open(io.BytesIO(image_data))

                # Görsel sıkıştır
                img_path = f"{image_dir}/page{page_number}_img{len(flow)}.jpg"
                img.save(img_path, "JPEG", quality=image_quality)

                flow.append({
                    "type": "image",
                    "page": page_number,
                    "bbox": block["bbox"],
                    "path": img_path
                })

        # 2️⃣ Tablo çıkarımı (pdfplumber)
        with pdfplumber.open(pdf_path) as plumber_doc:
            page_tables = plumber_doc.pages[page_index].extract_tables()
            for table in page_tables:
                flow.append({
                    "type": "table",
                    "page": page_number,
                    "content": table
                })

        # 3️⃣ OCR fallback (eğer hiç metin bulunamadıysa)
        if ocr and not any(b["type"] == "text" for b in flow):
            pix = page.get_pixmap()
            img_path = f"{image_dir}/page{page_number}_ocr.png"
            pix.save(img_path)
            ocr_text = pytesseract.image_to_string(Image.open(img_path))
            if ocr_text.strip():
                flow.append({
                    "type": "text",
                    "page": page_number,
                    "content": ocr_text.strip(),
                    "source": "ocr"
                })

        # Sayfa bazlı sırayı pozisyona göre sırala (y0'a göre)
        flow.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
        full_flow.extend(flow)

    return full_flow
