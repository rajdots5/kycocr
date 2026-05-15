import os
import json
import re
import cv2
import numpy as np
import logging
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KYCService:
    def __init__(self):
        logger.info("Initializing Multi-Page OCR engine...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("Engine ready.")

    def get_center_y(self, bbox):
        ys = [pt[1] for pt in bbox]
        return (min(ys) + max(ys)) / 2

    def get_center_x(self, bbox):
        xs = [pt[0] for pt in bbox]
        return (min(xs) + max(xs)) / 2

    def resize_image(self, img, max_size=1500):
        h, w = img.shape[:2]
        if max(h, w) <= max_size: return img
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def extract_raw_blocks(self, file_path):
        """Processes ALL pages of a PDF/Image and returns a unified list of blocks."""
        all_blocks = []
        y_offset = 0

        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                logger.info(f"Processing Page {page_num + 1}...")
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                
                # Run OCR on this page
                page_blocks = self.run_ocr_on_image(img)
                
                # Shift Y coordinates by y_offset to keep pages separate in space
                for b in page_blocks:
                    for pt in b['bbox']: pt[1] += y_offset
                
                all_blocks.extend(page_blocks)
                y_offset += pix.h + 500  # Large gap between pages to avoid overlap
            doc.close()
        else:
            img = cv2.imread(file_path)
            all_blocks = self.run_ocr_on_image(img)

        return all_blocks

    def run_ocr_on_image(self, img):
        """Helper to run OCR on a single image array."""
        img = self.resize_image(img)
        result = self.ocr.predict(img)
        blocks = []
        if result and len(result) > 0:
            item = result[0]
            for text, bbox, score in zip(item.get('rec_texts', []), item.get('dt_polys', []), item.get('rec_scores', [])):
                if text.strip():
                    blocks.append({"text": text.strip(), "bbox": bbox.tolist(), "score": float(score)})
        return blocks

    def find_block_containing(self, blocks, pattern):
        for b in blocks:
            if pattern.lower() in b['text'].lower(): return b
        return None

    def blocks_in_region(self, blocks, y_min, y_max, x_min=0, x_max=9999):
        res = [b for b in blocks if y_min <= self.get_center_y(b['bbox']) <= y_max and x_min <= self.get_center_x(b['bbox']) <= x_max]
        res.sort(key=lambda b: (self.get_center_y(b['bbox']), self.get_center_x(b['bbox'])))
        return res

    def process_form(self, blocks):
        logger.info("Mapping Multi-Page document...")
        
        # --- 1. Identify Page Sections ---
        isr1_anchor = self.find_block_containing(blocks, "Form ISR - 1")
        annexure_anchor = self.find_block_containing(blocks, "Annexure to Form ISR - 1")
        
        # --- 2. Extract Data from Form ISR-1 Page ---
        isr1_y = self.get_center_y(isr1_anchor['bbox']) if isr1_anchor else 0
        isr1_blocks = self.blocks_in_region(blocks, isr1_y, isr1_y + 3000) # Look within the next 3000px

        # Basic Details
        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', " ".join([b['text'] for b in isr1_blocks]))
        date = date_match.group(1) if date_match else ""
        
        issuer = ""
        b = self.find_block_containing(isr1_blocks, "COMPANY LIMITED")
        if b: issuer = b['text']

        folio = ""
        b = self.find_block_containing(isr1_blocks, "Folio No")
        if b:
            ly = self.get_center_y(b['bbox'])
            for val in isr1_blocks:
                if abs(self.get_center_y(val['bbox']) - ly) < 15 and self.get_center_x(val['bbox']) > self.get_center_x(b['bbox']):
                    folio = val['text']
                    break

        # Holder & PAN Table (PAN-Anchor Strategy)
        holders = []
        pan_regex = r'[A-Z]{5}\d{4}[A-Z]'
        pans = [b for b in isr1_blocks if re.search(pan_regex, b['text'])]
        for pb in pans:
            pan_val = re.search(pan_regex, pb['text']).group()
            py = self.get_center_y(pb['bbox'])
            name_parts = [b['text'] for b in isr1_blocks if abs(self.get_center_y(b['bbox']) - py) < 15 and self.get_center_x(b['bbox']) < self.get_center_x(pb['bbox'])]
            clean_name = " ".join([n for n in name_parts if not re.match(r'^[1-4]$', n) and len(n) > 2])
            holders.append({"name": clean_name, "pan": pan_val})

        # Bank Details
        bank_name, ifsc, acc_no = "", "", ""
        for b in isr1_blocks:
            t = b['text']
            if 'UBIN' in t: ifsc = t
            elif re.match(r'^\d{12,18}$', t): acc_no = t
            elif 'BANK' in t and len(t) > 5: bank_name = t

        # --- 3. Extract Data from Annexure Page ---
        additional_companies = []
        if annexure_anchor:
            ay = self.get_center_y(annexure_anchor['bbox'])
            annexure_blocks = self.blocks_in_region(blocks, ay, ay + 2000)
            
            # Find Folios in Annexure
            for b in annexure_blocks:
                # Look for company names that are near folio numbers in the Annexure table
                if re.match(r'^[A-Z0-9]{6,10}$', b['text']) and b['text'] != folio:
                    # Found an additional folio, look left for company name
                    cy = self.get_center_y(b['bbox'])
                    company = " ".join([v['text'] for v in annexure_blocks if abs(self.get_center_y(v['bbox']) - cy) < 20 and self.get_center_x(v['bbox']) < self.get_center_x(b['bbox'])])
                    if company: additional_companies.append({"company": company, "folio": b['text']})

        return {
            "main_form": {
                "date": date,
                "issuer": issuer,
                "folio": folio,
                "holders": holders,
                "bank": {"name": bank_name, "ifsc": ifsc, "account": acc_no}
            },
            "annexure_details": additional_companies,
            "status": "Multi-page extraction complete"
        }
