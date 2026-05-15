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
        logger.info("Initializing Pro-Level KYC Engine...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("Engine ready.")

    def get_center_y(self, bbox):
        return (bbox[0][1] + bbox[2][1]) / 2

    def get_center_x(self, bbox):
        return (bbox[0][0] + bbox[2][0]) / 2

    def resize_image(self, img, max_size=1500):
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def extract_raw_blocks(self, file_path):
        all_blocks = []
        y_offset = 0
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = self.resize_image(cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
                page_blocks = self.run_ocr_on_image(img)
                for b in page_blocks:
                    for pt in b['bbox']: pt[1] += y_offset
                all_blocks.extend(page_blocks)
                y_offset += 2500  # Virtual page height for unified space
            doc.close()
        else:
            img = cv2.imread(file_path)
            all_blocks = self.run_ocr_on_image(img)
        return all_blocks

    def run_ocr_on_image(self, img):
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

    def blocks_in_region(self, blocks, y_min, y_max, x_min=0, x_max=1500):
        res = [b for b in blocks if y_min <= self.get_center_y(b['bbox']) <= y_max and x_min <= self.get_center_x(b['bbox']) <= x_max]
        res.sort(key=lambda b: (self.get_center_y(b['bbox']), self.get_center_x(b['bbox'])))
        return res

    def clean_text(self, text, noise=None):
        if not text: return ""
        if noise:
            for n in noise: text = text.replace(n, "")
        return text.strip()

    def process_form(self, blocks):
        logger.info("Starting Intelligent Mapping...")
        
        # --- 1. Locate Pages via Primary Anchors ---
        main_form_anchor = self.find_block_containing(blocks, "Form ISR - 1")
        annexure_anchor = self.find_block_containing(blocks, "Annexure to Form ISR - 1")
        
        # --- 2. Main Form Extraction (ISR-1) ---
        main_data = {}
        if main_form_anchor:
            y_base = self.get_center_y(main_form_anchor['bbox'])
            # Section B: Security Details
            sec_b = self.find_block_containing(blocks, "Security and KYC Details")
            if sec_b:
                y_b = self.get_center_y(sec_b['bbox'])
                region = self.blocks_in_region(blocks, y_b, y_b + 400)
                main_data['issuer'] = self.clean_text(" ".join([b['text'] for b in region if 200 < self.get_center_x(b['bbox']) < 700 and "Company" not in b['text']]))
                main_data['folio'] = self.clean_text(" ".join([b['text'] for b in region if 700 < self.get_center_x(b['bbox']) < 1000 and "Folio" not in b['text']]))
                main_data['qty'] = self.clean_text(" ".join([b['text'] for b in region if 700 < self.get_center_x(b['bbox']) < 1000 and "Securities" not in b['text'] and "Number" not in b['text'] and len(b['text']) < 6]))

            # Section C: Holder Table
            sec_c = self.find_block_containing(blocks, "submitting documents")
            if sec_c:
                y_c = self.get_center_y(sec_c['bbox'])
                rows = self.blocks_in_region(blocks, y_c + 50, y_c + 400)
                holders = []
                # Define X-zones for Holder Name and PAN
                name_zone = (0, 600)
                pan_zone = (600, 1000)
                
                # Group by Y rows (approx 30px height)
                current_y = -999
                row_items = []
                for b in rows:
                    if abs(self.get_center_y(b['bbox']) - current_y) > 20:
                        if row_items:
                            # Process row
                            name = " ".join([i['text'] for i in row_items if name_zone[0] < self.get_center_x(i['bbox']) < name_zone[1] and len(i['text']) > 2 and not i['text'].isdigit()])
                            pan = " ".join([i['text'] for i in row_items if pan_zone[0] < self.get_center_x(i['bbox']) < pan_zone[1] and re.search(r'[A-Z]{5}\d{4}[A-Z]', i['text'])])
                            if name or pan: holders.append({"name": name, "pan": pan})
                        row_items = [b]
                        current_y = self.get_center_y(b['bbox'])
                    else:
                        row_items.append(b)
                main_data['holders'] = holders

            # Bank Details
            bank_anc = self.find_block_containing(blocks, "Bank Account Details")
            if bank_anc:
                y_ba = self.get_center_y(bank_anc['bbox'])
                region = self.blocks_in_region(blocks, y_ba, y_ba + 300)
                main_data['bank'] = {
                    "name": " ".join([b['text'] for b in region if "UNION" in b['text'].upper() or "BANK" in b['text'].upper()]),
                    "ifsc": " ".join([b['text'] for b in region if re.search(r'^[A-Z]{4}0[A-Z0-9]{6}$', b['text'])]),
                    "acc": " ".join([b['text'] for b in region if re.search(r'^\d{12,18}$', b['text'])])
                }

        # --- 3. Annexure Extraction (Multi-Company Table) ---
        annexure_list = []
        if annexure_anchor:
            y_a = self.get_center_y(annexure_anchor['bbox'])
            # Define Table Zones
            col_company = (150, 450)
            col_folio = (450, 580)
            col_qty = (580, 680)
            
            table_region = self.blocks_in_region(blocks, y_a + 200, y_a + 1200)
            
            current_y = -999
            row_blocks = []
            for b in table_region:
                if abs(self.get_center_y(b['bbox']) - current_y) > 25:
                    if row_blocks:
                        comp = " ".join([r['text'] for r in row_blocks if col_company[0] < self.get_center_x(r['bbox']) < col_company[1]])
                        fol = " ".join([r['text'] for r in row_blocks if col_folio[0] < self.get_center_x(r['bbox']) < col_folio[1]])
                        qty = " ".join([r['text'] for r in row_blocks if col_qty[0] < self.get_center_x(r['bbox']) < col_qty[1]])
                        if comp and fol: annexure_list.append({"issuer": comp, "folio": fol, "qty": qty})
                    row_blocks = [b]
                    current_y = self.get_center_y(b['bbox'])
                else:
                    row_blocks.append(b)
        
        # --- 4. Bottom Holder Mapping (Signatures/Addresses) ---
        sig_anc = self.find_block_containing(blocks, "Signature")
        holder_info = []
        if sig_anc:
            y_s = self.get_center_y(sig_anc['bbox'])
            region = self.blocks_in_region(blocks, y_s, y_s + 600)
            # Define 4 Vertical Columns
            columns = [(100, 350), (350, 550), (550, 750), (750, 1000)]
            col_names = ["First Holder", "Joint Holder 1", "Joint Holder 2", "Joint Holder 3"]
            
            for idx, zone in enumerate(columns):
                col_blocks = [b['text'] for b in region if zone[0] < self.get_center_x(b['bbox']) < zone[1]]
                holder_info.append({
                    "column": col_names[idx],
                    "name": col_blocks[0] if col_blocks else "",
                    "address_lines": col_blocks[1:-1] if len(col_blocks) > 2 else [],
                    "pin": col_blocks[-1] if col_blocks and col_blocks[-1].isdigit() else ""
                })

        return {
            "form_isr1": main_data,
            "annexure": annexure_list,
            "holder_details": holder_info,
            "metadata": {"pages_processed": len(blocks)//20, "status": "Success"}
        }
