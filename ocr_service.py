import os
import re
import cv2
import numpy as np
import logging
import fitz
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KYCService:
    def __init__(self):
        logger.info("Initializing KYC Engine...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("Engine ready.")

    # ------------------------------------------------------------------
    # LOW-LEVEL GEOMETRY HELPERS
    # ------------------------------------------------------------------

    def cy(self, b): return (b['bbox'][0][1] + b['bbox'][2][1]) / 2
    def cx(self, b): return (b['bbox'][0][0] + b['bbox'][2][0]) / 2
    def x_left(self, b): return b['bbox'][0][0]
    def x_right(self, b): return b['bbox'][2][0]
    def y_top(self, b): return b['bbox'][0][1]
    def y_bottom(self, b): return b['bbox'][2][1]

    def same_row(self, b1, b2, tol=12):
        return abs(self.cy(b1) - self.cy(b2)) < tol

    def same_col(self, b1, b2, tol=30):
        return abs(self.cx(b1) - self.cx(b2)) < tol

    # ------------------------------------------------------------------
    # BLOCK LOOKUP HELPERS
    # ------------------------------------------------------------------

    def find(self, blocks, pattern, case=False):
        """Find first block containing pattern (substring match)."""
        for b in blocks:
            t = b['text'] if case else b['text'].lower()
            p = pattern if case else pattern.lower()
            if p in t:
                return b
        return None

    def find_all(self, blocks, pattern):
        return [b for b in blocks if pattern.lower() in b['text'].lower()]

    def in_region(self, blocks, y_min, y_max, x_min=0, x_max=9999):
        res = [b for b in blocks
               if y_min <= self.cy(b) <= y_max and x_min <= self.cx(b) <= x_max]
        res.sort(key=lambda b: (self.cy(b), self.cx(b)))
        return res

    def group_rows(self, blocks, tol=12):
        """Cluster blocks into horizontal rows by Y-center proximity."""
        if not blocks: return []
        sorted_b = sorted(blocks, key=lambda b: self.cy(b))
        rows, cur = [], [sorted_b[0]]
        for b in sorted_b[1:]:
            if abs(self.cy(b) - self.cy(cur[-1])) < tol:
                cur.append(b)
            else:
                rows.append(sorted(cur, key=lambda b: self.cx(b)))
                cur = [b]
        rows.append(sorted(cur, key=lambda b: self.cx(b)))
        return rows

    # ------------------------------------------------------------------
    # LABEL-ANCHORED VALUE EXTRACTION
    # ------------------------------------------------------------------

    def val_right_of(self, blocks, label, y_tol=15, skip_words=None):
        """
        Finds label block, then returns the text of the first block
        that is to the RIGHT on the same row.
        skip_words: filter out blocks whose text matches these strings.
        """
        label_block = self.find(blocks, label)
        if not label_block:
            return ""
        label_cy = self.cy(label_block)
        label_xr = self.x_right(label_block)

        candidates = []
        for b in blocks:
            if abs(self.cy(b) - label_cy) <= y_tol and self.x_left(b) > label_xr:
                txt = b['text'].strip()
                if skip_words and any(sw.lower() in txt.lower() for sw in skip_words):
                    continue
                candidates.append(b)

        if not candidates:
            return ""
        # Pick the closest block to the right
        candidates.sort(key=lambda b: self.x_left(b))
        return candidates[0]['text'].strip()

    def val_below(self, blocks, label, x_tol=60, max_dy=80):
        """
        Finds label block, then returns text directly below it.
        """
        label_block = self.find(blocks, label)
        if not label_block:
            return ""
        lcx = self.cx(label_block)
        lyb = self.y_bottom(label_block)
        candidates = [b for b in blocks
                      if abs(self.cx(b) - lcx) < x_tol
                      and 0 < self.y_top(b) - lyb < max_dy]
        if not candidates:
            return ""
        return min(candidates, key=lambda b: self.y_top(b))['text'].strip()

    # ------------------------------------------------------------------
    # IMAGE / PDF LOADING
    # ------------------------------------------------------------------

    def _img_to_blocks(self, img):
        """Run OCR on a BGR numpy image, return list of block dicts."""
        h, w = img.shape[:2]
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        result = self.ocr.predict(img)
        blocks = []
        if result and len(result) > 0:
            item = result[0]
            for text, bbox, score in zip(
                    item.get('rec_texts', []),
                    item.get('dt_polys', []),
                    item.get('rec_scores', [])):
                if text.strip():
                    blocks.append({
                        "text": text.strip(),
                        "bbox": bbox.tolist(),
                        "score": float(score)
                    })
        return blocks

    def extract_raw_blocks(self, file_path):
        """Process every page; stack pages vertically with a 200px gap."""
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            all_blocks, y_offset = [], 0
            for page_num in range(len(doc)):
                logger.info(f"OCR page {page_num + 1}/{len(doc)}")
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                blocks = self._img_to_blocks(img)
                # Shift every block's Y coords by y_offset
                for b in blocks:
                    for pt in b['bbox']:
                        pt[1] += y_offset
                all_blocks.extend(blocks)
                y_offset += 1600  # virtual page height (post-resize ~1500px + 100px gap)
            doc.close()
            return all_blocks
        else:
            img = cv2.imread(file_path)
            return self._img_to_blocks(img)

    # ------------------------------------------------------------------
    # SECTION PARSERS
    # ------------------------------------------------------------------

    def _parse_section_a(self, blocks):
        """Checkbox items that are ticked — identified by presence in OCR text."""
        checkbox_labels = [
            ("PAN", "PAN"),
            ("Signature", "Signature"),
            ("Mobile Number", "Mobile Number"),
            ("Bank details", "Bank details"),
            ("Registered Address", "Registered Address"),
            ("E-mail address", "E-mail address"),
        ]
        ticked = []
        for key, label in checkbox_labels:
            if self.find(blocks, label):
                ticked.append(key)
        return ticked

    def _parse_section_b(self, blocks, sec_anchor_y, next_anchor_y):
        """Section B: Security and KYC Details."""
        region = self.in_region(blocks, sec_anchor_y - 10, next_anchor_y)
        return {
            "issuer_company": self.val_right_of(region, "Name of the Issuer Company",
                                                 skip_words=["Name", "Issuer"]),
            "folio_no": self.val_right_of(region, "Folio No"),
            "face_value_per_security": self.val_right_of(region, "Face value of Securities",
                                                          skip_words=["Face", "value"]),
            "number_of_securities": self.val_right_of(region, "Number of Securities",
                                                       skip_words=["Number", "Securities"]),
            "distinctive_numbers": {
                "from": self.val_right_of(region, "From", y_tol=20),
                "to": self.val_right_of(region, "To", y_tol=20,
                                        skip_words=["Distinctive", "Optional"]),
            },
            "email": self.val_right_of(region, "E-mail Address"),
            "mobile": self.val_right_of(region, "Mobile Number"),
        }

    def _parse_section_c(self, blocks, sec_anchor_y, next_anchor_y):
        """
        Section C: Holders table.
        Strategy:
          1. Find "PAN" header block to get the X boundary between Name and PAN columns.
          2. Find "Aadhaar" header to get Aadhaar column X.
          3. Group rows; for each data row extract name / pan / aadhaar.
        """
        region = self.in_region(blocks, sec_anchor_y, next_anchor_y)

        # Dynamically find column X boundaries from header row
        pan_header = self.find(region, "PAN")
        aadhaar_header = self.find(region, "Aadhaar")

        pan_col_x = self.cx(pan_header) if pan_header else 700
        aadhaar_col_x = self.cx(aadhaar_header) if aadhaar_header else 900

        # Name zone: left of PAN column
        # PAN zone: around pan_col_x ±150
        # Aadhaar zone: around aadhaar_col_x ±100

        pan_regex = re.compile(r'[A-Z]{5}\d{4}[A-Z]')
        noise_words = {'pan', 'aadhaar', 'yes', 'no', 'tick', 'linked', 'name',
                       'security', 'holder', 'copies', 'note', 'mandatory', 'capital'}

        holders = []
        rows = self.group_rows(region, tol=14)

        for row in rows:
            # Skip header rows
            row_text = " ".join(b['text'] for b in row).lower()
            if any(w in row_text for w in ['name(s)', 'holder(s)', 'copies of', 'note:', 'july']):
                continue

            # Extract PAN from PAN column zone
            pan_blocks = [b for b in row
                          if abs(self.cx(b) - pan_col_x) < 200
                          and pan_regex.search(b['text'])]
            if not pan_blocks:
                continue  # Only process rows that have a PAN
            pan_val = pan_regex.search(pan_blocks[0]['text']).group()

            # Extract name: text to the LEFT of the PAN column, non-noise
            name_blocks = [b for b in row
                           if self.cx(b) < pan_col_x - 50
                           and b['text'].lower() not in noise_words
                           and not b['text'].strip().isdigit()
                           and len(b['text']) > 1]
            name_val = " ".join(b['text'] for b in sorted(name_blocks, key=lambda b: self.cx(b)))
            # Strip leading row numbers like "1.", "2."
            name_val = re.sub(r'^\d[\.\s]+', '', name_val).strip()

            # Extract Aadhaar Y/N from Aadhaar column zone
            aadhaar_blocks = [b for b in row
                              if abs(self.cx(b) - aadhaar_col_x) < 150
                              and b['text'].lower() in ('yes', 'no', 'y', 'n')]
            aadhaar_val = aadhaar_blocks[0]['text'].capitalize() if aadhaar_blocks else "No"

            holders.append({
                "name": name_val,
                "pan": pan_val,
                "aadhaar_linked": aadhaar_val
            })

        return holders

    def _parse_bank(self, blocks, bank_anchor_y, next_anchor_y):
        """Bank Account Details of First Holder."""
        region = self.in_region(blocks, bank_anchor_y, next_anchor_y)

        # IFSC via regex
        ifsc = ""
        for b in region:
            m = re.search(r'[A-Z]{4}0[A-Z0-9]{6}', b['text'])
            if m:
                ifsc = m.group()
                break

        # Account number via regex
        acc = ""
        for b in region:
            m = re.search(r'\d{12,18}', b['text'].replace(" ", ""))
            if m:
                acc = m.group()
                break

        # Bank name: row containing BANK
        bank_name = ""
        for b in region:
            if "bank" in b['text'].lower() and len(b['text']) > 6:
                bank_name = b['text']
                break

        # Account type (Savings / Current)
        acc_type = "Savings"
        for b in region:
            if "current" in b['text'].lower():
                acc_type = "Current"
                break

        # Bank category (NRO / NRE)
        category = ""
        for b in region:
            if "nro" in b['text'].lower():
                category = "NRO"
                break
            if "nre" in b['text'].lower():
                category = "NRE"
                break

        return {
            "bank_name_branch": bank_name,
            "ifsc": ifsc,
            "account_no": acc,
            "account_type": acc_type,
            "bank_category": category,
        }

    def _parse_demat(self, blocks):
        """16-digit Demat DPID/Client ID anywhere in document."""
        all_text = " ".join(b['text'] for b in blocks)
        m = re.search(r'IN\d{6}\s*[-–]\s*\d{8}', all_text)
        return m.group().strip() if m else ""

    def _parse_declaration_table(self, blocks, decl_anchor_y, end_y):
        """
        Bottom table: First Named Holder + Joint Holder 1/2/3.
        Strategy: use X-center of each column header to define columns dynamically.
        """
        region = self.in_region(blocks, decl_anchor_y - 20, end_y)

        # Find column header blocks
        first_hdr = self.find(region, "First Named Holder")
        jh1_hdr = self.find(region, "Joint Holder - 1")
        jh2_hdr = self.find(region, "Joint Holder - 2")
        jh3_hdr = self.find(region, "Joint Holder - 3")

        if not first_hdr:
            return []

        # Column centres and boundaries
        col_centers = [
            self.cx(first_hdr),
            self.cx(jh1_hdr) if jh1_hdr else self.cx(first_hdr) + 200,
            self.cx(jh2_hdr) if jh2_hdr else self.cx(first_hdr) + 400,
            self.cx(jh3_hdr) if jh3_hdr else self.cx(first_hdr) + 600,
        ]
        col_labels = ["First Holder", "Joint Holder 1", "Joint Holder 2", "Joint Holder 3"]

        def in_col(b, col_idx, half_width=90):
            return abs(self.cx(b) - col_centers[col_idx]) < half_width

        # Find row anchors for Name, Address, PIN
        name_row_anchor = self.find(region, "Name")
        addr_row_anchor = self.find(region, "Address")
        pin_row_anchor = self.find(region, "PIN")

        name_y = self.cy(name_row_anchor) if name_row_anchor else decl_anchor_y + 150
        addr_y = self.cy(addr_row_anchor) if addr_row_anchor else decl_anchor_y + 200
        pin_y = self.cy(pin_row_anchor) if pin_row_anchor else end_y - 50

        holders = []
        for i in range(4):
            # NAME: blocks on the name row in this column
            name_blocks = [b for b in region
                           if in_col(b, i) and abs(self.cy(b) - name_y) < 25
                           and b['text'].lower() not in {'name', 'address', 'pin', 'signature'}]
            name_val = " ".join(b['text'] for b in sorted(name_blocks, key=lambda b: self.cx(b)))

            # ADDRESS: blocks between name_y and pin_y in this column
            addr_blocks = [b for b in region
                           if in_col(b, i)
                           and name_y + 10 < self.cy(b) < pin_y - 10
                           and b['text'].lower() not in {'name', 'address', 'pin', 'signature',
                                                         'first named holder', 'joint holder - 1',
                                                         'joint holder - 2', 'joint holder - 3'}
                           and not b['text'].strip().isdigit()]
            addr_val = " ".join(b['text'] for b in sorted(addr_blocks, key=lambda b: self.cy(b)))

            # PIN: 6-digit number near pin_y in this column
            pin_val = ""
            for b in region:
                if in_col(b, i) and abs(self.cy(b) - pin_y) < 30 and re.match(r'^\d{6}$', b['text']):
                    pin_val = b['text']
                    break

            holders.append({
                "holder_type": col_labels[i],
                "name": name_val,
                "address": addr_val,
                "pin_code": pin_val,
                "signature_present": True  # presence inferred from the form being submitted
            })

        return holders

    def _parse_annexure(self, blocks, annex_anchor_y, end_y):
        """
        Annexure table: dynamically find column X-centers from the header row,
        then assign each data block to the nearest column.
        """
        region = self.in_region(blocks, annex_anchor_y, end_y)

        # Find Annexure table header blocks
        sr_hdr = self.find(region, "Sr.No") or self.find(region, "Sr No")
        comp_hdr = self.find(region, "Name of the Issuer Company") or self.find(region, "Issuer Company")
        folio_hdr = self.find(region, "Folio No")
        qty_hdr = self.find(region, "Quantity")
        fv_hdr = self.find(region, "Face Value")

        if not comp_hdr:
            return []

        # Header Y — everything BELOW this is data
        header_y = max(self.cy(h) for h in [sr_hdr, comp_hdr, folio_hdr, qty_hdr, fv_hdr] if h)

        # Column X centres
        cols = {
            "sr_no": self.cx(sr_hdr) if sr_hdr else 60,
            "company": self.cx(comp_hdr),
            "folio": self.cx(folio_hdr) if folio_hdr else self.cx(comp_hdr) + 150,
            "qty": self.cx(qty_hdr) if qty_hdr else self.cx(comp_hdr) + 250,
            "fv": self.cx(fv_hdr) if fv_hdr else self.cx(comp_hdr) + 350,
        }

        def nearest_col(b):
            dists = {k: abs(self.cx(b) - v) for k, v in cols.items()}
            return min(dists, key=dists.get)

        # Group data rows
        data_region = self.in_region(region, header_y + 10, end_y)
        rows = self.group_rows(data_region, tol=20)

        annexure = []
        for row in rows:
            row_map = {}
            for b in row:
                col_key = nearest_col(b)
                row_map.setdefault(col_key, []).append(b['text'])

            company = " ".join(row_map.get("company", []))
            folio = " ".join(row_map.get("folio", []))
            if not company or not folio:
                continue
            # Filter noise rows (headers repeated, notes, etc.)
            if any(w in company.lower() for w in ['issuer', 'company', 'note', 'authorization']):
                continue

            sr_text = " ".join(row_map.get("sr_no", []))
            try:
                sr = int(re.search(r'\d+', sr_text).group()) if sr_text else len(annexure) + 1
            except Exception:
                sr = len(annexure) + 1

            qty_text = " ".join(row_map.get("qty", []))
            try:
                qty = int(re.search(r'\d+', qty_text).group()) if qty_text else None
            except Exception:
                qty = None

            annexure.append({
                "sr_no": sr,
                "issuer_company": company,
                "folio_no": folio,
                "quantity": qty,
                "face_value": " ".join(row_map.get("fv", [])),
            })

        return annexure

    # ------------------------------------------------------------------
    # MASTER PROCESS FUNCTION
    # ------------------------------------------------------------------

    def process_form(self, blocks):
        logger.info("Starting label-anchored extraction...")

        # ── Identify page section anchors ──────────────────────────────
        isr1_anchor   = self.find(blocks, "Form ISR - 1")
        annex_anchor  = self.find(blocks, "Annexure to Form ISR - 1")

        sec_b_anchor  = self.find(blocks, "Security and KYC Details")
        sec_c_anchor  = self.find(blocks, "submitting documents")
        bank_anchor   = self.find(blocks, "Bank Account Details")
        demat_anchor  = self.find(blocks, "Demat Account Number")
        decl_anchor   = self.find(blocks, "Declaration")

        def Y(anchor): return self.cy(anchor['bbox']) if anchor else 0

        isr1_y   = Y(isr1_anchor)
        sec_b_y  = Y(sec_b_anchor)
        sec_c_y  = Y(sec_c_anchor)
        bank_y   = Y(bank_anchor)
        demat_y  = Y(demat_anchor)
        decl_y   = Y(decl_anchor)
        annex_y  = Y(annex_anchor)
        end_y    = decl_y + 1200  # generous end boundary

        logger.info(f"Anchors — ISR1:{isr1_y:.0f} SecB:{sec_b_y:.0f} SecC:{sec_c_y:.0f} "
                    f"Bank:{bank_y:.0f} Decl:{decl_y:.0f} Annex:{annex_y:.0f}")

        # ── Section A ──────────────────────────────────────────────────
        requested_updates = self._parse_section_a(
            self.in_region(blocks, isr1_y, sec_b_y)
        )

        # ── Section B ──────────────────────────────────────────────────
        security_details = self._parse_section_b(blocks, sec_b_y, sec_c_y)

        # ── Section C ──────────────────────────────────────────────────
        holders_raw = self._parse_section_c(blocks, sec_c_y, bank_y)

        # ── Bank ───────────────────────────────────────────────────────
        bank_details = self._parse_bank(blocks, bank_y, demat_y + 200)

        # ── Demat ──────────────────────────────────────────────────────
        demat_id = self._parse_demat(blocks)

        # ── Declaration / Bottom Table ─────────────────────────────────
        holder_addresses = self._parse_declaration_table(blocks, decl_y, end_y)

        # Merge PAN/Aadhaar info from Section C into holder_addresses
        for i, h in enumerate(holder_addresses):
            if i < len(holders_raw):
                h['pan'] = holders_raw[i]['pan']
                h['aadhaar_linked'] = holders_raw[i]['aadhaar_linked']
                # Prefer name from Section C (printed/cleaner) if better
                if not h['name'] and holders_raw[i]['name']:
                    h['name'] = holders_raw[i]['name']
            else:
                h['pan'] = ""
                h['aadhaar_linked'] = "No"

        # ── Annexure ───────────────────────────────────────────────────
        annexure_details = []
        if annex_anchor:
            annexure_details = self._parse_annexure(blocks, annex_y, annex_y + 2000)

        # ── Final JSON ─────────────────────────────────────────────────
        return {
            "form_metadata": {
                "form_type": "Form ISR-1",
                "date": re.search(r'\d{2}/\d{2}/\d{4}',
                                  " ".join(b['text'] for b in
                                           self.in_region(blocks, isr1_y, sec_b_y))).group()
                       if re.search(r'\d{2}/\d{2}/\d{4}',
                                    " ".join(b['text'] for b in
                                             self.in_region(blocks, isr1_y, sec_b_y))) else "",
                "requested_updates": requested_updates,
            },
            "security_details": security_details,
            "contact_details": {
                "email": security_details.pop("email", ""),
                "mobile": security_details.pop("mobile", ""),
            },
            "bank_account_details": bank_details,
            "demat_details": {"dpid_client_id": demat_id},
            "holders_information": holder_addresses,
            "annexure_details": annexure_details,
        }
