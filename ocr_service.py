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

    def cy(self, b): return (b['bbox'][0][1] + b['bbox'][2][1]) / 2
    def cx(self, b): return (b['bbox'][0][0] + b['bbox'][2][0]) / 2
    def x_left(self, b):  return b['bbox'][0][0]
    def x_right(self, b): return b['bbox'][2][0]

    def find(self, blocks, pattern):
        p = pattern.lower()
        for b in blocks:
            if p in b['text'].lower(): return b
        return None

    def in_band(self, blocks, y_min, y_max, x_min=0, x_max=9999):
        r = [b for b in blocks if y_min <= self.cy(b) <= y_max and x_min <= self.cx(b) <= x_max]
        r.sort(key=lambda b: (self.cy(b), self.cx(b)))
        return r

    def group_rows(self, blocks, tol=12):
        if not blocks: return []
        sb = sorted(blocks, key=lambda b: self.cy(b))
        rows, cur = [], [sb[0]]
        for b in sb[1:]:
            if abs(self.cy(b) - self.cy(cur[-1])) <= tol: cur.append(b)
            else:
                rows.append(sorted(cur, key=lambda b: self.cx(b)))
                cur = [b]
        rows.append(sorted(cur, key=lambda b: self.cx(b)))
        return rows

    def _ocr_img(self, img):
        h, w = img.shape[:2]
        if max(h, w) > 1500:
            s = 1500 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        result = self.ocr.predict(img)
        blocks = []
        if result:
            item = result[0]
            for text, bbox, score in zip(item.get('rec_texts', []), item.get('dt_polys', []), item.get('rec_scores', [])):
                if text.strip():
                    blocks.append({"text": text.strip(), "bbox": bbox.tolist(), "score": float(score)})
        return blocks

    def extract_raw_blocks(self, file_path):
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            all_blocks, y_off = [], 0
            for pn in range(len(doc)):
                page = doc.load_page(pn)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                blks = self._ocr_img(img)
                for b in blks:
                    for pt in b['bbox']: pt[1] += y_off
                all_blocks.extend(blks)
                y_off += 1600
            doc.close()
            return all_blocks
        return self._ocr_img(cv2.imread(file_path))

    def process_form(self, blocks):
        logger.info("Starting extraction...")

        # ── ANCHOR Y POSITIONS ──────────────────────────────────────────
        # Find each section anchor
        isr1    = self.find(blocks, "Form ISR - 1")
        sec_b   = self.find(blocks, "Folio No")          # Section B anchor
        sec_c   = self.find(blocks, "Na  Sy") or self.find(blocks, "Scyo") or self.find(blocks, "submitting")
        bank_a  = self.find(blocks, "Bank Account Details")
        demat_a = self.find(blocks, "Demat Account")
        decl_a  = self.find(blocks, "Declaration") or self.find(blocks, "First Named Holder")
        annex_a = self.find(blocks, "Annexure to Form ISR")

        def Y(a): return self.cy(a) if a else 0

        isr1_y  = Y(isr1)
        folioN_y = Y(sec_b)                # Y of "Folio No." label
        sec_c_y = Y(sec_c)
        bank_y  = Y(bank_a)
        demat_y = Y(demat_a)
        decl_y  = Y(decl_a)
        annex_y = Y(annex_a)

        logger.info(f"Anchors: isr1={isr1_y:.0f} secB_folio={folioN_y:.0f} "
                    f"secC={sec_c_y:.0f} bank={bank_y:.0f} decl={decl_y:.0f}")

        # ── 1. METADATA ─────────────────────────────────────────────────
        date_txt = " ".join(b['text'] for b in self.in_band(blocks, isr1_y, isr1_y + 200))
        date_m = re.search(r'\d{2}/\d{2}/\d{4}', date_txt)
        # Fallback: partial date like "21/07/202" → try broader
        if not date_m:
            date_m = re.search(r'\d{2}/\d{2}/\d{2,4}', date_txt)
        date = date_m.group() if date_m else ""

        # Section A checkboxes:
        # The form prints ALL label text regardless of tick state.
        # We detect ticked items by looking for a tick-like character (v, V, ✓, [v], ■)
        # appearing NEAR each label in the same row.
        sec_a_region = self.in_band(blocks, isr1_y, folioN_y if folioN_y else isr1_y + 300)
        chk_labels = ["PAN", "Signature", "Mobile Number", "Bank details",
                      "Registered Address", "E-mail address"]
        # Collect all text in section A to scan for tick indicators
        sec_a_text_full = " ".join(b['text'] for b in sec_a_region).lower()
        tick_chars = re.compile(r'[✓✔☑■▪]|\[v\]|\[x\]')
        # Heuristic: if any tick char found, only return labels that have a tick nearby.
        # If no tick chars detected (OCR missed them), fall back to all labels.
        has_ticks = bool(tick_chars.search(sec_a_text_full))
        if has_ticks:
            req_updates = []
            for lbl in chk_labels:
                lbl_blk = self.find(sec_a_region, lbl)
                if not lbl_blk: continue
                ly = self.cy(lbl_blk)
                lx = self.cx(lbl_blk)
                # Check for tick char in same row, near label
                nearby = [b for b in sec_a_region
                          if abs(self.cy(b) - ly) < 15 and abs(self.cx(b) - lx) < 80]
                if any(tick_chars.search(b['text']) for b in nearby):
                    req_updates.append(lbl)
        else:
            # OCR missed ticks — return only the standard always-present ones
            always_ticked = {"PAN", "Signature", "Mobile Number", "Bank details"}
            req_updates = [l for l in chk_labels
                           if self.find(sec_a_region, l) and l in always_ticked]

        # ── 2. SECTION B — Security Details ─────────────────────────────
        # Key coordinates (from actual OCR dump):
        # Folio No. label — whatever Y it is, the row looks like:
        #   [issuer company (left-center)]  [Folio No. label]  [folio value (right)]
        sec_b_end = sec_c_y if sec_c_y else bank_y
        sec_b_blks = self.in_band(blocks, folioN_y - 30, sec_b_end)

        # Issuer Company: same row as Folio No., X strictly between 150 and folio_lbl_left
        # This skips far-left label zone (X<130) and the folio label itself
        folio_lbl = sec_b  # already found above
        issuer = ""
        folio  = ""
        if folio_lbl:
            fy       = self.cy(folio_lbl)
            folio_lx = self.x_left(folio_lbl)
            # Issuer: blocks left of folio label, same row
            # Exclude far-left labels (X<130) AND known label phrases
            _lbl_phrases = ['name of the issuer', 'name of the', 'issuer company',
                            'folio no', 'number of', 'face value']
            i_blks = [b for b in sec_b_blks
                      if abs(self.cy(b) - fy) < 15
                      and self.x_right(b) < folio_lx
                      and self.cx(b) > 100
                      and not any(p in b['text'].lower() for p in _lbl_phrases)]
            issuer = " ".join(b['text'] for b in sorted(i_blks, key=lambda b: self.cx(b)))

            # Folio value: first block to right of label
            f_blks = [b for b in sec_b_blks
                      if abs(self.cy(b) - fy) < 15
                      and self.x_left(b) > self.x_right(folio_lbl)]
            f_blks.sort(key=lambda b: self.x_left(b))
            if f_blks: folio = f_blks[0]['text']

        # Face value: label "Face value", value is FIRST block to its right (not "Number of Securities")
        # Key: face value is in LEFT half of form; "Number of Securities" label is in RIGHT half
        fv_lbl = self.find(sec_b_blks, "Face value")
        fv = ""
        if fv_lbl and folio_lbl:
            # Face value is left of the Folio No. label — use Folio X as boundary
            mid_x = self.x_left(folio_lbl) - 20
            fv_blks = [b for b in sec_b_blks
                       if abs(self.cy(b) - self.cy(fv_lbl)) < 20
                       and self.x_left(b) > self.x_right(fv_lbl)
                       and self.cx(b) < mid_x
                       and not any(p in b['text'].lower() for p in ['number', 'securities', 'face'])
                       and len(b['text']) < 20]  # values are short
            fv_blks.sort(key=lambda b: self.x_left(b))
            if fv_blks: fv = fv_blks[0]['text']

        # Number of securities: label in right half, value to its right
        qty_lbl = self.find(sec_b_blks, "Number of Securities")
        qty = ""
        if qty_lbl:
            q_blks = [b for b in sec_b_blks
                      if abs(self.cy(b) - self.cy(qty_lbl)) < 20
                      and self.x_left(b) > self.x_right(qty_lbl)]
            q_blks.sort(key=lambda b: self.x_left(b))
            if q_blks: qty = q_blks[0]['text']

        # Distinctive From / To:
        # From debug: "From" and "To" labels are at similar Y.
        # Numbers appear ~20px BELOW those labels, at same X-centers.
        from_lbl = self.find(sec_b_blks, "From")
        to_lbl   = self.find(sec_b_blks, "To")
        d_from, d_to = "", ""
        if from_lbl and to_lbl:
            from_cx = self.cx(from_lbl)
            to_cx   = self.cx(to_lbl)
            label_y = (self.cy(from_lbl) + self.cy(to_lbl)) / 2
            # Numbers are below the labels (within 60px) and have 5+ digits
            num_blks = [b for b in sec_b_blks
                        if self.cy(b) > label_y + 5
                        and self.cy(b) < label_y + 70
                        and re.search(r'\d{5,}', b['text'])]
            for nb in num_blks:
                if abs(self.cx(nb) - from_cx) < abs(self.cx(nb) - to_cx):
                    d_from = nb['text']
                else:
                    d_to = nb['text']

        # Email and Mobile: search across entire sec_b region
        email_lbl = self.find(sec_b_blks, "E-mail Address")
        mobile_lbl = self.find(sec_b_blks, "Mobile Number")
        email, mobile = "", ""
        if email_lbl:
            e_blks = [b for b in sec_b_blks
                      if abs(self.cy(b) - self.cy(email_lbl)) < 25
                      and self.x_left(b) > self.x_right(email_lbl)]
            e_blks.sort(key=lambda b: self.x_left(b))
            if e_blks: email = e_blks[0]['text']
        if mobile_lbl:
            m_blks = [b for b in sec_b_blks
                      if abs(self.cy(b) - self.cy(mobile_lbl)) < 25
                      and self.x_left(b) > self.x_right(mobile_lbl)]
            m_blks.sort(key=lambda b: self.x_left(b))
            if m_blks: mobile = m_blks[0]['text']

        # ── 3. SECTION C — PAN Table ────────────────────────────────────
        # From actual dump:
        #   PAN blocks at X≈499-508, names at X≈58-179
        #   Each holder row: [row_num(X~12)] [name parts (X~58-179)] [PAN(X~500)] [Yes/No(X~645)]
        pan_re = re.compile(r'[A-Z]{3,5}\d{3,4}[A-Z0-9]{1,2}')  # flexible to handle OCR errors
        sec_c_blks = self.in_band(blocks, sec_c_y if sec_c_y else folioN_y + 200, bank_y)

        holders_c = []
        pan_blocks = sorted([b for b in sec_c_blks if pan_re.search(b['text'])
                             and len(b['text']) >= 9], key=lambda b: self.cy(b))

        for pb in pan_blocks:
            py = self.cy(pb)
            px = self.cx(pb)
            # Name: same row, X between 30 and (pan_x - 50)
            name_blks = [b for b in sec_c_blks
                         if abs(self.cy(b) - py) < 20
                         and 30 < self.cx(b) < px - 40
                         and len(b['text']) > 1
                         and not b['text'].strip().isdigit()
                         and '.' not in b['text']]
            name = re.sub(r'^\d+[\.\s]+', '', " ".join(
                b['text'] for b in sorted(name_blks, key=lambda b: self.cx(b)))).strip()

            # Aadhaar: right of PAN
            aad_blks = [b for b in sec_c_blks
                        if abs(self.cy(b) - py) < 20 and self.cx(b) > px + 50]
            aad = "Yes" if aad_blks and "yes" in aad_blks[0]['text'].lower() else "No"

            pan_val = pb['text'].strip()
            holders_c.append({"name": name, "pan": pan_val, "aadhaar": aad})

        # ── 4. BANK DETAILS ─────────────────────────────────────────────
        # From actual dump:
        #   IFSC label X≈522, UBIN951237 X≈616 — same row Y≈757
        #   UNION(X:183) BANK(X:274) Of(X:336) INDIA(X:408) — same row
        #   Bank label "Name of the Bank &" at X≈68 — far left, exclude
        bank_blks = self.in_band(blocks, bank_y, demat_y if demat_y else bank_y + 300)
        ifsc_lbl = self.find(bank_blks, "IFSC")
        bank_name, ifsc, acc_no, acc_type, bank_cat = "", "", "", "Savings", ""

        if ifsc_lbl:
            row_y = self.cy(ifsc_lbl)
            # IFSC value: right of label
            ifsc_blks = [b for b in bank_blks
                         if abs(self.cy(b) - row_y) < 25
                         and self.x_left(b) > self.x_right(ifsc_lbl)]
            ifsc_blks.sort(key=lambda b: self.x_left(b))
            if ifsc_blks: ifsc = ifsc_blks[0]['text']

            # Bank name: same row, skip far-left label text and known label phrases
            _bank_lbl = ['name of the bank', 'name of the', 'branch', 'ifsc']
            name_blks = [b for b in bank_blks
                         if abs(self.cy(b) - row_y) < 25
                         and self.cx(b) > 100
                         and self.cx(b) < self.cx(ifsc_lbl) - 30
                         and not any(p in b['text'].lower() for p in _bank_lbl)]
            bank_name = " ".join(b['text'] for b in sorted(name_blks, key=lambda b: self.cx(b)))

        # Account number
        acc_blks = [b for b in bank_blks if re.search(r'\d{12,18}', b['text'].replace(" ", ""))]
        if acc_blks: acc_no = acc_blks[0]['text'].replace(" ", "")

        # Account type: look for "Savings" specifically
        all_bank = " ".join(b['text'] for b in bank_blks).lower()
        if "savings" in all_bank: acc_type = "Savings"
        elif "current" in all_bank: acc_type = "Current"

        # Bank category
        bank_cat_txt = " ".join(b['text'] for b in bank_blks).upper()
        if "NRO" in bank_cat_txt: bank_cat = "NRO"
        elif "NRE" in bank_cat_txt: bank_cat = "NRE"

        # ── 5. DEMAT ────────────────────────────────────────────────────
        # From dump: "IN300212-" at X:436 and "30432688" at X:548 — two blocks!
        demat_id = ""
        demat_region = self.in_band(blocks, demat_y - 20, demat_y + 80) if demat_y else blocks
        demat_txt = " ".join(b['text'] for b in sorted(demat_region, key=lambda x: self.cx(x)))
        dm = re.search(r'IN\s*\d{6}\s*[-–]\s*\d{8}', demat_txt.replace(" ", " "))
        if not dm:
            # Try combining adjacent blocks
            dm = re.search(r'(IN\d{6}[-–]?\s*\d{0,8})', demat_txt)
        if dm: demat_id = dm.group().strip()
        # Clean up spacing
        demat_id = re.sub(r'IN(\d{6})\s*[-–]\s*(\d{8})', r'IN\1 - \2', demat_id)

        # ── 6. DECLARATION TABLE (Bottom Holder Info) ───────────────────
        # From actual dump:
        #   Column headers at Y≈968-978:
        #     First Named Holder X≈100, Joint Holder-1 X≈284, Joint Holder-2 X≈462, Joint Holder-3 X≈630
        #   Names at Y≈1098: RAJ SINGH X≈96, SHYAM+DONGRE X≈238+318, ABHINAV VARBUDENAMAN X≈506, T SHA X≈656
        #   Addresses: Y≈1138-1206
        #   PINs at Y≈1220-1242: 400006 X≈98, 459318 X≈286, 400007 X≈462, 412379 X≈625

        decl_end = annex_y if annex_y else decl_y + 1200
        decl_blks = self.in_band(blocks, decl_y, decl_end)

        # Find column headers dynamically
        fnh  = self.find(decl_blks, "First Named Holder")
        jh1  = self.find(decl_blks, "Joint Holder - 1") or self.find(decl_blks, "Joint Holder - 1")
        jh2  = self.find(decl_blks, "Joint Holder - 2")
        jh3  = self.find(decl_blks, "Joint Holder - 3")

        col_types = ["First Holder", "Joint Holder 1", "Joint Holder 2", "Joint Holder 3"]
        holder_info = []

        if fnh:
            header_y = self.cy(fnh)
            col_cx = [
                self.cx(fnh),
                self.cx(jh1) if jh1 else self.cx(fnh) + 185,
                self.cx(jh2) if jh2 else self.cx(fnh) + 365,
                self.cx(jh3) if jh3 else self.cx(fnh) + 530,
            ]

            # Data rows: everything below header (header_y + 20)
            data_blks = self.in_band(decl_blks, header_y + 20, decl_end)
            rows = self.group_rows(data_blks, tol=14)

            # Classify each row by its primary content
            # PINs: rows with 6-digit numbers
            pin_rows_y = [self.cy(r[0]) for r in rows if any(re.match(r'^\d{6}$', b['text']) for b in r)]
            pin_y = min(pin_rows_y) if pin_rows_y else decl_end - 50

            for i, cx in enumerate(col_cx):
                half = 80

                # NAME: ALL-CAPS text in this column, appearing FIRST below header
                # Skip: short noise, lowercase/mixed text (signatures), column headers
                name_val = ""
                addr_parts = []
                pin_val = ""

                for row in rows:
                    col_blks = sorted([b for b in row if abs(self.cx(b) - cx) < half],
                                      key=lambda b: self.cx(b))
                    if not col_blks: continue

                    row_txt = " ".join(b['text'] for b in col_blks).strip()
                    row_cy  = self.cy(row[0])

                    # Skip column headers and noise
                    if any(kw in row_txt.lower() for kw in
                           ['joint holder', 'first named', 'declaration', 'authorization',
                            'nand', 'srne', 'naeb', 'aizat']):
                        continue
                    # Skip short noise (1-3 chars, lowercase)
                    if len(row_txt) <= 3 and not row_txt.isupper():
                        continue

                    # PIN row
                    if re.search(r'\d{6}', row_txt) and row_cy >= pin_y - 30:
                        m = re.search(r'\d{6}', row_txt)
                        if m: pin_val = m.group()
                        continue

                    # ALL-CAPS text = name or address
                    # Signature artifacts are usually mixed-case garbled text
                    cleaned = row_txt.replace(" ", "")
                    is_caps = cleaned.isupper() and len(cleaned) > 3
                    if not is_caps and not name_val:
                        continue  # Skip signature area noise

                    if not name_val:
                        name_val = row_txt
                    else:
                        addr_parts.append(row_txt)

                # Merge PAN/Aadhaar from Section C, and prefer Section C names for better accuracy
                pan_val, aadh_val = "", "No"
                if i < len(holders_c):
                    pan_val  = holders_c[i]['pan']
                    aadh_val = holders_c[i]['aadhaar']
                    if holders_c[i]['name']:
                        name_val = holders_c[i]['name']

                holder_info.append({
                    "holder_type": col_types[i],
                    "name": name_val,
                    "pan": pan_val,
                    "aadhaar_linked": aadh_val,
                    "address": " ".join(addr_parts),
                    "pin_code": pin_val,
                    "signature_present": True,
                })
        else:
            # Fallback: use Section C data only
            for i in range(4):
                hc = holders_c[i] if i < len(holders_c) else {}
                holder_info.append({
                    "holder_type": col_types[i],
                    "name": hc.get("name", ""),
                    "pan": hc.get("pan", ""),
                    "aadhaar_linked": hc.get("aadhaar", "No"),
                    "address": "", "pin_code": "", "signature_present": True,
                })

        # ── 7. ANNEXURE ─────────────────────────────────────────────────
        annexure = []
        if annex_a:
            ann_blks = self.in_band(blocks, annex_y, annex_y + 2000)
            comp_h = self.find(ann_blks, "Name of the Issuer") or self.find(ann_blks, "Issuer Company")
            folio_h = self.find(ann_blks, "Folio No")
            qty_h = self.find(ann_blks, "Quantity")
            fv_h = self.find(ann_blks, "Face Value")
            sr_h = self.find(ann_blks, "Sr.No") or self.find(ann_blks, "Sr No")
            if comp_h:
                header_y = self.cy(comp_h)
                cols = {
                    "sr": self.cx(sr_h) if sr_h else self.cx(comp_h) - 100,
                    "company": self.cx(comp_h),
                    "folio": self.cx(folio_h) if folio_h else self.cx(comp_h) + 150,
                    "qty": self.cx(qty_h) if qty_h else self.cx(comp_h) + 280,
                    "fv": self.cx(fv_h) if fv_h else self.cx(comp_h) + 380,
                }
                data = self.in_band(ann_blks, header_y + 10, annex_y + 2000)
                for row in self.group_rows(data, tol=18):
                    rm = {}
                    for b in row:
                        key = min(cols, key=lambda k: abs(self.cx(b) - cols[k]))
                        rm.setdefault(key, []).append(b['text'])
                    comp = " ".join(rm.get("company", []))
                    fol  = " ".join(rm.get("folio", []))
                    if not comp or not fol: continue
                    if any(w in comp.lower() for w in ['issuer', 'name', 'note', 'author']): continue
                    sr_t = " ".join(rm.get("sr", []))
                    qt_t = " ".join(rm.get("qty", []))
                    try: sr = int(re.search(r'\d+', sr_t).group())
                    except: sr = len(annexure) + 1
                    try: qt = int(re.search(r'\d+', qt_t).group())
                    except: qt = None
                    annexure.append({"sr_no": sr, "issuer_company": comp,
                                     "folio_no": fol, "quantity": qt,
                                     "face_value": " ".join(rm.get("fv", []))})

        return {
            "form_metadata": {"form_type": "Form ISR-1", "date": date,
                              "requested_updates": req_updates},
            "security_details": {"issuer_company": issuer, "folio_no": folio,
                                 "face_value_per_security": fv, "number_of_securities": qty,
                                 "distinctive_numbers": {"from": d_from, "to": d_to}},
            "contact_details": {"email": email, "mobile": mobile},
            "bank_account_details": {"bank_name_branch": bank_name, "ifsc": ifsc,
                                     "account_no": acc_no, "account_type": acc_type,
                                     "bank_category": bank_cat},
            "demat_details": {"dpid_client_id": demat_id},
            "holders_information": holder_info,
            "annexure_details": annexure,
        }
