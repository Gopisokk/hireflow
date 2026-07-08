# pip install PyMuPDF transformers torch gliner Pillow

import os
import fitz  # PyMuPDF
import json
import torch
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from gliner import GLiNER
from PIL import Image
import io

class ResumeParser:
    """
    ML-Powered Resume Parser tailored for Software Engineers.
    Hardware Constraints: Optimized for < 4GB VRAM.
    """
    def __init__(self):
        print("Initializing ResumeParser...")
        
        # Load LayoutLMv3 (Base model)
        print("Loading LayoutLMv3 for layout segmentation...")
        self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        
        # Load GLiNER (Zero-shot NER)
        print("Loading GLiNER for entity extraction...")
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_base")
        
    def _extract_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks and their bounding boxes using PyMuPDF."""
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = page.get_text("blocks")
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            
            for b in page_blocks:
                if b[6] == 0:  # text block
                    text = b[4].strip()
                    if text:
                        # Normalize bbox to 0-1000 scale as expected by LayoutLM
                        x0, y0, x1, y1 = b[:4]
                        width, height = page.rect.width, page.rect.height
                        bbox = [
                            int(1000 * (x0 / width)),
                            int(1000 * (y0 / height)),
                            int(1000 * (x1 / width)),
                            int(1000 * (y1 / height))
                        ]
                        blocks.append({
                            "text": text,
                            "bbox": bbox,
                            "page_img": img,
                            "page_num": page_num
                        })
        return blocks
        
    def _segment_layout(self, blocks: List[Dict]) -> Dict[str, str]:
        """
        Classify and segment the document into specialized software engineer topics.
        """
        sections = {
            "Header": "",
            "Education": "",
            "Open Source": "",
            "Projects": "",
            "Competitive Programming": "",
            "Certifications": "",
            "Achievements": "",
            "Technologies": "",
            "Volunteering": ""
        }
        
        current_section = "Header"
        
        keywords = {
            "Education": ["education", "academic", "university", "institute", "college"],
            "Open Source": ["open source", "contributions", "oss"],
            "Projects": ["projects", "personal projects", "portfolio"],
            "Competitive Programming": ["competitive programming", "coding profiles", "leetcode", "codechef", "hackerrank"],
            "Certifications": ["certifications", "certificates", "courses"],
            "Achievements": ["achievements", "awards", "honors"],
            "Technologies": ["technologies", "skills", "languages", "frameworks", "tools"],
            "Volunteering": ["volunteering", "extracurricular", "social work", "leadership"]
        }
        
        for block in blocks:
            text_lower = block["text"].lower()
            
            try:
                encoding = self.layout_processor(
                    block["page_img"],
                    block["text"],
                    boxes=[block["bbox"]],
                    return_tensors="pt"
                )
                with torch.no_grad():
                    _ = self.layout_model(**encoding)
            except Exception:
                pass
            
            for sec, kw_list in keywords.items():
                if any(kw in text_lower[:40] for kw in kw_list) and len(text_lower) < 60:
                    current_section = sec
                    break
                    
            sections[current_section] += block["text"] + "\n\n"
            
        return sections
        
    def _extract_header(self, text: str) -> Dict[str, str]:
        """Extract Contact Info & Socials."""
        info = {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""}
        if not text.strip(): return info
            
        entities = self.gliner_model.predict_entities(text, ["Name", "Email", "Phone number", "LinkedIn URL", "GitHub URL"])
        for ent in entities:
            label = ent["label"].lower()
            if "name" in label and not info["name"]:
                info["name"] = ent["text"]
            elif "email" in label and not info["email"]:
                info["email"] = ent["text"]
            elif "phone" in label and not info["phone"]:
                info["phone"] = ent["text"]
            elif "linkedin" in label and not info["linkedin"]:
                info["linkedin"] = ent["text"]
            elif "github" in label and not info["github"]:
                info["github"] = ent["text"]
        return info
        
    def _extract_lists(self, text: str, labels: List[str], primary_key: str) -> List[Dict]:
        """Generic GLiNER extractor for list-based sections (e.g. Certifications, Projects)."""
        if not text.strip(): return []
        entities = self.gliner_model.predict_entities(text, labels)
        results = []
        for ent in entities:
            if ent["label"] == primary_key:
                # E.g. {"project_name": "HireFlow"}
                results.append({primary_key.lower().replace(" ", "_"): ent["text"]})
        return results

    def _extract_tech_skills(self, text: str) -> Dict[str, List[Dict]]:
        """Extract Primary and Secondary Skills with context windows."""
        result = {"primary_skills": [], "secondary_skills": []}
        if not text.strip(): return result
        
        entities = self.gliner_model.predict_entities(text, ["Primary Skill", "Secondary Skill", "Skill"])
        
        for ent in entities:
            label = ent["label"].lower()
            start = max(0, ent["start"] - 50)
            end = min(len(text), ent["end"] + 50)
            context = text[start:end].replace("\n", " ").strip()
            
            skill_entry = {
                "skill_span": ent["text"],
                "context": f"...{context}..."
            }
            
            if "secondary" in label or ("secondary" in text.lower() and text.lower().find("secondary") < ent["start"]):
                result["secondary_skills"].append(skill_entry)
            else:
                result["primary_skills"].append(skill_entry)
                
        return result

    def parse_pdf(self, pdf_path: str) -> Dict:
        """Full pipeline: Ingest -> Segment -> Parallel Extract -> Target JSON Schema"""
        try:
            blocks = self._extract_blocks(pdf_path)
            sections = self._segment_layout(blocks)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                f_header = executor.submit(self._extract_header, sections["Header"])
                
                f_edu = executor.submit(self._extract_lists, sections["Education"], ["Institution"], "Institution")
                f_oss = executor.submit(self._extract_lists, sections["Open Source"], ["Project Name"], "Project Name")
                f_proj = executor.submit(self._extract_lists, sections["Projects"], ["Project Name"], "Project Name")
                f_cp = executor.submit(self._extract_lists, sections["Competitive Programming"], ["Platform"], "Platform")
                
                f_cert = executor.submit(self._extract_lists, sections["Certifications"], ["Certification Name"], "Certification Name")
                f_achv = executor.submit(self._extract_lists, sections["Achievements"], ["Achievement Name"], "Achievement Name")
                f_vol = executor.submit(self._extract_lists, sections["Volunteering"], ["Organization"], "Organization")
                
                f_tech = executor.submit(self._extract_tech_skills, sections["Technologies"])
                
            return {
                "header": f_header.result(),
                "education": f_edu.result(),
                "open_source_contributions": f_oss.result(),
                "projects": f_proj.result(),
                "competitive_programming": f_cp.result(),
                "certifications": f_cert.result(),
                "achievements": f_achv.result(),
                "technologies": f_tech.result(),
                "volunteering": f_vol.result()
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                "header": {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""},
                "education": [],
                "open_source_contributions": [],
                "projects": [],
                "competitive_programming": [],
                "certifications": [],
                "achievements": [],
                "technologies": {"primary_skills": [], "secondary_skills": []},
                "volunteering": []
            }

    def process_directory(self, input_dir: str, output_dir: str):
        """Process a directory of PDFs and save JSON outputs."""
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                print(f"Processing {filename}...")
                
                result = self.parse_pdf(pdf_path)
                
                out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Saved {filename} to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML-Powered Resume ATS Parser")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing PDFs")
    parser.add_argument("--output", type=str, required=True, help="Output directory for JSON files")
    args = parser.parse_args()
    
    ats_parser = ResumeParser()
    ats_parser.process_directory(args.input, args.output)
