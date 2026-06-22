import sys
sys.path.insert(0, '.')
from pipeline.parser import extract_text

text = extract_text(r'C:\Users\radha\Desktop\gopi__resumesmbc.docx')
print("=== RAW RESUME TEXT (first 3000 chars) ===")
print(repr(text[:3000]))
print()
print("=== LINES AROUND 'PROJECT' ===")
lines = text.splitlines()
for i, line in enumerate(lines):
    if 'project' in line.lower() or 'pricing' in line.lower() or 'chatbot' in line.lower() or 'amazon' in line.lower():
        start = max(0, i-1)
        end = min(len(lines), i+6)
        print(f"--- Line {i} context ---")
        for l in lines[start:end]:
            print(f"  |{repr(l)}")
        print()
