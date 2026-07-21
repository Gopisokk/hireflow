"""
Compare DOCX extraction using:

1. python-docx
2. docx2txt
3. mammoth
4. Apache Tika
5. unstructured

Resume:
C:\\Users\\radha\\Desktop\\gopi_iresume.docx
"""

from pathlib import Path

DOCX_PATH = r"C:\Users\radha\Desktop\gopi_iresume.docx"

if not Path(DOCX_PATH).exists():
    raise FileNotFoundError(f"File not found: {DOCX_PATH}")


# ==========================================================
# 1. python-docx
# ==========================================================

def extract_python_docx(path):
    from docx import Document

    doc = Document(path)

    text = "\n".join(
        para.text
        for para in doc.paragraphs
    )

    return text


# ==========================================================
# 2. docx2txt
# ==========================================================

def extract_docx2txt(path):
    import docx2txt

    return docx2txt.process(path)


# ==========================================================
# 3. mammoth
# ==========================================================

def extract_mammoth(path):
    import mammoth

    with open(path, "rb") as docx_file:
        result = mammoth.extract_raw_text(docx_file)

    return result.value


# ==========================================================
# 4. Apache Tika
# ==========================================================

def extract_tika(path):
    from tika import parser

    parsed = parser.from_file(path)

    content = parsed.get("content")

    return content if content else ""


# ==========================================================
# 5. unstructured
# ==========================================================

def extract_unstructured(path):
    from unstructured.partition.docx import partition_docx

    elements = partition_docx(filename=path)

    text = "\n".join(
        str(element)
        for element in elements
    )

    return text


# ==========================================================
# Run Benchmark
# ==========================================================

extractors = {
    "python-docx": extract_python_docx,
    "docx2txt": extract_docx2txt,
    "mammoth": extract_mammoth,
    "Apache Tika": extract_tika,
    "unstructured": extract_unstructured,
}


for name, extractor in extractors.items():

    print("\n" + "=" * 100)
    print(f"EXTRACTOR: {name}")
    print("=" * 100)

    try:
        text = extractor(DOCX_PATH)

        print(text[:10000])

        print("\n")
        print(f"Characters Extracted : {len(text)}")
        print(f"Words Extracted      : {len(text.split())}")

        with open(
            f"{name.replace(' ', '_')}.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(text)

        print(f"Saved Output         : {name}.txt")

    except Exception as e:
        print(f"ERROR: {e}")

print("\nBenchmark Complete.")