from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "text": text,
                "metadata": {
                    "source": file_path,
                    "page": i + 1,
                    "type": "pdf"
                }
            })

    return documents