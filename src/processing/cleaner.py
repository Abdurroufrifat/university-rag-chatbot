import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()