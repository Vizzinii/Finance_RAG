import pdfplumber
import re
import os
from typing import List

class DocumentLoader:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']

    def load_file(self, file_path: str) -> str:
        """Load content from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.supported_formats}")

        content = ""
        try:
            if ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")
        
        return self.clean_text(content)

    def clean_text(self, text: str) -> str:
        """Clean noise from financial reports."""
        if not text:
            return ""
            
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # 1. Strip whitespace
            line = line.strip()
            
            # 2. Skip empty lines
            if not line:
                continue
                
            # 3. Skip page numbers (e.g., "1", "Page 1", "1 / 10")
            if re.match(r'^page\s*\d+(\s*[/of-]\s*\d+)?$', line, re.IGNORECASE) or re.match(r'^\d+$', line):
                continue
                
            # 4. Skip common disclaimer headers (simplified)
            if "disclaimer" in line.lower() and len(line) < 50:
                continue
                
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
