# pipeline/pdf_processor.py
"""Enhanced PDF processor for Lifesum exports"""

import pdfplumber
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional
import io

class LifesumPDFProcessor:
    def __init__(self):
        self.daily_summaries = []
        self.extraction_errors = []
    
    def process_pdf_bytes(self, pdf_bytes: bytes, filename: str = "unknown.pdf") -> pd.DataFrame:
        """Process PDF from bytes"""
        self.daily_summaries = []
        self.extraction_errors = []
        
        try:
            pdf_buffer = io.BytesIO(pdf_bytes)
            with pdfplumber.open(pdf_buffer) as pdf:
                print(f"Processing {filename}: {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        self._extract_daily_summaries(text, page_num)
                    else:
                        print(f"  Warning: No text found on page {page_num}")
            
            if self.daily_summaries:
                df = pd.DataFrame(self.daily_summaries)
                df['source_file'] = filename
                df['extracted_at'] = datetime.now().isoformat()
                
                # Remove duplicates keeping latest
                df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                
                print(f"✅ Extracted {len(df)} unique daily records from {filename}")
                return df
            else:
                print(f"⚠️ No data extracted from {filename}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            self.extraction_errors.append(str(e))
            return pd.DataFrame()
    
    def _extract_daily_summaries(self, text: str, page_num: int):
        """Extract daily summary data from PDF text"""
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines starting with "Summary for YYYY-MM-DD"
            if 'Summary for 20' in line:
                try:
                    # Extract date
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                    if not date_match:
                        continue
                    
                    date_str = date_match.group(1)
                    
                    # Extract all numbers from the line
                    numbers = re.findall(r'[\d.]+', line[date_match.end():])
                    
                    if len(numbers) >= 11:  # Ensure we have all nutrition fields
                        summary = {
                            'date': date_str,
                            'calories': float(numbers[0]),
                            'carbs_total': float(numbers[1]),
                            'carbs_fiber': float(numbers[2]),
                            'carbs_sugar': float(numbers[3]),
                            'fat_total': float(numbers[4]),
                            'fat_saturated': float(numbers[5]),
                            'fat_unsaturated': float(numbers[6]),
                            'cholesterol': float(numbers[7]),
                            'protein': float(numbers[8]),
                            'sodium': float(numbers[9]) if len(numbers) > 9 else 0,
                            'potassium': float(numbers[10]) if len(numbers) > 10 else 0,
                            'page_num': page_num
                        }
                        
                        self.daily_summaries.append(summary)
                        
                except (ValueError, IndexError) as e:
                    print(f"  Error parsing line on page {page_num}: {str(e)[:50]}")
                    continue