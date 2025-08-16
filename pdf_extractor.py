# pdf_extractor.py
"""PDF extractor for Streamlit app - processes Lifesum PDFs"""

import pdfplumber
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
import os
import io

class LifesumPDFExtractor:
    def __init__(self):
        self.daily_summaries = []
        self.meal_details = []
    
    def extract_pdf_data(self, pdf_path_or_bytes):
        """Extract data from a single Lifesum PDF"""
        
        # Reset data
        self.daily_summaries = []
        
        # Handle both file paths and bytes
        if isinstance(pdf_path_or_bytes, (str, Path)):
            print(f"Processing: {pdf_path_or_bytes}")
            pdf_file = pdf_path_or_bytes
        else:
            # It's bytes from uploaded file
            pdf_file = io.BytesIO(pdf_path_or_bytes)
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                
                if text:
                    # Extract daily summaries (the key data we need)
                    self._extract_daily_summaries(text, page_num)
    
    def _extract_daily_summaries(self, text, page_num=1):
        """Extract the daily summary rows"""
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines that contain "Summary for" followed by a date
            if 'Summary for' in line and '20' in line:
                try:
                    # Extract date
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                    if not date_match:
                        continue
                    
                    date_str = date_match.group(1)
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Extract all numbers after the date
                    numbers = re.findall(r'[\d.]+', line[date_match.end():])
                    
                    if len(numbers) >= 11:  # Ensure we have all the data columns
                        summary = {
                            'date': date,
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
                            'potassium': float(numbers[10]) if len(numbers) > 10 else 0
                        }
                        
                        self.daily_summaries.append(summary)
                        print(f"Extracted: {date} - {summary['calories']} calories, {summary['protein']}g protein")
                        
                except (ValueError, IndexError) as e:
                    print(f"Error parsing summary line on page {page_num}: {str(e)[:50]}")
                    continue
    
    def process_folder(self, folder_path):
        """Process all PDFs in a folder"""
        folder = Path(folder_path)
        pdf_files = list(folder.glob('*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in sorted(pdf_files):
            try:
                self.extract_pdf_data(pdf_file)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
    
    def get_daily_nutrition_df(self):
        """Return daily summaries as a pandas DataFrame"""
        if not self.daily_summaries:
            print("No daily summaries extracted. Check your PDF files.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.daily_summaries)
        
        # Remove duplicates, keeping the most recent
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Extracted {len(df)} unique daily nutrition records")
        return df
    
    def save_to_csv(self, output_dir='data'):
        """Save extracted data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save daily summaries
        daily_df = self.get_daily_nutrition_df()
        if not daily_df.empty:
            daily_path = Path(output_dir) / 'daily_nutrition.csv'
            daily_df.to_csv(daily_path, index=False)
            print(f"Saved daily nutrition data to: {daily_path}")
            return daily_path
        return None