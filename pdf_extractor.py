import pdfplumber
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
import os

class LifesumPDFExtractor:
    def __init__(self):
        self.daily_summaries = []
        self.meal_details = []
    
    def extract_pdf_data(self, pdf_path):
        """Extract data from a single Lifesum PDF"""
        print(f"Processing: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                
                # Extract daily summaries (the key data we need)
                self._extract_daily_summaries(text)
                
                # Extract detailed meal data (optional, for deeper analysis)
                self._extract_meal_details(text)
    
    def _extract_daily_summaries(self, text):
        """Extract the daily summary rows"""
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines that start with "Summary for YYYY-MM-DD"
            if line.startswith('Summary for 2025-'):
                parts = line.split()
                if len(parts) >= 13:  # Ensure we have all the data columns
                    try:
                        date_str = parts[2]  # "2025-07-28"
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        summary = {
                            'date': date,
                            'calories': float(parts[3]),
                            'carbs_total': float(parts[4]),
                            'carbs_fiber': float(parts[5]),
                            'carbs_sugar': float(parts[6]),
                            'fat_total': float(parts[7]),
                            'fat_saturated': float(parts[8]),
                            'fat_unsaturated': float(parts[9]),
                            'cholesterol': float(parts[10]),
                            'protein': float(parts[11]),
                            'potassium': float(parts[12]),
                            'sodium': float(parts[13])
                        }
                        
                        self.daily_summaries.append(summary)
                        print(f"Extracted: {date} - {summary['calories']} calories")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing summary line: {line}")
                        print(f"Error: {e}")
    
    def _extract_meal_details(self, text):
        """Extract individual meal entries (optional for detailed analysis)"""
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines with date pattern at start
            if re.match(r'^\d{4}-\d{2}-\d{2}\s+\w+', line):
                parts = line.split('\t') if '\t' in line else line.split()
                
                if len(parts) >= 6:  # Basic meal data
                    try:
                        date_str = parts[0]
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        meal = {
                            'date': date,
                            'meal_type': parts[1],
                            'food_item': parts[2],
                            'amount': parts[3],
                            'serving_unit': parts[4],
                            'grams': float(parts[5]) if parts[5].replace('.', '').isdigit() else 0,
                            'calories': float(parts[6]) if len(parts) > 6 and parts[6].replace('.', '').isdigit() else 0
                        }
                        
                        self.meal_details.append(meal)
                        
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines
    
    def process_folder(self, folder_path):
        """Process all PDFs in the lifesum folder"""
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
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Extracted {len(df)} daily nutrition records")
        return df
    
    def get_meal_details_df(self):
        """Return meal details as a pandas DataFrame"""
        if not self.meal_details:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.meal_details)
        df = df.sort_values(['date', 'meal_type']).reset_index(drop=True)
        
        return df
    
    def save_to_csv(self, output_dir='data'):
        """Save extracted data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save daily summaries (main data we need)
        daily_df = self.get_daily_nutrition_df()
        if not daily_df.empty:
            daily_path = Path(output_dir) / 'daily_nutrition.csv'
            daily_df.to_csv(daily_path, index=False)
            print(f"Saved daily nutrition data to: {daily_path}")
        
        # Save meal details (optional)
        meal_df = self.get_meal_details_df()
        if not meal_df.empty:
            meal_path = Path(output_dir) / 'meal_details.csv'
            meal_df.to_csv(meal_path, index=False)
            print(f"Saved meal details to: {meal_path}")


def main():
    """Example usage"""
    extractor = LifesumPDFExtractor()
    
    # Process all PDFs in the lifesum folder
    extractor.process_folder('data/lifesum')
    
    # Get the daily nutrition data
    daily_df = extractor.get_daily_nutrition_df()
    
    if not daily_df.empty:
        print("\nDaily Nutrition Summary:")
        print(daily_df[['date', 'calories', 'protein', 'carbs_total', 'fat_total']].head())
        
        # Save to CSV
        extractor.save_to_csv()
    else:
        print("No data extracted. Please check your PDF files.")


if __name__ == "__main__":
    main()