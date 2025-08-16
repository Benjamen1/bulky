# pipeline/sheets_updater.py
"""Google Sheets updater for processed data"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class SheetsUpdater:
    def __init__(self, credentials_dict: dict):
        """Initialize Google Sheets client"""
        self.creds = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        self.gc = gspread.authorize(self.creds)
        self.spreadsheet = None
        self.sheets = {}
        
    def connect(self, spreadsheet_name: str = "Bulking Tracker"):
        """Connect to the spreadsheet"""
        try:
            self.spreadsheet = self.gc.open(spreadsheet_name)
            
            # Map all worksheets
            for worksheet in self.spreadsheet.worksheets():
                self.sheets[worksheet.title] = worksheet
            
            print(f"✅ Connected to {spreadsheet_name}")
            print(f"   Found sheets: {list(self.sheets.keys())}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def get_processed_files(self) -> List[str]:
        """Get list of already processed file IDs"""
        try:
            worksheet = self.sheets.get('processed_files')
            if not worksheet:
                return []
            
            records = worksheet.get_all_records()
            # Only return successfully processed files
            return [r['file_id'] for r in records if r.get('status') == 'success']
        except Exception as e:
            print(f"Error getting processed files: {e}")
            return []
    
    def update_nutrition_data(self, df: pd.DataFrame) -> bool:
        """Update nutrition data, avoiding duplicates"""
        try:
            worksheet = self.sheets.get('daily_nutrition')
            if not worksheet:
                print("❌ daily_nutrition sheet not found")
                return False
            
            # Get existing dates
            existing_records = worksheet.get_all_records()
            existing_dates = set(r['date'] for r in existing_records if r.get('date'))
            
            # Filter out existing dates
            df['date'] = df['date'].astype(str)
            new_data = df[~df['date'].isin(existing_dates)]
            
            if new_data.empty:
                print("ℹ️ No new nutrition data to add")
                return True
            
            # Prepare data for insertion
            rows_to_add = []
            for _, row in new_data.iterrows():
                row_data = [
                    row.get('date', ''),
                    row.get('calories', 0),
                    row.get('protein', 0),
                    row.get('carbs_total', 0),
                    row.get('carbs_fiber', 0),
                    row.get('carbs_sugar', 0),
                    row.get('fat_total', 0),
                    row.get('fat_saturated', 0),
                    row.get('fat_unsaturated', 0),
                    row.get('cholesterol', 0),
                    row.get('sodium', 0),
                    row.get('potassium', 0),
                    row.get('source_file', ''),
                    row.get('extracted_at', datetime.now().isoformat())
                ]
                rows_to_add.append(row_data)
            
            # Batch append
            if rows_to_add:
                worksheet.append_rows(rows_to_add)
                print(f"✅ Added {len(rows_to_add)} new nutrition records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating nutrition data: {e}")
            return False
    
    def mark_file_processed(self, file_info: Dict, status: str = "success", 
                           rows_extracted: int = 0, error_msg: str = "") -> bool:
        """Mark a file as processed"""
        try:
            worksheet = self.sheets.get('processed_files')
            if not worksheet:
                print("❌ processed_files sheet not found")
                return False
            
            row_data = [
                file_info.get('id', ''),
                file_info.get('name', ''),
                datetime.now().isoformat(),
                status,
                rows_extracted,
                error_msg
            ]
            
            worksheet.append_row(row_data)
            print(f"✅ Marked {file_info.get('name', 'file')} as processed")
            return True
            
        except Exception as e:
            print(f"❌ Error marking file processed: {e}")
            return False
    
    def update_weekly_averages(self) -> bool:
        """Calculate and update weekly averages"""
        try:
            # Get nutrition and weight data
            nutrition_records = self.sheets['daily_nutrition'].get_all_records()
            weight_records = self.sheets['daily_weight'].get_all_records()
            
            if not nutrition_records or not weight_records:
                print("ℹ️ Not enough data for weekly averages")
                return True
            
            # Convert to DataFrames
            nutrition_df = pd.DataFrame(nutrition_records)
            weight_df = pd.DataFrame(weight_records)
            
            # Process dates
            nutrition_df['date'] = pd.to_datetime(nutrition_df['date'])
            weight_df['date'] = pd.to_datetime(weight_df['date'])
            
            # Calculate weekly averages
            nutrition_df['week'] = nutrition_df['date'].dt.to_period('W')
            weight_df['week'] = weight_df['date'].dt.to_period('W')
            
            # Group by week
            weekly_nutrition = nutrition_df.groupby('week').agg({
                'calories': 'mean',
                'protein': 'mean',
                'carbs_total': 'mean',
                'fat_total': 'mean'
            }).round(1)
            
            weekly_weight = weight_df.groupby('week').agg({
                'weight_kg': ['mean', 'first', 'last']
            }).round(2)
            
            # Calculate weight change
            weekly_weight['weight_change'] = weekly_weight['weight_kg']['last'] - weekly_weight['weight_kg']['first']
            
            # Clear and update weekly averages sheet
            worksheet = self.sheets['weekly_averages']
            worksheet.clear()
            
            # Add headers
            headers = ['week_ending', 'avg_weight', 'weight_change', 'avg_calories', 
                      'avg_protein', 'avg_carbs', 'avg_fat', 'training_days', 'notes']
            worksheet.update('A1', [headers])
            
            # Add data
            rows_to_add = []
            for week in weekly_nutrition.index:
                if week in weekly_weight.index:
                    row = [
                        str(week.end_time.date()),
                        float(weekly_weight.loc[week, ('weight_kg', 'mean')]),
                        float(weekly_weight.loc[week, 'weight_change']),
                        float(weekly_nutrition.loc[week, 'calories']),
                        float(weekly_nutrition.loc[week, 'protein']),
                        float(weekly_nutrition.loc[week, 'carbs_total']),
                        float(weekly_nutrition.loc[week, 'fat_total']),
                        '',  # training_days (can be added later)
                        ''   # notes
                    ]
                    rows_to_add.append(row)
            
            if rows_to_add:
                worksheet.append_rows(rows_to_add)
                print(f"✅ Updated {len(rows_to_add)} weeks of averages")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating weekly averages: {e}")
            return False
    
    def get_settings(self) -> Dict[str, str]:
        """Get settings from the settings sheet"""
        try:
            worksheet = self.sheets.get('settings')
            if not worksheet:
                return {}
            
            records = worksheet.get_all_records()
            return {r['setting_name']: r['value'] for r in records}
        except Exception as e:
            print(f"Error getting settings: {e}")
            return {}