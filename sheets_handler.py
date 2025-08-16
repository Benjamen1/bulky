# sheets_handler.py
"""Google Sheets handler for Streamlit app"""

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
import json
from datetime import datetime, date

class SheetsHandler:
    def __init__(self):
        self.gc = None
        self.spreadsheet = None
        self._connect()
    
    def _connect(self):
        """Connect to Google Sheets using service account credentials"""
        try:
            # Get credentials from Streamlit secrets
            credentials_dict = st.secrets["gcp_service_account"]
            
            # Convert to proper format
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive.file",
                    "https://www.googleapis.com/auth/drive"
                ]
            )
            
            self.gc = gspread.authorize(credentials)
            
            # Open the spreadsheet by name
            self.spreadsheet = self.gc.open("Bulking Tracker")
            
        except Exception as e:
            st.error(f"Failed to connect to Google Sheets: {e}")
            st.error("Please check your credentials in Streamlit secrets.")
    
    def load_weight_data(self):
        """Load weight data from Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet("daily_weight")
            data = worksheet.get_all_records()
            
            if not data:
                return pd.DataFrame(columns=['date', 'weight_kg'])
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remove any rows with NaN weights
            df = df.dropna(subset=['weight_kg'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading weight data: {e}")
            return pd.DataFrame(columns=['date', 'weight_kg'])
    
    def load_nutrition_data(self):
        """Load nutrition data from Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet("daily_nutrition")
            data = worksheet.get_all_records()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Convert numeric columns
            numeric_columns = ['calories', 'protein', 'carbs_total', 'carbs_fiber', 
                             'carbs_sugar', 'fat_total', 'fat_saturated', 'fat_unsaturated',
                             'cholesterol', 'sodium', 'potassium']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remove duplicates (keep most recent)
            df = df.drop_duplicates(subset=['date'], keep='last')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading nutrition data: {e}")
            return pd.DataFrame()
    
    def save_weight_entry(self, entry_date, weight_kg, body_fat=None, notes=""):
        """Save or update a weight entry"""
        try:
            worksheet = self.spreadsheet.worksheet("daily_weight")
            
            # Convert date to string format
            if isinstance(entry_date, date):
                date_str = entry_date.strftime('%Y-%m-%d')
            else:
                date_str = str(entry_date)
            
            # Get all existing data
            existing_data = worksheet.get_all_records()
            
            # Check if date already exists
            row_to_update = None
            for i, row in enumerate(existing_data):
                if str(row.get('date', '')) == date_str:
                    row_to_update = i + 2  # +2 because sheets are 1-indexed and we skip header
                    break
            
            # Prepare row data
            row_data = [
                date_str,
                float(weight_kg),
                float(body_fat) if body_fat else '',
                notes,
                datetime.now().isoformat()
            ]
            
            if row_to_update:
                # Update existing row
                worksheet.update(f'A{row_to_update}:E{row_to_update}', [row_data])
                return f"Updated weight for {date_str}"
            else:
                # Add new row
                worksheet.append_row(row_data)
                return f"Added new weight entry for {date_str}"
                
        except Exception as e:
            st.error(f"Error saving weight data: {e}")
            return f"Error: {e}"
    
    def save_nutrition_data(self, nutrition_df):
        """Save nutrition data to Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet("daily_nutrition")
            
            # Get existing data to check for duplicates
            existing_data = worksheet.get_all_records()
            existing_dates = [str(row.get('date', '')) for row in existing_data]
            
            # Filter out dates that already exist
            new_data = []
            updated_count = 0
            
            for _, row in nutrition_df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                
                if date_str not in existing_dates:
                    # Convert row to list, ensuring date is string
                    row_data = [
                        date_str,
                        float(row.get('calories', 0)),
                        float(row.get('protein', 0)),
                        float(row.get('carbs_total', 0)),
                        float(row.get('carbs_fiber', 0)),
                        float(row.get('carbs_sugar', 0)),
                        float(row.get('fat_total', 0)),
                        float(row.get('fat_saturated', 0)),
                        float(row.get('fat_unsaturated', 0)),
                        float(row.get('cholesterol', 0)),
                        float(row.get('sodium', 0)),
                        float(row.get('potassium', 0)),
                        row.get('source_file', ''),
                        datetime.now().isoformat()
                    ]
                    new_data.append(row_data)
                else:
                    updated_count += 1
            
            if new_data:
                # Append new rows
                worksheet.append_rows(new_data)
                return f"Added {len(new_data)} new nutrition entries. Skipped {updated_count} existing dates."
            else:
                return f"No new data to add. {updated_count} dates already exist."
                
        except Exception as e:
            st.error(f"Error saving nutrition data: {e}")
            return f"Error: {e}"
    
    def get_sheet_info(self):
        """Get basic info about the sheets"""
        try:
            weight_ws = self.spreadsheet.worksheet("daily_weight")
            nutrition_ws = self.spreadsheet.worksheet("daily_nutrition")
            
            weight_count = len(weight_ws.get_all_records())
            nutrition_count = len(nutrition_ws.get_all_records())
            
            return {
                'weight_records': weight_count,
                'nutrition_records': nutrition_count,
                'connected': True
            }
        except Exception as e:
            return {
                'weight_records': 0,
                'nutrition_records': 0,
                'connected': False,
                'error': str(e)
            }