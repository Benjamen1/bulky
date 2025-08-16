#!/usr/bin/env python3
"""
Setup script to create/update Google Sheets structure
Run this once to set up your Bulking Tracker spreadsheet
"""

import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime

def setup_sheets(credentials_path):
    """Create or update the Bulking Tracker spreadsheet structure"""
    
    # Authenticate
    creds = Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive']
    )
    gc = gspread.authorize(creds)
    
    # Open or create spreadsheet
    try:
        spreadsheet = gc.open("Bulking Tracker")
        print("✅ Found existing Bulking Tracker spreadsheet")
    except gspread.SpreadsheetNotFound:
        spreadsheet = gc.create("Bulking Tracker")
        print("✅ Created new Bulking Tracker spreadsheet")
    
    # Define sheet structures
    sheets_config = {
        'daily_weight': {
            'headers': ['date', 'weight_kg', 'body_fat_percent', 'notes', 'recorded_at'],
            'sample_data': [
                ['2025-01-01', 82.5, 15.0, 'Morning weight', datetime.now().isoformat()],
            ]
        },
        'daily_nutrition': {
            'headers': ['date', 'calories', 'protein', 'carbs_total', 'carbs_fiber', 
                       'carbs_sugar', 'fat_total', 'fat_saturated', 'fat_unsaturated',
                       'cholesterol', 'sodium', 'potassium', 'source_file', 'extracted_at'],
            'sample_data': []
        },
        'processed_files': {
            'headers': ['file_id', 'file_name', 'processed_at', 'status', 'rows_extracted', 'error_message'],
            'sample_data': []
        },
        'weekly_averages': {
            'headers': ['week_ending', 'avg_weight', 'weight_change', 'avg_calories', 
                       'avg_protein', 'avg_carbs', 'avg_fat', 'training_days', 'notes'],
            'sample_data': []
        },
        'settings': {
            'headers': ['setting_name', 'value', 'updated_at'],
            'sample_data': [
                ['target_calories', '3724', datetime.now().isoformat()],
                ['target_protein', '250', datetime.now().isoformat()],
                ['target_carbs', '231', datetime.now().isoformat()],
                ['target_fat', '200', datetime.now().isoformat()],
                ['drive_folder_id', 'YOUR_FOLDER_ID_HERE', datetime.now().isoformat()],
                ['weekly_goal_kg', '0.35', datetime.now().isoformat()],
            ]
        }
    }
    
    # Create/update each sheet
    existing_sheets = [ws.title for ws in spreadsheet.worksheets()]
    
    for sheet_name, config in sheets_config.items():
        if sheet_name in existing_sheets:
            worksheet = spreadsheet.worksheet(sheet_name)
            print(f"📊 Found existing sheet: {sheet_name}")
            
            # Check if headers need updating
            try:
                existing_headers = worksheet.row_values(1)
                if existing_headers != config['headers']:
                    print(f"   Updating headers for {sheet_name}")
                    worksheet.update('A1', [config['headers']])
            except:
                # Empty sheet, add headers
                worksheet.update('A1', [config['headers']])
        else:
            # Create new sheet
            worksheet = spreadsheet.add_worksheet(
                title=sheet_name,
                rows=1000,
                cols=len(config['headers'])
            )
            print(f"✅ Created new sheet: {sheet_name}")
            
            # Add headers
            worksheet.update('A1', [config['headers']])
            
            # Add sample data if provided
            if config['sample_data']:
                start_row = 2
                for i, row in enumerate(config['sample_data']):
                    worksheet.update(f'A{start_row + i}', [row])
    
    # Format the sheets
    print("\n🎨 Applying formatting...")
    
    # Make headers bold (requires additional API calls)
    for sheet_name in sheets_config.keys():
        worksheet = spreadsheet.worksheet(sheet_name)
        worksheet.format('A1:Z1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
        })
    
    # Share spreadsheet link
    print(f"\n✅ Setup complete!")
    print(f"📊 Spreadsheet URL: {spreadsheet.url}")
    print(f"\n⚠️  Don't forget to:")
    print(f"   1. Update the 'drive_folder_id' in the settings sheet")
    print(f"   2. Share the spreadsheet with your service account email")
    
    return spreadsheet.url

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python setup_sheets.py path/to/credentials.json")
        sys.exit(1)
    
    credentials_path = sys.argv[1]
    setup_sheets(credentials_path)