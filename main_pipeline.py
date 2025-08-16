#!/usr/bin/env python3
"""
Main pipeline script to process Lifesum PDFs from Google Drive
This runs daily via GitHub Actions
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List

# Add pipeline directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.drive_handler import DriveHandler
from pipeline.pdf_processor import LifesumPDFProcessor
from pipeline.sheets_updater import SheetsUpdater


class BulkingDataPipeline:
    def __init__(self, credentials_dict: dict):
        """Initialize all components"""
        self.drive = DriveHandler(credentials_dict)
        self.processor = LifesumPDFProcessor()
        self.sheets = SheetsUpdater(credentials_dict)
        self.stats = {
            'files_processed': 0,
            'records_added': 0,
            'errors': []
        }
    
    def run(self, folder_id: str = None) -> Dict:
        """Run the complete pipeline"""
        print(f"\n{'='*60}")
        print(f"🚀 Bulking Data Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")
        
        # Connect to sheets
        if not self.sheets.connect():
            self.stats['errors'].append("Failed to connect to Google Sheets")
            return self.stats
        
        # Get folder ID from settings if not provided
        if not folder_id:
            settings = self.sheets.get_settings()
            folder_id = settings.get('drive_folder_id')
            
            if not folder_id or folder_id == 'YOUR_FOLDER_ID_HERE':
                print("❌ Error: No Google Drive folder ID configured")
                print("   Please update 'drive_folder_id' in the settings sheet")
                self.stats['errors'].append("No folder ID configured")
                return self.stats
        
        print(f"📁 Using Drive folder: {folder_id}\n")
        
        # Get list of already processed files
        processed_file_ids = self.sheets.get_processed_files()
        print(f"📊 Found {len(processed_file_ids)} previously processed files\n")
        
        # List PDFs in Drive folder
        pdf_files = self.drive.list_pdfs_in_folder(folder_id)
        
        if not pdf_files:
            print("ℹ️ No PDF files found in folder")
            return self.stats
        
        # Filter out already processed files
        new_files = [f for f in pdf_files if f['id'] not in processed_file_ids]
        
        if not new_files:
            print("✅ All files already processed!")
            return self.stats
        
        print(f"📥 Found {len(new_files)} new files to process:\n")
        for f in new_files:
            print(f"   - {f['name']}")
        print()
        
        # Process each new file
        for file_info in new_files:
            print(f"\n{'='*40}")
            print(f"Processing: {file_info['name']}")
            print(f"{'='*40}")
            
            try:
                # Download PDF
                print("⬇️ Downloading from Drive...")
                pdf_bytes = self.drive.download_file(file_info['id'])
                
                # Process PDF
                print("📄 Extracting nutrition data...")
                nutrition_df = self.processor.process_pdf_bytes(
                    pdf_bytes, 
                    file_info['name']
                )
                
                if nutrition_df.empty:
                    print("⚠️ No data extracted from PDF")
                    self.sheets.mark_file_processed(
                        file_info, 
                        status="no_data",
                        error_msg="No nutrition data found in PDF"
                    )
                    continue
                
                # Update nutrition data
                print("💾 Saving to Google Sheets...")
                if self.sheets.update_nutrition_data(nutrition_df):
                    rows_extracted = len(nutrition_df)
                    self.stats['files_processed'] += 1
                    self.stats['records_added'] += rows_extracted
                    
                    # Mark as successfully processed
                    self.sheets.mark_file_processed(
                        file_info,
                        status="success",
                        rows_extracted=rows_extracted
                    )
                    
                    print(f"✅ Successfully processed {rows_extracted} records")
                else:
                    self.sheets.mark_file_processed(
                        file_info,
                        status="error",
                        error_msg="Failed to save to Sheets"
                    )
                    self.stats['errors'].append(f"Failed to save {file_info['name']}")
                    
            except Exception as e:
                error_msg = f"Error processing {file_info['name']}: {str(e)}"
                print(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
                
                # Mark as failed
                self.sheets.mark_file_processed(
                    file_info,
                    status="error",
                    error_msg=str(e)[:200]  # Truncate long errors
                )
        
        # Update weekly averages if we added new data
        if self.stats['records_added'] > 0:
            print("\n📊 Updating weekly averages...")
            self.sheets.update_weekly_averages()
        
        # Print summary
        print(f"\n{'='*60}")
        print("📈 Pipeline Summary")
        print(f"{'='*60}")
        print(f"✅ Files processed: {self.stats['files_processed']}")
        print(f"📊 Records added: {self.stats['records_added']}")
        if self.stats['errors']:
            print(f"⚠️ Errors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
        
        return self.stats


def main():
    """Main entry point for the pipeline"""
    
    # Get credentials from environment or file
    if 'GOOGLE_CREDENTIALS' in os.environ:
        # Running in GitHub Actions
        credentials_dict = json.loads(os.environ['GOOGLE_CREDENTIALS'])
    elif os.path.exists('credentials.json'):
        # Local development
        with open('credentials.json', 'r') as f:
            credentials_dict = json.load(f)
    else:
        print("❌ Error: No Google credentials found")
        print("   Set GOOGLE_CREDENTIALS env var or create credentials.json")
        sys.exit(1)
    
    # Get optional folder ID from environment
    folder_id = os.environ.get('DRIVE_FOLDER_ID')
    
    # Run pipeline
    pipeline = BulkingDataPipeline(credentials_dict)
    stats = pipeline.run(folder_id)
    
    # Exit with error if there were failures
    if stats['errors']:
        sys.exit(1)


if __name__ == "__main__":
    main()