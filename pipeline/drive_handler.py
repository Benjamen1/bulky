# pipeline/drive_handler.py
"""Google Drive handler for PDF management"""

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from typing import List, Dict, Optional
from datetime import datetime

class DriveHandler:
    def __init__(self, credentials_dict: dict):
        """Initialize Google Drive API client"""
        self.creds = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=[
                'https://www.googleapis.com/auth/drive.readonly',
                'https://www.googleapis.com/auth/drive.file',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        self.service = build('drive', 'v3', credentials=self.creds)
    
    def list_pdfs_in_folder(self, folder_id: str) -> List[Dict]:
        """List all PDFs in the specified folder"""
        try:
            results = self.service.files().list(
                q=f"'{folder_id}' in parents and mimeType='application/pdf'",
                fields="files(id, name, createdTime, size)",
                orderBy="createdTime desc"
            ).execute()
            
            files = results.get('files', [])
            print(f"Found {len(files)} PDFs in folder")
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_id: str) -> bytes:
        """Download a file from Google Drive"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download {int(status.progress() * 100)}%")
            
            file_buffer.seek(0)
            return file_buffer.read()
        except Exception as e:
            print(f"Error downloading file {file_id}: {e}")
            raise