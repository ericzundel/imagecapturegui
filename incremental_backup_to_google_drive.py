"""Backup caputured image data to Google Drive. 

Intended to be called from cron for a periodic backup
"""
import os
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PARENT_FOLDER_ID = "1OETVNj1z8CXX-AAIwGVh2v7QIfMWePv7"
INCREMENTAL_BACKUP_FOLDER_NAME = "EngineeringConcepts2025"

# Set the scope to the required API access
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def authenticate():
    creds = None

    # The file token.json stores the user's access and refresh tokens
    # Created automatically when authorization flow completes for the first time
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json")

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def create_folder(service, folder_name, parent_folder_id=None):

    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }

    if parent_folder_id:
        file_metadata["parents"] = [parent_folder_id]

    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder.get("id")


def upload_file(service, local_file_path, folder_id):
    media = MediaFileUpload(local_file_path, resumable=True)
    file_metadata = {"name": os.path.basename(local_file_path), "parents": [folder_id]}

    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    print(f'File uploaded: {file.get("id")}')

    
def fetch_files_recursively(service, folder_id, results):
    """
    Recursively fetch all filenames in a folder.
    :param service: Google Drive API service instance.
    :param folder_id: The ID of the folder to fetch files from.
    :param results: A list to store filenames.
    """
    try:
        # Query to find files and folders in the current folder
        query = f"'{folder_id}' in parents and trashed = false"
        page_token = None
        
        while True:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            
            for file in response.get('files', []):
                results.append(file['name'])
                # If the file is a folder, recurse into it
                if file['mimeType'] == 'application/vnd.google-apps.folder':
                    fetch_files_recursively(service, file['id'], results)
            
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

    except HttpError as error:
        print(f"An error occurred: {error}")    

def incremental_backup_to_google_drive(parent_folder_id, local_folder_path,
                           drive_folder_name, depth=0):
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    
    # Create the top level directory
    if (depth == 0):
        folder_id = create_folder(
            service, drive_folder_name, parent_folder_id=parent_folder_id
        )
    else:
        folder_id = parent_folder_id

    # Fetch all existing files under the folder
    all_files = []
    fetch_files_recursively(service, folder_id, all_files)

    # Print all filenames
    print("Files and Folders:")
    for filename in all_files:
        print(filename)

    return

    # Recursively upload files from the local folder to the Google Drive folder
    for entry in os.scandir(local_folder_path):
        print(" got entry: %s on path %s" % (entry.name, local_folder_path))
        if entry.name == ".":
            pass
        if entry.is_dir():
            subfolder_id = create_folder(
                service, entry.name, parent_folder_id=folder_id
            )
            backup_to_google_drive(
                subfolder_id, os.path.join(local_folder_path, entry.name),
                entry.name, depth=depth+1
            )
        elif entry.is_file():
            local_file_path = os.path.join(local_folder_path, entry.name)
            upload_file(service, local_file_path, folder_id)
    
def backup_to_google_drive(parent_folder_id, local_folder_path,
                           drive_folder_name, depth=0):
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    # Create the top level directory
    if (depth == 0):
        folder_id = create_folder(
            service, drive_folder_name, parent_folder_id=parent_folder_id
        )
    else:
        folder_id = parent_folder_id

    # Recursively upload files from the local folder to the Google Drive folder
    for entry in os.scandir(local_folder_path):
        print(" got entry: %s on path %s" % (entry.name, local_folder_path))
        if entry.name == ".":
            pass
        if entry.is_dir():
            subfolder_id = create_folder(
                service, entry.name, parent_folder_id=folder_id
            )
            backup_to_google_drive(
                subfolder_id, os.path.join(local_folder_path, entry.name),
                entry.name, depth=depth+1
            )
        elif entry.is_file():
            local_file_path = os.path.join(local_folder_path, entry.name)
            upload_file(service, local_file_path, folder_id)


if __name__ == "__main__":
    current_datetime = datetime.now()
    parent_folder_id = PARENT_FOLDER_ID
    local_folder_path = "./images"
    drive_full_folder_name = "MLImages-backup-%s" % (
        current_datetime.strftime("%y%m%d-%H%M%S")
    )
    drive_incremental_folder_name = INCREMENTAL_BACKUP_FOLDER_NAME

    print("Incremental backup to %s" % (drive_incremental_folder_name))
    incremental_backup_to_google_drive(parent_folder_id, local_folder_path, drive_incremental_folder_name)
          
    print("Full backup up %s to %s" % (local_folder_path, drive_full_folder_name))
    backup_to_google_drive(parent_folder_id, local_folder_path, drive_full_folder_name)
