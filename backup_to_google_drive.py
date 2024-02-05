import os
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Set the scope to the required API access
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    creds = None

    # The file token.json stores the user's access and refresh tokens
    # It is created automatically when the authorization flow completes for the first time
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

def create_folder(service, folder_name, parent_folder_id=None):

    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }

    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]

    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')

def upload_file(service, local_file_path, folder_id):
    media = MediaFileUpload(local_file_path, resumable=True)
    file_metadata = {
        'name': os.path.basename(local_file_path),
        'parents': [folder_id]
    }

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File uploaded: {file.get("id")}')

def backup_to_google_drive(parent_folder_id, local_folder_path, drive_folder_name):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    folder_id = create_folder(service, drive_folder_name, parent_folder_id=parent_folder_id)

    # Recursively upload files from the local folder to the Google Drive folder
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            upload_file(service, local_file_path, folder_id)

if __name__ == '__main__':
    current_datetime = datetime.now()
    parent_folder_id = '1OETVNj1z8CXX-AAIwGVh2v7QIfMWePv7'
    local_folder_path = './images'
    drive_folder_name = 'MLImages-backup-%s' % (current_datetime.strftime("%y%m%d-%H%M%S"))


    print ("Backing up %s to %s" % (local_folder_path, drive_folder_name))
    backup_to_google_drive(parent_folder_id, local_folder_path, drive_folder_name)
