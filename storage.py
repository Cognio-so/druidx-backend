# storage.py
import boto3
import requests
import os
from typing import List, Optional, Tuple, Union
from botocore.exceptions import ClientError, EndpointConnectionError
import hashlib
from urllib.parse import urlparse
import shutil
from io import BytesIO
import httpx
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class CloudflareR2Storage:
    def __init__(self):
        self.use_local_fallback = False
        self.r2 = None
        try:
            self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            self.access_key = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
            self.secret_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
            self.bucket_name = os.getenv("CLOUDFLARE_BUCKET_NAME", "ai-agents")
            if not all([self.account_id, self.access_key, self.secret_key]):
                raise ValueError("Missing Cloudflare R2 env vars.")
            endpoint_url = f'https://{self.account_id}.r2.cloudflarestorage.com'
            from botocore.config import Config
            session = boto3.session.Session()
            self.r2 = session.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name='auto',
                config=Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 3})
            )
            try:
                self.r2.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                code = e.response.get('Error', {}).get('Code')
                if code == '404' or "NotFound" in str(e):
                    self.r2.create_bucket(Bucket=self.bucket_name)
                else:
                    self.use_local_fallback = True
        except Exception as e:
            logger.critical(f"R2 init failed: {e}")
            self.use_local_fallback = True
            self.r2 = None
        if self.use_local_fallback:
            os.makedirs("local_storage/kb", exist_ok=True)

    def _upload_local_kb(self, file_data: bytes, filename: str) -> Tuple[bool, str]:
        try:
            folder = "kb"
            local_path = f"local_storage/{folder}/{filename}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(file_data)
            file_url = f"file://{os.path.abspath(local_path)}"
            return True, file_url
        except Exception as e:
            return False, str(e)

    def upload_file(self, file_data: bytes, filename: str, is_user_doc: bool = False, schedule_deletion_hours: int = 72) -> Tuple[bool, str]:
        folder = "user_docs" if is_user_doc else "kb"
        key = f"{folder}/{filename}"
        if is_user_doc and (self.use_local_fallback or not self.r2):
            return False, f"Cannot upload user doc '{filename}', R2 unavailable."
        if not self.use_local_fallback and self.r2:
            try:
                self.r2.upload_fileobj(BytesIO(file_data), self.bucket_name, key)
                file_url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{key}"
                self.schedule_deletion(key, schedule_deletion_hours)
                return True, file_url
            except EndpointConnectionError as e:
                if not is_user_doc:
                    return self._upload_local_kb(file_data, filename)
                return False, f"R2 endpoint error: {e}"
            except Exception as e:
                if not is_user_doc:
                    return self._upload_local_kb(file_data, filename)
                return False, f"R2 upload failed: {e}"
        elif not is_user_doc:
            return self._upload_local_kb(file_data, filename)
        return False, "Critical error; could not store file."

    def get_file_content_bytes(self, key: str) -> Optional[bytes]:
        is_user = key.startswith("user_docs/")
        if self.use_local_fallback or not self.r2:
            if is_user:
                return None
            local = f"local_storage/{key}"
            if os.path.exists(local):
                with open(local, 'rb') as f:
                    return f.read()
            return None
        try:
            buf = BytesIO()
            self.r2.download_fileobj(self.bucket_name, key, buf)
            return buf.getvalue()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ['404', 'NoSuchKey']:
                if not is_user:
                    local = f"local_storage/{key}"
                    if os.path.exists(local):
                        with open(local, 'rb') as f:
                            return f.read()
                return None
            return None
        except Exception:
            return None

    def download_file(self, key: str, local_download_path: str) -> bool:
        is_user = key.startswith("user_docs/")
        if not self.use_local_fallback and self.r2:
            try:
                os.makedirs(os.path.dirname(local_download_path), exist_ok=True)
                self.r2.download_file(self.bucket_name, key, local_download_path)
                return True
            except Exception:
                pass
        if not is_user:
            local_src = f"local_storage/{key}"
            if os.path.exists(local_src):
                try:
                    import shutil
                    shutil.copy2(local_src, local_download_path)
                    return True
                except Exception:
                    return False
        return False

    def list_files(self, prefix: str = "") -> List[str]:
        if self.use_local_fallback or not self.r2:
            base = "local_storage/"
            path = os.path.join(base, prefix)
            if os.path.isdir(path):
                return [os.path.join(prefix, f).replace("\\", "/") for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            return []
        try:
            resp = self.r2.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in resp:
                return [o['Key'] for o in resp['Contents']]
            return []
        except Exception:
            return []

    def schedule_deletion(self, key: str, hours: int = 72) -> bool:
        if self.use_local_fallback or not self.r2:
            return False
        try:
            try:
                self.r2.head_object(Bucket=self.bucket_name, Key=key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False
                raise
            import datetime
            expiration_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
            expiration_ts = int(expiration_time.timestamp())
            self.r2.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': key},
                Key=key,
                Metadata={'expiration_time': str(expiration_ts), 'auto_delete': 'true'},
                MetadataDirective='REPLACE'
            )
            return True
        except Exception:
            return False

    def check_and_delete_expired_files(self) -> int:
        if self.use_local_fallback or not self.r2:
            return 0
        import datetime
        deleted = 0
        now = datetime.datetime.now().timestamp()
        try:
            pag = self.r2.get_paginator('list_objects_v2').paginate(Bucket=self.bucket_name)
            for page in pag:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    try:
                        head = self.r2.head_object(Bucket=self.bucket_name, Key=key)
                        md = head.get('Metadata', {})
                        if md.get('auto_delete') == 'true' and 'expiration_time' in md:
                            if now > int(md['expiration_time']):
                                self.r2.delete_object(Bucket=self.bucket_name, Key=key)
                                deleted += 1
                    except Exception:
                        pass
            return deleted
        except Exception:
            return 0
