import oci
from config import Config


def get_object_storage_client():
    config = oci.config.from_file(Config.OCI_CONFIG_FILE)
    return oci.object_storage.ObjectStorageClient(config)


def upload_text(object_name: str, content: str) -> str:
    try:
        client = get_object_storage_client()
        client.put_object(
            namespace_name=Config.OCI_NAMESPACE,
            bucket_name=Config.OCI_BUCKET_NAME,
            object_name=object_name,
            put_object_body=content.encode("utf-8"),
        )
        return object_name
    except Exception as e:
        print(f"[OCI Storage Upload Error] {object_name}: {str(e)}")
        return None
