import os

import requests
from tqdm import tqdm


def get_folder(folder_path, verbose=True):
    """Creates folder, if it doesn't exist, and returns folder path.
    Args:
        folder_path (str): Folder path, either existing or to be created.
    Returns:
        str: folder path.
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"-created directory: {folder_path}")
    return folder_path


def download_from_url(file_url, file_path):
    """
    Downloads files from given url. If file has been partially downloaded it will
    continue the download. Requests are sent in chunks of 1024 bytes. Taken from
    https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2.
    Args:
        file_url (str): string with url to download from.
        file_path (str): file path to save downloaded file to.
    Returns:
        int: file size.
    """
    file_size = int(requests.head(file_url).headers["Content-Length"])
    if os.path.exists(file_path):
        first_byte = os.path.getsize(file_path)
    else:
        first_byte = 0

    retries = 0
    while first_byte < file_size:
        if retries > 0:
            print(f"Current number of retries: {retries}")
        retries += 1
        header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
        pbar = tqdm(
            total=file_size,
            initial=first_byte,
            unit="B",
            unit_scale=True,
            desc=file_url.split("/")[-1],
        )
        req = requests.get(file_url, headers=header, stream=True)
        with (open(file_path, "ab")) as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
        pbar.close()
        first_byte = os.path.getsize(file_path)
    print(f"-saved: {file_path}")
    return file_size


def simple_download_from_url(file_url, file_path):
    """
    Alternative function to download files from url, without support for partial
    downloads and chunking. Useful for some websites that block such functionalities
    or file formats that do not allow partial snapshots.
    Args:
        file_url (str): string with url to download from.
        file_path (str): file path to save downloaded file to.
    """
    print(file_url)
    r = requests.get(file_url, allow_redirects=True)
    open(file_path, "wb").write(r.content)