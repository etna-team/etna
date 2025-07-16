import hashlib
import inspect
import json
import os
import pathlib
import warnings
import zipfile
from copy import deepcopy
from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from urllib import request

from hydra_slayer import get_factory


# Known model hashes for integrity verification
# To add a hash for a model URL, download the file and compute its MD5 hash
KNOWN_MODEL_HASHES: Dict[str, str] = {
    # Add known model URL -> hash mappings here
    # Example: "http://example.com/model.ckpt": "abcd1234...",
}


def verify_file_hash(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """
    Verify file integrity using MD5 hash.

    Parameters
    ----------
    file_path:
        Path to the file to verify
    expected_hash:
        Expected MD5 hash. If None, verification is skipped.

    Returns
    -------
    :
        True if hash matches or no expected hash provided, False otherwise
    """
    if expected_hash is None:
        return True

    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception:
        return False


def download_with_integrity_check(
    url: str, 
    destination_path: str, 
    expected_hash: Optional[str] = None,
    force_redownload: bool = False
) -> None:
    """
    Download a file with integrity verification.
    
    Parameters
    ----------
    url:
        URL to download from
    destination_path:
        Local path to save the file
    expected_hash:
        Expected MD5 hash for verification. If None, no verification is performed.
    force_redownload:
        If True, download even if file exists and passes verification
        
    Raises
    ------
    RuntimeError:
        If download fails integrity check
    """
    # Check if file exists and verify integrity
    if os.path.exists(destination_path) and not force_redownload:
        if verify_file_hash(destination_path, expected_hash):
            return  # File exists and is valid
        else:
            # File exists but hash doesn't match, re-download
            if expected_hash is not None:
                warnings.warn(
                    f"Local file hash does not match expected hash. "
                    f"This may indicate a corrupted download. Re-downloading from {url}"
                )
            os.remove(destination_path)

    # Download the file
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    request.urlretrieve(url=url, filename=destination_path)

    # Verify the downloaded file
    if not verify_file_hash(destination_path, expected_hash):
        if expected_hash is not None:
            os.remove(destination_path)
            raise RuntimeError(
                f"Downloaded file from {url} failed integrity check. "
                f"This may indicate a network issue or corrupted download."
            )


def get_known_hash(url: str) -> Optional[str]:
    """
    Get known hash for a URL from the registry.
    
    Parameters
    ----------
    url:
        URL to look up
        
    Returns
    -------
    :
        Known hash for the URL, or None if not found
    """
    return KNOWN_MODEL_HASHES.get(url)


def load(path: pathlib.Path, **kwargs: Any) -> Any:
    """Load saved object by path.

    Warning
    -------
    This method uses :py:mod:`dill` module which is not secure.
    It is possible to construct malicious data which will execute arbitrary code during loading.
    Never load data that could have come from an untrusted source, or that could have been tampered with.

    Parameters
    ----------
    path:
        Path to load object from.
    kwargs:
        Parameters for loading specific for the loaded object.

    Returns
    -------
    :
        Loaded object.
    """
    with zipfile.ZipFile(path, "r") as archive:
        # read object class
        with archive.open("metadata.json", "r") as input_file:
            metadata_bytes = input_file.read()
        metadata_str = metadata_bytes.decode("utf-8")
        metadata = json.loads(metadata_str)
        object_class_name = metadata["class"]

        # create object for that class
        object_class = get_factory(object_class_name)
        loaded_object = object_class.load(path=path, **kwargs)

    return loaded_object


def init_collector(init: Callable) -> Callable:
    """
    Make decorator for collecting init parameters.
    N.B. if init method has positional only parameters, they will be ignored.
    """

    @wraps(init)
    def wrapper(*args, **kwargs):
        self, *args = args
        init_args = inspect.signature(self.__init__).parameters

        deepcopy_args = deepcopy(args)
        deepcopy_kwargs = deepcopy(kwargs)

        self._init_params = {}
        args_dict = dict(
            zip([arg for arg, param in init_args.items() if param.kind == param.POSITIONAL_OR_KEYWORD], deepcopy_args)
        )
        self._init_params.update(args_dict)
        self._init_params.update(deepcopy_kwargs)

        return init(self, *args, **kwargs)

    return wrapper


def create_type_with_init_collector(type_: type) -> type:
    """Create type with init decorated with init_collector."""
    previous_frame = inspect.stack()[1]
    module = inspect.getmodule(previous_frame[0])
    if module is None:
        return type_
    new_type = type(type_.__name__, (type_,), {"__module__": module.__name__})
    if hasattr(type_, "__init__"):
        new_type.__init__ = init_collector(new_type.__init__)  # type: ignore
    return new_type
