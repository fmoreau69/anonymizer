import mimetypes

mimetypes.init()


def is_image(path):
    """
    Check if a file is an image based on its MIME type.

    Args:
        path: File path to check

    Returns:
        bool: True if the file is an image, False otherwise
    """
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type and mime_type.startswith("image")
