"""File uploader component"""

import streamlit as st
from PIL import Image


class FileUploader:
    """File uploader component for image enhancement."""

    def __init__(self):
        """Initialize file uploader."""
        self.supported_types = ["png", "jpg", "jpeg"]

    def upload(self):
        """Create file uploader widget.

        Returns:
            Uploaded file object or None
        """
        return st.file_uploader(
            "Upload Image",
            type=self.supported_types,
            help="Supported formats: PNG, JPG",
        )

    def validate_file(self, file):
        """Validate uploaded file.

        Args:
            file: Uploaded file object

        Returns:
            bool: True if valid
        """
        if not file:
            return False

        try:
            image = Image.open(file)
            return (
                image.format.lower() in self.supported_types
                and image.size[0] > 0
                and image.size[1] > 0
            )
        except Exception:
            return False

    def display_preview(self, file):
        """Display image preview.

        Args:
            file: Uploaded file object
        """
        if not self.validate_file(file):
            return

        try:
            image = Image.open(file)
            st.image(image, caption="Preview", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")

    def get_file_info(self, file):
        """Get file information.

        Args:
            file: Uploaded file object

        Returns:
            dict: File information
        """
        if not self.validate_file(file):
            return {}

        try:
            image = Image.open(file)
            return {
                "name": file.name,
                "size": file.size,
                "width": image.size[0],
                "height": image.size[1],
                "format": image.format,
            }
        except Exception:
            return {}
