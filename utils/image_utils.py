from langchain.schema import Document
from PIL import Image
import tempfile
import os
import easyocr

ocr_reader = easyocr.Reader(['en'],gpu=False)  # Initialize EasyOCR once to avoid reloading models repeatedly.

def extract_image_text(uploaded_image):
    """
    Extract text from an uploaded image file using EasyOCR or Tesseract OCR.
    Args:
        uploaded_image: File-like object containing the image.
    Returns:
        List[Document]: Extracted text in LangChain document format.
    """
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        temp_image.write(uploaded_image.read())
        temp_image_path = temp_image.name

    try:
        # Use EasyOCR to extract text
        result = ocr_reader.readtext(temp_image_path)
        text = " ".join([item[1] for item in result])  # Extract and combine the detected text

        # Include the source file name in the metadata
        return [
            Document(
                page_content=text,
                metadata={"source": "image", "file_name": uploaded_image.name}
            )
        ]
    except Exception as e:
        raise ValueError(f"Failed to process image with EasyOCR: {e}")
    finally:
        # Ensure the temporary file is removed after processing
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
