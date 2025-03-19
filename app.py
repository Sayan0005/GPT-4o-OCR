import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import time  # Add time module for measuring processing time

# Page configuration
st.set_page_config(
    page_title="Invoice OCR",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    h1 {
        color: #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# Function to encode the image to base64
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

# Function to extract text from invoice using OpenAI API
def extract_invoice_data(api_key, base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",  # Updated to use the current GPT-4o model which supports vision
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is an invoice image. Extract and organize all the information in a structured format. Include invoice number, date, vendor details, customer details, line items, subtotal, taxes, total amount, payment terms, and any other relevant information. Format your response in a clear, well-organized markdown format."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }

    # Record start time for API call
    api_call_start = time.time()

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Calculate API call duration
    api_call_duration = time.time() - api_call_start

    if response.status_code == 200:
        response_data = response.json()

        # Print detailed response info to terminal
        print("\n" + "="*50)
        print("API RESPONSE DETAILS:")
        print("="*50)

        # Print time information
        print(f"EXTRACTION TIME:")
        print(f"  - API call duration: {api_call_duration:.2f} seconds")

        # Extract and print token usage information
        if 'usage' in response_data:
            usage = response_data['usage']
            print(f"\nTOKEN USAGE:")
            print(f"  - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  - Total tokens: {usage.get('total_tokens', 'N/A')}")

            # Calculate token processing speed if available
            total_tokens = usage.get('total_tokens')
            if total_tokens and api_call_duration > 0:
                tokens_per_second = total_tokens / api_call_duration
                print(f"  - Processing speed: {tokens_per_second:.2f} tokens/second")

        # Print model used
        print(f"\nMODEL: {response_data.get('model', 'N/A')}")

        # Print response content
        print("\nRESPONSE CONTENT:")
        print("-"*50)
        content = response_data['choices'][0]['message']['content']
        print(content)
        print("-"*50 + "\n")

        return content
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        print("\n" + "="*50)
        print("API ERROR:")
        print("="*50)
        print(error_message)
        print("="*50 + "\n")

        st.error(error_message)
        return None

# Main app layout
st.title("📑 Invoice OCR Extractor")
st.markdown("### Upload an invoice image to extract structured information")

# API key handling
api_key = ""

# File uploader
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an invoice image...", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image with a decent size
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Invoice", use_column_width=True)

            if st.button("Extract Invoice Data 🔍", type="primary"):
                with st.spinner("Processing invoice image..."):
                    print("\n" + "="*50)
                    print("PROCESSING NEW INVOICE IMAGE")
                    print("="*50)

                    # Record total processing start time
                    total_start_time = time.time()

                    # Record encoding start time
                    encoding_start = time.time()

                    # Encode the image
                    base64_image = encode_image(uploaded_file)

                    # Calculate encoding time
                    encoding_time = time.time() - encoding_start
                    print(f"Image encoding time: {encoding_time:.2f} seconds")

                    # Extract data
                    extracted_text = extract_invoice_data(api_key, base64_image)

                    if extracted_text:
                        # Calculate total processing time
                        total_time = time.time() - total_start_time
                        print(f"\nTOTAL PROCESSING TIME: {total_time:.2f} seconds")

                        # Save to session state
                        st.session_state['ocr_result'] = extracted_text
        except Exception as e:
            error_message = f"Error opening the file: {str(e)}. If it's a PDF, please convert it to an image first."
            print("\n" + "="*50)
            print("FILE ERROR:")
            print("="*50)
            print(error_message)
            print("="*50 + "\n")

            st.error(error_message)

with col2:
    # Results area
    if 'ocr_result' in st.session_state:
        st.markdown("### 📝 Extracted Invoice Data")
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(st.session_state['ocr_result'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Add option to download as text
        result_str = st.session_state['ocr_result']
        st.download_button(
            label="Download Results as Text",
            data=result_str,
            file_name="invoice_extraction.txt",
            mime="text/plain"
        )
    else:
        st.info("Upload an invoice and click 'Extract Invoice Data' to see the extracted information here.")

# Footer
st.markdown("---")
st.markdown("*Invoice OCR powered by OpenAI's GPT-4o Vision API*")
