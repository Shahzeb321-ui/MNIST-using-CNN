import base64
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


@st.cache_resource
def load_model():
    model = CNN()
    state_dict = torch.load("mnist_cnn.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image):
    gray = image.convert("L").resize((28, 28))
    img_array = np.array(gray, dtype=np.float32) / 255.0

    # Invert the image so dark strokes on a light background match MNIST formatting.
    img_array = 1.0 - img_array

    # Normalize like the training pipeline.
    img_array = (img_array - 0.5) / 0.5

    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor


def image_from_canvas(image_data):
    if image_data is None:
        return None
    if image_data.dtype != np.uint8:
        if image_data.max() <= 1.0:
            image_data = (image_data * 255).astype(np.uint8)
        else:
            image_data = image_data.astype(np.uint8)
    return Image.fromarray(image_data[:, :, :3]).convert("L")


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def predict_image(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted = int(torch.argmax(probs).item())
        confidence = float(probs[predicted].item() * 100)
    return predicted, confidence, probs.numpy()


st.set_page_config(page_title="MNIST CNN Predictor", page_icon="✍️", layout="centered")
st.title("MNIST Digit Classifier")
st.write("Draw a digit or upload an image and the model will guess the number.")

model = load_model()

st.subheader("Draw a digit")

if "drawn_image" not in st.session_state:
    st.session_state["drawn_image"] = None

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#ffffff",
    update_streamlit=True,
    return_image_data=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
)

if canvas_result is not None and canvas_result.image_data is not None:
    st.session_state["drawn_image"] = canvas_result.image_data

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Predict"):
        if st.session_state["drawn_image"] is not None:
            image = image_from_canvas(st.session_state["drawn_image"])
            if image is not None:
                predicted, confidence, probs = predict_image(image)
                st.success(f"Prediction: {predicted}")
                st.write(f"Confidence: {confidence:.2f}%")
                st.bar_chart(probs)
                st.download_button(
                    "Download drawn image",
                    data=image_to_bytes(image),
                    file_name="drawn_digit.png",
                    mime="image/png",
                )
            else:
                st.warning("Unable to convert canvas image.")
        else:
            st.warning("Draw a digit first.")

st.markdown("---")

with st.container():
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded image", width=200)
        predicted, confidence, probs = predict_image(image)
        st.success(f"Prediction: {predicted}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.bar_chart(probs)
        st.download_button(
            "Download uploaded image",
            data=image_to_bytes(image),
            file_name="uploaded_digit.png",
            mime="image/png",
        )
