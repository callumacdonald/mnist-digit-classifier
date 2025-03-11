import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import psycopg2
import io
import pandas as pd
from model import CNN  # Import trained CNN model structure

# PostgreSQL connection function
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname="mnist_logger",
            user="postgres",
            password="yourpassword",  # Replace with your actual password
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Function to fetch all predictions from the database
def fetch_predictions():
    """Retrieves all logged predictions from the database."""
    conn = connect_db()
    if conn is None:
        return []

    try:
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, predicted_digit, confidence, user_correction, user_drawing FROM predictions ORDER BY timestamp DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []

# Function to save predictions to PostgreSQL
def log_prediction(predicted_digit, confidence, user_drawing):
    """Logs the initial prediction and returns the row ID."""
    conn = connect_db()
    if conn is None:
        st.error("❌ Database connection failed.")
        return None

    try:
        cur = conn.cursor()

        # Convert image to binary
        img_byte_arr = io.BytesIO()
        user_drawing.save(img_byte_arr, format="PNG")
        img_binary = img_byte_arr.getvalue()

        # Insert prediction into DB
        cur.execute(
            "INSERT INTO predictions (predicted_digit, confidence, user_drawing) VALUES (%s, %s, %s) RETURNING id",
            (predicted_digit, confidence, img_binary)
        )
        prediction_id = cur.fetchone()[0]  # Get the ID of the inserted row
        conn.commit()
        cur.close()
        conn.close()

        return prediction_id
    except Exception as e:
        st.error(f"❌ Error logging prediction: {e}")
        return None

# Function to update a user correction in the database
def update_correction(prediction_id, corrected_digit):
    """Updates the prediction row with the user's correction."""
    conn = connect_db()
    if conn is None:
        return False

    try:
        cur = conn.cursor()
        cur.execute("UPDATE predictions SET user_correction = %s WHERE id = %s", (corrected_digit, prediction_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating correction: {e}")
        return False


# Function to convert binary image data back to an image
def binary_to_image(binary_data):
    return Image.open(io.BytesIO(binary_data))

# Load the model
@st.cache_resource
def load_trained_model():
    model = CNN()
    model.load_state_dict(torch.load("cnn_mnist_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Load the model
model = load_trained_model()

# Streamlit UI
st.title("MNIST Digit Recognizer 🧠✏️")
st.write("Draw a digit below and get the model's prediction!")

# Create a drawing canvas
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Initialize session state for storing the prediction ID
if "prediction_id" not in st.session_state:
    st.session_state["prediction_id"] = None

if "correcting" not in st.session_state:
    st.session_state["correcting"] = False

# Process the drawn image and make a prediction with confidence score
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    st.write(f"**Prediction: {predicted.item()} 🎯**")
    st.write(f"**Confidence: {confidence.item() * 100:.2f}%** ✅")

    if st.button("✅ Yes, correct"):
        st.session_state["prediction_id"] = log_prediction(predicted.item(), confidence.item(), img)
        st.success("Prediction logged successfully! ✅")

    if st.button("❌ No, incorrect"):
        st.session_state["prediction_id"] = log_prediction(predicted.item(), confidence.item(), img)
        st.session_state["correcting"] = True

if st.session_state["correcting"]:
    corrected_digit = st.number_input("Enter the correct digit:", min_value=0, max_value=9, step=1)
    if st.button("Submit Correction", key="submit_correction"):
        if st.session_state["prediction_id"] is not None:
            update_correction(st.session_state["prediction_id"], corrected_digit)
            st.success(f"Correction saved: {corrected_digit} ✅")
        st.session_state["correcting"] = False

# 📊 Display Past Predictions in a Table
st.subheader("📊 Past Predictions and Corrections")

# Fetch predictions from DB
predictions = fetch_predictions()

if predictions:
    df = pd.DataFrame(predictions, columns=["ID", "Timestamp", "Predicted Digit", "Confidence", "User Correction", "User Drawing"])
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.2f}%")  # Format confidence as percentage
    
    # Convert binary image data back to images for display
    image_list = []
    for _, row in df.iterrows():
        if row["User Drawing"]:
            img = binary_to_image(row["User Drawing"])
            image_list.append(img)
        else:
            image_list.append(None)

    # Display predictions in a table
    st.dataframe(df.drop(columns=["User Drawing"]))  # Hide image column in the table

    # Show images below the table
    st.subheader("🖼️ User Drawings")
    cols = st.columns(5)  # Adjust columns based on how many images to display

    for i, img in enumerate(image_list[:10]):  # Show only the last 10 images
        if img:
            with cols[i % 5]:  # Distribute images across columns
                st.image(img, caption=f"Prediction: {df.iloc[i]['Predicted Digit']}", width=100)
else:
    st.info("No predictions logged yet.")
