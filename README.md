import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization
from tensorflow.keras.initializers import Zeros, Ones
from tensorflow.keras.utils import custom_object_scope
import cv2
import numpy as np

# ✅ Return dictionary format for compatibility
def fixed_variance_scaling(*args, **kwargs):
    return {"class_name": "GlorotUniform", "config": {}}

def fixed_zeros(*args, **kwargs):
    return Zeros()

def fixed_ones(*args, **kwargs):
    return Ones()

# ✅ Fix SeparableConv2D
def fixed_separable_conv2d(**kwargs):
    allowed_keys = [
        "filters", "kernel_size", "strides", "padding", "data_format", "dilation_rate",
        "activation", "use_bias", "depthwise_initializer", "pointwise_initializer",
        "bias_initializer", "depthwise_regularizer", "pointwise_regularizer",
        "bias_regularizer", "activity_regularizer", "depthwise_constraint", "pointwise_constraint",
        "bias_constraint", "trainable", "name"
    ]
    sanitized_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    
    if "kernel_size" in sanitized_kwargs and isinstance(sanitized_kwargs["kernel_size"], list):
        sanitized_kwargs["kernel_size"] = tuple(sanitized_kwargs["kernel_size"])
    
    if "strides" in sanitized_kwargs and isinstance(sanitized_kwargs["strides"], list):
        sanitized_kwargs["strides"] = tuple(sanitized_kwargs["strides"])
    
    return SeparableConv2D(**sanitized_kwargs)

# ✅ Fix BatchNormalization
class FixedBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs):
        for init_key in ["beta_initializer", "gamma_initializer", "moving_mean_initializer", "moving_variance_initializer"]:
            if init_key in kwargs and isinstance(kwargs[init_key], dict):
                kwargs[init_key].pop("dtype", None)
        super().__init__(*args, **kwargs)

# ✅ Function to Load Model
def load_fixed_model():
    try:
        with custom_object_scope({
            'VarianceScaling': fixed_variance_scaling,  # ✅ Now returns dict format
            'Zeros': fixed_zeros,
            'Ones': fixed_ones,
            'SeparableConv2D': fixed_separable_conv2d,
            'BatchNormalization': FixedBatchNormalization
        }):
            model = load_model("Models/video.h5", compile=False)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ✅ Function to Process Frame
def process_frame(frame, model):
    try:
        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
        resized_frame = resized_frame / 255.0  # Normalize

        prediction = model.predict(resized_frame)
        print(f"Prediction: {prediction}")

    except Exception as e:
        print(f"❌ Error processing frame: {e}")

# ✅ Main Webcam Function
def show_webcam():
    model = load_fixed_model()
    if model is None:
        return  

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break

        process_frame(frame, model)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Run the webcam function
if __name__ == "__main__":
    show_webcam()
