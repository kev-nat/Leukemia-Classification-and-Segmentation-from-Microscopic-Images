import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from PIL import Image, ImageEnhance
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2

# Set page configuration
st.set_page_config(
    page_title="Leukemia Analysis Tool",
    page_icon="🔬",
)

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']

def apply_augmentations(image, rotation_angle=0, flip_horizontal=False, flip_vertical=False, 
                       brightness=1.0, contrast=1.0):
    """Apply various augmentations to the input image"""
    # Convert to PIL Image if it's not already
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.uint8(image))
    
    # Apply rotation
    if rotation_angle != 0:
        image = image.rotate(rotation_angle, expand=True)
    
    # Apply flips
    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Apply brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Apply contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    return image

@st.cache_resource
def load_classification_model():
    """Load the classification model"""
    model = tf.keras.models.load_model('InceptionResNetV2_model.keras')
    return model

@st.cache_resource
def load_segmentation_model():
    """Load the segmentation model"""
    return load_model('UNet_fpn_model.h5', compile=False)

def process_classification_image(image_data):
    """Process image for classification"""
    # Resize image
    image = ImageOps.fit(image_data, IMG_SIZE, Image.LANCZOS)
    image = np.asarray(image)
    
    # Preprocess
    img = inceptionresnetv2(image)
    img_reshape = img[np.newaxis,...]
    
    return img_reshape

def segment_image(img, model):
    """Perform image segmentation"""
    try:
        # Convert and preprocess
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction = model.predict(img_array, verbose=0)
        
        if isinstance(prediction, tf.Tensor):
            prediction = prediction.numpy()
        
        # Set a higher threshold for more selective segmentation
        threshold = 0.70
        
        # Convert prediction to binary mask
        mask = prediction[0, :, :, 0]
        mask = (mask > threshold).astype(np.uint8) * 255
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Resize mask to original size
        original_size = (img.size[0], img.size[1])
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        print("\nSegmentation Analysis:")
        print(f"Raw prediction range: {np.min(prediction):.4f} to {np.max(prediction):.4f}")
        print(f"Using threshold: {threshold}")
        print(f"Mask coverage: {np.mean(mask > 0):.2%} of image")
        
        return mask
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        return None

def main():
    st.title("Leukemia Analysis Tool")
    
    # Sidebar navigation and controls
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the application mode", ["Classification", "Segmentation"])
    
    # Image augmentation controls in sidebar
    st.sidebar.title("Image Augmentation")
    with st.sidebar:
        st.markdown("### Transformation Controls")
        rotation_angle = st.slider("Rotation Angle", -180, 180, 0)
        flip_horizontal = st.checkbox("Flip Horizontal")
        flip_vertical = st.checkbox("Flip Vertical")
        
        st.markdown("### Enhancement Controls")
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
        
        if st.button("Reset Augmentations"):
            rotation_angle = 0
            flip_horizontal = False
            flip_vertical = False
            brightness = 1.0
            contrast = 1.0
    
    # Main content area
    if app_mode == "Classification":
        st.header("Image Classification")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display original image
            original_image = Image.open(uploaded_file).convert("RGB")
            
            # Apply augmentations
            augmented_image = apply_augmentations(
                original_image,
                rotation_angle=rotation_angle,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                brightness=brightness,
                contrast=contrast
            )
            
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader("Augmented Image")
                st.image(augmented_image, use_container_width=True)
            
            # Add option to use original or augmented image
            image_choice = st.radio("Select image for classification:", ["Original", "Augmented"])
            selected_image = original_image if image_choice == "Original" else augmented_image
            
            if st.button('Classify'):
                try:
                    with st.spinner('Running classification...'):
                        # Process image
                        processed_image = process_classification_image(selected_image)
                        
                        # Get prediction
                        model = load_classification_model()
                        predictions = model.predict(processed_image)
                        score = tf.nn.softmax(predictions[0])
                        
                        # Display results
                        st.write("Raw predictions:", predictions)
                        st.write("Softmax scores:", score)
                        
                        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                            CLASS_NAMES[np.argmax(score)], 
                            100 * np.max(score)
                        )
                        st.success(result)
                        
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
    
    else:  # Segmentation mode
        st.header("Image Segmentation")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display original image
            original_image = Image.open(uploaded_file).convert("RGB")
            
            # Apply augmentations
            augmented_image = apply_augmentations(
                original_image,
                rotation_angle=rotation_angle,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                brightness=brightness,
                contrast=contrast
            )
            
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            with col2:
                st.subheader("Augmented Image")
                st.image(augmented_image, use_container_width=True)
            
            # Add option to use original or augmented image
            image_choice = st.radio("Select image for segmentation:", ["Original", "Augmented"])
            selected_image = original_image if image_choice == "Original" else augmented_image
            
            if st.button('Segment'):
                try:
                    with st.spinner('Running segmentation...'):
                        model = load_segmentation_model()
                        mask = segment_image(selected_image, model)
                        
                        if mask is not None:
                            # Create RGB mask for better visualization
                            mask_rgb = np.stack([mask] * 3, axis=-1)
                            
                            # Create overlay
                            img_array = np.array(selected_image.resize((mask.shape[1], mask.shape[0])))
                            overlay = cv2.addWeighted(img_array, 0.7, mask_rgb, 0.3, 0)
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(mask, caption="Segmentation Mask")
                            with col2:
                                st.image(overlay, caption="Overlay")
                        else:
                            st.error("Failed to generate segmentation mask")
                            
                except Exception as e:
                    st.error(f"Error during segmentation: {str(e)}")

if __name__ == '__main__':
    main()