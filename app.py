import streamlit as st
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import tempfile
import os
import random
from PIL import Image, ImageColor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io

# Set page config
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👕",
    layout="wide"
)

# Title and description
st.title("👕 Fashion Recommendation System")
st.markdown("""
Upload a photo of yourself and get personalized fashion recommendations based on your face analysis and skin tone!
""")

# Define skin tone mapping
skin_tones = {
    "#373028": "Deepest Skin",
    "#422811": "Very Deep",
    "#513B2E": "Deep Brown",
    "#6F503C": "Medium Brown",
    "#81654F": "Tan",
    "#9D7A54": "Light Tan",
    "#BEA07E": "Medium Fair",
    "#E5C8A6": "Light Fair",
    "#E7C1B8": "Warm Fair",
    "#F3DAD6": "Very Fair",
    "#FBF2F3": "Pale",
}

# Color recommendations for each skin tone
skin_tone_to_color_mapping = {
    "#373028": ["Navy Blue", "Black", "Charcoal", "Burgundy", "Maroon", "Olive", "Rust", "Gold", "Cream", "Peach"],
    "#422811": ["Navy Blue", "Brown", "Khaki", "Olive", "Maroon", "Mustard", "Teal", "Tan", "Rust", "Burgundy"],
    "#513B2E": ["Cream", "Beige", "Olive", "Burgundy", "Red", "Orange", "Mustard", "Bronze", "Teal", "Peach"],
    "#6F503C": ["Beige", "Brown", "Green", "Khaki", "Cream", "Peach", "Lime Green", "Olive", "Maroon", "Rust", "Mustard"],
    "#81654F": ["Beige", "Off White", "Sea Green", "Cream", "Lavender", "Mauve", "Burgundy", "Yellow", "Lime Green"],
    "#9D7A54": ["Olive", "Khaki", "Yellow", "Sea Green", "Turquoise Blue", "Coral", "White", "Gold", "Peach"],
    "#BEA07E": ["Coral", "Sea Green", "Turquoise Blue", "Pink", "Lavender", "Rose", "White", "Peach", "Teal", "Fluorescent Green"],
    "#E5C8A6": ["Turquoise Blue", "Peach", "Teal", "Pink", "Red", "Rose", "Off White", "White", "Cream", "Gold", "Yellow"],
    "#E7C1B8": ["Pink", "Rose", "Peach", "White", "Off White", "Beige", "Lavender", "Teal", "Fluorescent Green"],
    "#F3DAD6": ["White", "Cream", "Peach", "Pink", "Rose", "Lavender", "Mustard", "Lime Green", "Light Blue", "Fluorescent Green"],
    "#FBF2F3": ["Peach", "Lavender", "Pink", "White", "Off White", "Rose", "Light Blue", "Sea Green", "Silver", "Cream", "Tan"]
}

# Load datasets
@st.cache_data
def load_data():
    # Load fashion dataset
    df = pd.read_csv("styles.csv", on_bad_lines="skip")
    df = df[["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "usage", "productDisplayName"]]
    df = df.dropna()
    
    # Load image links
    images_links = pd.read_csv("images.csv", on_bad_lines="skip")
    
    return df, images_links

# Initialize face analysis
@st.cache_resource
def load_face_analyzer():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=1)
    return app

# Function to detect skin tone from image
def detect_skin_tone(image, bbox=None):
    """
    Detects the closest skin tone from the provided face image.
    
    Parameters:
    - image: The input image in BGR format.
    - bbox: Bounding box coordinates (x1, y1, x2, y2) for cropping the face.
    
    Returns:
    - Closest skin tone hex color.
    - Skin tone name.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]

    # For faster processing
    resized_image = cv2.resize(image, (200, 200))

    # Compute average color
    average_color = resized_image.mean(axis=0).mean(axis=0)
    avg_color_hex = "#{:02x}{:02x}{:02x}".format(
        int(average_color[0]), int(average_color[1]), int(average_color[2])
    )

    # Convert hex colors to RGB for comparison
    avg_color_rgb = np.array(ImageColor.getrgb(avg_color_hex))

    # Finding the closest matching skin tone using Euclidean distance
    closest_tone_hex = min(
        skin_tones.keys(),
        key=lambda hex_code: np.linalg.norm(avg_color_rgb - np.array(ImageColor.getrgb(hex_code)))
    )

    return closest_tone_hex, skin_tones[closest_tone_hex]

# Function to display skin tone color block
def display_skin_tone_sample(skin_tone_hex):
    # Create a color block
    color_block = np.ones((100, 100, 3))
    skin_rgb = mcolors.hex2color(skin_tone_hex)
    color_block[:, :] = skin_rgb

    # Create plot
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(color_block)
    ax.axis("off")
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

# Function to display combination images
def display_combination_images(top_id, bottom_id, foot_id, watch_id, images_links):
    # Fetch image links
    top_img = images_links[images_links['filename'] == f"{top_id}.jpg"]['link'].values
    bottom_img = images_links[images_links['filename'] == f"{bottom_id}.jpg"]['link'].values
    foot_img = images_links[images_links['filename'] == f"{foot_id}.jpg"]['link'].values
    watch_img = images_links[images_links['filename'] == f"{watch_id}.jpg"]['link'].values
    
    # Handle missing images
    top_img = top_img[0] if len(top_img) > 0 else "https://via.placeholder.com/200?text=No+Image"
    bottom_img = bottom_img[0] if len(bottom_img) > 0 else "https://via.placeholder.com/200?text=No+Image"
    foot_img = foot_img[0] if len(foot_img) > 0 else "https://via.placeholder.com/200?text=No+Image"
    watch_img = watch_img[0] if len(watch_img) > 0 else "https://via.placeholder.com/200?text=No+Image"
    
    # Create columns for display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Topwear")
        st.image(top_img, width=200)
        
    with col2:
        st.subheader("Bottomwear")
        st.image(bottom_img, width=200)
        
    with col3:
        st.subheader("Footwear")
        st.image(foot_img, width=200)
        
    with col4:
        st.subheader("Watch")
        st.image(watch_img, width=200)

# Function to generate outfit combinations based on gender, usage, and recommended colors
def generate_outfit_combinations(df, gender, skin_tone_hex, usage="Casual", n_combinations=5):
    # Get recommended colors for the skin tone
    recommended_colors = skin_tone_to_color_mapping.get(skin_tone_hex, ["Black", "White", "Blue", "Grey"])
    
    # Filter data by gender and usage
    filtered_data = df[(df["gender"] == gender) & (df["usage"] == usage)]
    
    # Find items for each category with preference for recommended colors
    # For topwear, prioritize recommended colors
    topwear_recommended = filtered_data[
        (filtered_data['subCategory'].str.lower() == 'topwear') & 
        (filtered_data['baseColour'].isin(recommended_colors))
    ]
    
    # If not enough items with recommended colors, add other topwear items
    if len(topwear_recommended) < 5:
        topwear_other = filtered_data[
            (filtered_data['subCategory'].str.lower() == 'topwear') & 
            (~filtered_data['baseColour'].isin(recommended_colors))
        ]
        topwear = pd.concat([topwear_recommended, topwear_other]).drop_duplicates()
    else:
        topwear = topwear_recommended
    
    # For bottomwear, prefer neutral colors
    neutral_colors = ["Black", "White", "Grey", "Navy Blue", "Beige", "Cream", "Charcoal"]
    bottomwear_neutral = filtered_data[
        (filtered_data['subCategory'].str.lower() == 'bottomwear') & 
        (filtered_data['baseColour'].isin(neutral_colors))
    ]
    
    # If not enough neutral bottomwear, add other items
    if len(bottomwear_neutral) < 5:
        bottomwear_other = filtered_data[
            (filtered_data['subCategory'].str.lower() == 'bottomwear') & 
            (~filtered_data['baseColour'].isin(neutral_colors))
        ]
        bottomwear = pd.concat([bottomwear_neutral, bottomwear_other]).drop_duplicates()
    else:
        bottomwear = bottomwear_neutral
    
    # For footwear and watches
    footwear = filtered_data[
        (filtered_data['masterCategory'].str.lower() == 'footwear') & 
        (filtered_data['subCategory'].str.lower() == 'shoes')
    ]
    
    watches = filtered_data[
        (filtered_data['masterCategory'].str.lower() == 'accessories') & 
        (filtered_data['articleType'].str.lower() == 'watches')
    ]
    
    # Debug info
    st.sidebar.write(f"Topwear items: {len(topwear)}")
    st.sidebar.write(f"Bottomwear items: {len(bottomwear)}")
    st.sidebar.write(f"Footwear items: {len(footwear)}")
    st.sidebar.write(f"Watches: {len(watches)}")
    st.sidebar.write(f"Recommended colors: {recommended_colors}")
    
    outfit_combinations = []
    
    # Generate combinations
    if not topwear.empty and not bottomwear.empty and not footwear.empty:
        sample_topwear = topwear.sample(min(5, len(topwear)))
        sample_bottomwear = bottomwear.sample(min(5, len(bottomwear)))
        sample_footwear = footwear.sample(min(5, len(footwear)))
        sample_watch = watches.sample(min(3, len(watches))) if not watches.empty else None
        
        for top in sample_topwear.itertuples():
            for bottom in sample_bottomwear.itertuples():
                for foot in sample_footwear.itertuples():
                    for watch in (sample_watch.itertuples() if sample_watch is not None else [None]):
                        watch_id = watch.id if watch else "None"
                        watch_color = watch.baseColour if watch else "None"
                        
                        outfit_combinations.append({
                            "Topwear": top.id,
                            "Bottomwear": bottom.id,
                            "Footwear": foot.id,
                            "Watch": watch_id,
                            "Topwear Color": top.baseColour,
                            "Bottomwear Color": bottom.baseColour,
                            "Footwear Color": foot.baseColour,
                            "Watch Color": watch_color
                        })
    
    # Return random subset of combinations
    if outfit_combinations:
        return random.sample(outfit_combinations, min(n_combinations, len(outfit_combinations)))
    return []

# Load data
df, images_links = load_data()

# UI for photo upload
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        image_path = tmp_file.name

    # Display the uploaded image
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(uploaded_file, caption="Your Photo", use_column_width=True)

    # Process the image
    try:
        # Load face analyzer
        app = load_face_analyzer()
        
        # Read and process image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = app.get(img)
        
        if len(faces) > 0:
            # Get the first face
            face = faces[0]
            
            # Get face bounding box
            bbox = (
                int(face.bbox[0]),
                int(face.bbox[1]),
                int(face.bbox[2]),
                int(face.bbox[3])
            )
            
            # Detect skin tone
            skin_tone_hex, skin_tone_name = detect_skin_tone(img, bbox)
            
            # Display face analysis results
            with col2:
                st.subheader("Your Analysis Results")
                gender = "Men" if face.gender == 1 else "Women"
                st.write(f"Detected Gender: {gender}")
                st.write(f"Estimated Age: {int(face.age)}")
                
                # Display skin tone
                st.write(f"Detected Skin Tone: {skin_tone_name}")
                
                # Display color block for skin tone
                skin_tone_img = display_skin_tone_sample(skin_tone_hex)
                st.image(skin_tone_img, width=100)
                
                # Display recommended colors
                recommended_colors = skin_tone_to_color_mapping.get(skin_tone_hex, ["Default"])
                st.write(f"Recommended Colors: {', '.join(recommended_colors[:5])}")
                
                # Add usage selection
                usage = st.selectbox("Select Usage", ["Casual", "Formal", "Sports", "Ethnic", "Travel"])
                
                # Generate outfit combinations based on skin tone
                outfit_combinations = generate_outfit_combinations(df, gender, skin_tone_hex, usage)
                
                if outfit_combinations:
                    st.success(f"✅ Found {len(outfit_combinations)} outfit combinations for you based on your skin tone!")
                else:
                    st.warning("No outfit combinations found. Try a different usage type.")
        else:
            st.error("No face detected in the image. Please upload a clear photo with a visible face.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(image_path)

    # Display outfit combinations if available
    if 'outfit_combinations' in locals() and outfit_combinations:
        st.markdown("---")
        st.subheader("Recommended Outfits")
        
        # Create tabs for each outfit
        tabs = st.tabs([f"Outfit {i+1}" for i in range(len(outfit_combinations))])
        
        for i, (tab, combo) in enumerate(zip(tabs, outfit_combinations)):
            with tab:
                st.markdown(f"""
                👕 **Topwear**: {combo['Topwear Color']} (ID: {combo['Topwear']})  
                👖 **Bottomwear**: {combo['Bottomwear Color']} (ID: {combo['Bottomwear']})  
                👞 **Footwear**: {combo['Footwear Color']} (ID: {combo['Footwear']})  
                ⌚ **Watch**: {combo['Watch Color']} (ID: {combo['Watch']})
                """)
                
                # Display outfit images
                display_combination_images(
                    combo['Topwear'], 
                    combo['Bottomwear'], 
                    combo['Footwear'], 
                    combo['Watch'],
                    images_links
                )
else:
    # Display info when no image is uploaded
    st.info("Upload a photo to get personalized fashion recommendations based on your face and skin tone!")

# Add some styling
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True) 