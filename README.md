# AI-Powered Fashion Recommendation System

This project is an advanced fashion recommendation system that leverages AI to provide personalized outfit suggestions. It analyzes a user's facial features from an uploaded image to determine their gender and skin tone, then recommends complementary clothing colors and generates complete outfits from a fashion dataset.

## 🌟 Features

* **Face Analysis:** Utilizes the `insightface` library to detect faces and determine attributes like gender.
* **Skin Tone Detection:** Analyzes the detected face to identify the dominant skin tone.
* **Color Palette Recommendation:** Suggests a palette of clothing colors that complement the user's detected skin tone.
* **Intelligent Outfit Generation:** Filters a dataset of fashion items based on gender, recommended colors, and user-defined occasion (e.g., Formal, Casual) to create harmonious outfits.
* **LLM-Powered Preference Extraction:** Uses Google's Gemini model to understand natural language input from the user, extracting preferences for colors and occasion.
* **Visual Display:** Displays the recommended outfits with images for each item (Topwear, Bottomwear, Footwear, and Watch).

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for analysis and experimentation.

### Prerequisites

* Python 3.7+
* Pip (Python package installer)
* Jupyter Notebook or Jupyter Lab

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/NikunjSachdeva/Fashion1.git](https://github.com/NikunjSachdeva/Fashion1.git)
    cd Fashion1
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    The notebook installs the necessary Python packages. You can also install them directly from the `requirements.txt` file:
    ```sh
    pip install -r requirements.txt
    ```

## 🏃‍♀️ How to Use

The core logic of this project is contained within the `main.ipynb` Jupyter Notebook.

1.  **Set up your API Key:**
    * Open the `main.ipynb` notebook.
    * In the cell containing the Gemini API setup, replace the placeholder with your own Google API Key:
        ```python
        GOOGLE_API_KEY = "YOUR_API_KEY"
        genai.configure(api_key=GOOGLE_API_KEY)
        ```

2.  **Specify Image Path:**
    * Find the cell where the `image_path` is defined.
    * Update the path to point to an image of a person on your local machine:
        ```python
        image_path = "/path/to/your/image.jpg"
        ```

3.  **Launch Jupyter:**
    Start Jupyter Lab or Notebook from your terminal in the project's root directory:
    ```sh
    # For Jupyter Lab
    jupyter lab

    # Or for the classic Jupyter Notebook
    jupyter notebook
    ```

4.  **Run the Notebook:**
    * Open `main.ipynb` in Jupyter.
    * When prompted, enter your clothing preferences in natural language (e.g., "I want to wear a blue shirt and brown pants for a formal event").
    * Run the cells sequentially to see the entire process, from face detection to the final outfit recommendations.
