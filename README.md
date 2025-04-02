# Body Posture Analysis (pre recorded file upload)

## Installation  

1. **Clone the Repository:**  
   ```sh
   git clone https://github.com/Manav-Pathak/body-posture.git  
   cd body-posture  
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**  
   ```sh
   python -m venv venv  
   source venv/bin/activate  # On macOS/Linux  
   venv\Scripts\activate  # On Windows  
   ```

3. **Install Dependencies:**  
   ```sh
   pip install -r requirements.txt  
   ```
   Extra step for moviepy:
   ```sh
   pip uninstall moviepy
   pip install moviepy==1.0.3
   ```
   

## Running the Application  

1. **Start the Flask Server:**  
   ```sh
   python app.py  
   ```
   OR Just run the **app.py** file

2. **Access the Web App:**  
   Open a browser and go to:  
   ```
   http://127.0.0.1:5000  
   ```

## Upload and Process Videos  

1. Navigate to the **Facial Expression** page to upload a video file.  
2. The processed file will be available for download after analysis.  
