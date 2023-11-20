from pydub import AudioSegment
import os

def convert_mp3_to_wav(input_path, output_root_dir):
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(input_path)
        
        # Construct the output path
        relative_path = os.path.relpath(input_path, input_root_dir)
        output_path = os.path.join(output_root_dir, os.path.splitext(relative_path)[0] + ".wav")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to WAV
        audio.export(output_path, format="wav")
        print(f"Conversion complete: {output_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    # Specify the input and output root directories
    input_root_dir = "D:\\MusicGenreClassification\\ismir04_genre\\audio\\development"
    output_root_dir = "D:\\MusicGenreClassification\\ismir04_genre\\audio\\development"  # You can use the same directory if you want 

    # Loop through all directories and subdirectories
    for subdir, dirs, files in os.walk(input_root_dir):
        for file in files:
            if file.endswith(".mp3"):
                input_path = os.path.join(subdir, file)
                convert_mp3_to_wav(input_path, output_root_dir)
