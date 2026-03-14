import os

def append_letter_to_images(folder_path, letter_to_append):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    # Change the current working directory
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print("Error: The system cannot find the path specified.")
        return

    for filename in files:
        # Split the name and the extension
        name, ext = os.path.splitext(filename)
        
        # Check if the file is an image
        if ext.lower() in valid_extensions:
            # Create the new name (e.g., photo.jpg -> photo_A.jpg)
            new_name = f"{name}{letter_to_append}{ext}"
            
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")


my_folder = "data/Shoko-samples" # Use forward slashes
my_letter = "_S"  # The letter or string you want to add

append_letter_to_images(my_folder, my_letter)