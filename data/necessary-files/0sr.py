from __future__ import annotations
import os

def replace_spaces_with_underscores(filename):
    # Replace spaces with underscores
    new_filename = filename.replace(' ', '_')
    
    # Rename the file
    try:
        os.rename(filename, new_filename)
        print(f"Renamed '{filename}' to '{new_filename}'")
    except OSError as e:
        print(f"Error renaming file '{filename}': {e}")

def rename_files_in_directory():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Iterate over all files in the directory
    for filename in os.listdir(current_dir):
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(current_dir, filename)):
            replace_spaces_with_underscores(filename)

# Run the script
rename_files_in_directory()