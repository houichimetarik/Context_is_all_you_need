import os
import chardet

def read_file_with_auto_detect(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']
    try:
        return raw_data.decode(encoding)
    except UnicodeDecodeError:
        return raw_data.decode('utf-8', errors='ignore')

def move_future_imports_to_top(file_path):
    try:
        content = read_file_with_auto_detect(file_path)
        lines = content.splitlines()

        # Collect all future imports
        future_imports = [line for line in lines if line.strip().startswith('from __future__ import')]
        
        # Remove all future imports from their original positions
        lines = [line for line in lines if not line.strip().startswith('from __future__ import')]

        # Find the insertion point (after any shebang or encoding declarations)
        insert_index = 0
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#'):
                insert_index = i
                break

        # Insert all future imports at the top
        for import_line in reversed(future_imports):
            lines.insert(insert_index, import_line)

        new_content = '\n'.join(lines)

        if future_imports:
            print(f"Modified {file_path}: Moved {len(future_imports)} future import(s) to the top.")
        else:
            print(f"No changes made to {file_path}: No future imports found.")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                move_future_imports_to_top(file_path)

if __name__ == "__main__":
    directory_path = input("Enter the directory path to process: ")
    process_directory(directory_path)