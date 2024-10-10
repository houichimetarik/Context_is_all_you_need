import os
import re
import sys

def process_file(file_path):
    # Try reading with UTF-8 encoding first
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with the system's default encoding
        try:
            with open(file_path, 'r', encoding=sys.getdefaultencoding(), errors='replace') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return

    # Rest of the function remains the same
    # Check if the imports are already present and in order
    required_imports = [
        "from pycallgraph2 import PyCallGraph",
        "from pycallgraph2.output import GraphvizOutput",
        "test = GraphvizOutput()",
    ]

    # Check if test.output_file is already set
    output_file_match = re.search(r'test\.output_file\s*=\s*"(.+?)"', content)
    if output_file_match:
        output_file = output_file_match.group(1)
    else:
        # Set a default output file name based on the Python file name
        output_file = os.path.splitext(os.path.basename(file_path))[0] + ".json"

    required_imports.extend([
        f'test.output_file = "{output_file}"',
        "test.output_type = 'json'"
    ])

    if all(line in content for line in required_imports):
        print(f"File {file_path} already has the required imports.")
        return

    # Remove existing imports if present
    content = re.sub(r'from pycallgraph2.*\n|test.*\n', '', content, flags=re.MULTILINE)

    # Find the insertion point (after any __future__ imports or at the beginning)
    lines = content.split('\n')
    insert_index = 0
    for i, line in enumerate(lines):
        if line.startswith('from __future__'):
            insert_index = i + 1
        elif not line.strip().startswith('#') and line.strip():
            break

    # Insert the required imports
    for i, import_line in enumerate(required_imports):
        lines.insert(insert_index + i, import_line)

    # Write the modified content back to the file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))
        print(f"Updated {file_path} with the required imports.")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

# The rest of the script (process_directory and main block) remains the same