from __future__ import annotations
import os
import re

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
test = GraphvizOutput()
test.output_file = "update_output_file.json"
test.output_type = 'json'

def update_output_file(directory):
    changed_files = 0
    pattern = re.compile(r'test\.output_file\s*=\s*"(.+?)"')

    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                print(f"Unable to read {filename} with UTF-8 encoding. Skipping.")
                continue

            match = pattern.search(content)
            if match:
                current_output = match.group(1)
                expected_output = f"{os.path.splitext(filename)[0]}.json"

                if current_output != expected_output:
                    new_content = pattern.sub(f'test.output_file = "{expected_output}"', content)
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    changed_files += 1
                    print(f"Updated {filename}")

    print(f"Total files changed: {changed_files}")

# Use the current directory, or specify a different one
directory = "."

if __name__ == "__main__":
    with PyCallGraph(output=test):
        update_output_file(directory)