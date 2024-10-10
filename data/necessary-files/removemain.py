from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
test = GraphvizOutput()
test.output_file = "removemain.json"
test.output_type = 'json'

import os
import ast
import chardet

def has_main(file_path):
    try:
        with open(file_path, 'rb') as file:
            raw_content = file.read()
            encoding = chardet.detect(raw_content)['encoding']
        
        with open(file_path, 'r', encoding=encoding) as file:
            tree = ast.parse(file.read())
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}. Skipping.")
        return False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return True
        elif isinstance(node, ast.If):
            if isinstance(node.test, ast.Compare):
                if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                    if isinstance(node.test.comparators[0], ast.Str) and node.test.comparators[0].s == '__main__':
                        return True
    return False

def check_and_delete(folder_path):
    removed = 0
    kept = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if not has_main(file_path):
                    try:
                        os.remove(file_path)
                        print(f"{file_path} deleted as it doesn't have a main function or __main__ block.")
                        removed += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")
                else:
                    print(f"{file_path} has a main function or __main__ block. Keeping the file.")
                    kept += 1
    return removed, kept

# List of folders to check
folders = [
    'abstractfactory', 'adapter', 'bridge', 'builder', 'chainofresponsibility',
    'command', 'composite', 'decorator', 'facade', 'factory', 'flyweight',
    'interpreter', 'iterator', 'mediator', 'memento', 'observer', 'prototype',
    'proxy', 'singleton', 'state', 'strategy', 'templatemethod', 'visitor'
]

# Initialize overall statistics
total_removed = 0
total_kept = 0

# Check each folder
for folder in folders:
    print(f"\nChecking folder: {folder}")
    removed, kept = check_and_delete(folder)
    total_removed += removed
    total_kept += kept
    print(f"Folder statistics: {removed} files removed, {kept} files kept")

# Print overall statistics
print(f"\nOverall statistics:")
print(f"Total files removed: {total_removed}")
print(f"Total files kept: {total_kept}")
print(f"Total files processed: {total_removed + total_kept}")