#!/bin/bash

# Counter for successful and failed runs
success_count=0
fail_count=0

# Use the system's default Python interpreter
PYTHON_PATH="C:\Users\pi\AppData\Local\Programs\Python\Python311\python.exe"

# Name of the error log file
ERROR_LOG="error_log.txt"

# Clear the error log file if it exists, or create a new one
> "$ERROR_LOG"

# List of files to exclude
EXCLUDE_FILES=("0fix.py" "0naming.py" "0sr.py" )

# Loop through all .py files in the current directory
for file in *.py; do
    # Check if the file should be excluded
    if [[ " ${EXCLUDE_FILES[@]} " =~ " ${file} " ]]; then
        echo "Skipping $file as it's in the exclude list."
        continue
    fi

    if [ -f "$file" ]; then
        echo "Running $file..."
        
        # Run the Python file and capture both stdout and stderr
        output=$("$PYTHON_PATH" "$file" 2>&1)
        exit_status=$?
        
        # Check the exit status of the Python command
        if [ $exit_status -eq 0 ]; then
            echo "Successfully ran $file"
            ((success_count++))
            echo "$output"
        else
            echo "Error running $file"
            ((fail_count++))
            
            # Display the error output
            echo "Error description:"
            echo "$output"
            
            # Append the error to the log file
            {
                echo "-----------"
                echo "$file"
                echo "Error description:"
                echo "$output"
                echo "-----------"
                echo ""
            } >> "$ERROR_LOG"
        fi
        echo "------------------------"
    fi
done

echo "Execution complete."
echo "Successfully ran $success_count files."
echo "Encountered errors in $fail_count files."
echo "Errors have been logged to $ERROR_LOG"