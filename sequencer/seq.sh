#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
WHITE='\033[1;37m'  
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' 


display_banner() {
    echo -e "${WHITE}"
    echo "=============================================================="
    echo "**************************************************************"
    echo "*                                                            *"
    echo "*                  DP SEQUENCE GENERATOR                     *"
    echo "*                                                            *"
    echo "**************************************************************"
    echo "=============================================================="
    echo -e "${NC}"
}

display_banner

# Python interpreter path
PYTHON_PATH="C:/Users/pi/AppData/Local/Programs/Python/Python311/python.exe"

# Designpatterns directory
DESIGN_PATTERNS_DIR="./designpatterns"

# Errors log file
ERROR_LOG="sequence_generation_errors.log"

# Output file
SEQUENCES_FILE="sequences.csv"

# Counting processed files
PROCESSED_FILES=0

# Max allowed sequence length
MAX_SEQUENCE_LENGTH=148

# Errors logger
log_error() {
    echo "$(date): $1" >> "$ERROR_LOG"
}

# Generating sequence for a single file at one time, the file .py should match a file .json
generate_sequence() {
    py_file="$1"
    base_name=$(basename "${py_file%.py}")
    json_file="${py_file%.py}.json"
    
    relative_path="${py_file#$DESIGN_PATTERNS_DIR/}"
    
    label=$(basename "$(dirname "$py_file")")

    # Check if corresponding .json file exists
    if [ ! -f "$json_file" ]; then
        echo -e "${WHITE}Warning: No corresponding JSON file found for $relative_path. Skipping.${NC}"
        log_error "No JSON file for $relative_path"
        return
    fi

    echo -e "${WHITE}Processing ${WHITE}$relative_path${WHITE}...${NC}"

    # Generation
    sequence=$("$PYTHON_PATH" -c "
import importlib.util
import sys
import os
import io
from contextlib import redirect_stdout

sys.path.append(os.path.dirname('${py_file}'))

from seqgen import Sequencer

spec = importlib.util.spec_from_file_location('${base_name}', '${py_file}')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sequencer = Sequencer('${json_file}')
sequencer.initiate()
print(sequencer.generate_sequence(module))
" 2>> "$ERROR_LOG" || log_error "Python error processing $relative_path")

    if [ $? -eq 0 ] && [ ! -z "$sequence" ]; then
        # Check if the sequence length is within the allowed limit
        if [ ${#sequence} -le $MAX_SEQUENCE_LENGTH ]; then
            # Escape any commas in the sequence
            escaped_sequence=$(echo "$sequence" | sed 's/,/\\,/g')
            echo "$relative_path,$escaped_sequence,$label" >> "$SEQUENCES_FILE"
            echo -e "${GREEN}Generated sequence for ${WHITE}$relative_path${NC}"
            ((PROCESSED_FILES++))
        else
            echo -e "${YELLOW}Sequence for ${WHITE}$relative_path${YELLOW} exceeds maximum length (${#sequence}). Skipping.${NC}"
            log_error "Sequence for $relative_path exceeds maximum length (${#sequence})"
        fi
    else
        echo -e "${RED}Error generating sequence for ${WHITE}$relative_path${NC}. Check the error log."
    fi
    echo -e "${WHITE}==================================${NC}"
}

echo -e "${GREEN}Generating sequences for all Python files in the designpatterns directory and its subdirectories...${NC}"

# Check if the designpatterns directory exists
if [ ! -d "$DESIGN_PATTERNS_DIR" ]; then
    echo -e "${RED}Error: The designpatterns directory does not exist.${NC}"
    log_error "The designpatterns directory does not exist"
    exit 1
fi

# Clear the error log file and create/clear the sequences CSV file with header
> "$ERROR_LOG"
echo "path,sequence,label" > "$SEQUENCES_FILE"

# Get total number of Python files (excluding seqgen.py)
TOTAL_FILES=$(find "$DESIGN_PATTERNS_DIR" -name "*.py" ! -name "seqgen.py" | wc -l)

# Process all .py files in the designpatterns directory and its subdirectories
find "$DESIGN_PATTERNS_DIR" -name "*.py" | while read -r file; do
    # Skip seqgen.py
    if [[ $(basename "$file") != "seqgen.py" ]]; then
        generate_sequence "$file"
        REMAINING=$((TOTAL_FILES - PROCESSED_FILES))
        echo -e "${WHITE}Processed: ${GREEN}$PROCESSED_FILES${WHITE} | Remaining: ${WHITE}$REMAINING${WHITE} | Total: ${WHITE}$TOTAL_FILES${NC}"
    fi
done

echo -e "${GREEN}Sequence generation complete.${NC}"

# Check if there were any errors
if [ -s "$ERROR_LOG" ]; then
    echo -e "${WHITE}There were some errors during sequence generation. Please check ${WHITE}$ERROR_LOG${WHITE} for details.${NC}"
else
    echo -e "${GREEN}No errors were encountered during sequence generation.${NC}"
    rm "$ERROR_LOG"
fi

# Inform about the sequences file
echo -e "${GREEN}Generated sequences have been saved to ${WHITE}$SEQUENCES_FILE${GREEN}.${NC}"
echo -e "${WHITE}Total files processed: ${GREEN}$PROCESSED_FILES${WHITE} out of ${WHITE}$TOTAL_FILES${NC}"