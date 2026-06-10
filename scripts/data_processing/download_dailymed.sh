#!/bin/bash
# ==============================================================================
# Simple DailyMed Human Prescription Download & Unzip Script
# ==============================================================================

# --- Configuration ---

# Find the directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Assume the project root is one level above the script's directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Set the main directory where all data will be stored, inside the project's data folder.
TARGET_DATA_DIR="$PROJECT_ROOT/data/raw/dailymed_prescription"

# Define subdirectories for zips and the final unzipped data.
ZIPS_DIR="$TARGET_DATA_DIR/zips"
UNZIPPED_DIR="$TARGET_DATA_DIR/unzipped_data"

# An array of the local filenames we want to save the files as.
declare -a RX_FILES=(
    "dm_spl_release_human_rx_part1.zip"
    "dm_spl_release_human_rx_part2.zip"
    "dm_spl_release_human_rx_part3.zip"
    "dm_spl_release_human_rx_part4.zip"
    "dm_spl_release_human_rx_part5.zip"
)

# An array of the exact, full URLs to download from.
declare -a RX_URLS=(
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip"
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part2.zip"
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part3.zip"
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part4.zip"
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part5.zip"
)


# --- Main Execution ---

echo "================================================="
echo "Starting DailyMed Prescription Data Download"
echo "Target Directory: $TARGET_DATA_DIR"
echo "================================================="

# Create the main data directory and subdirectories
mkdir -p "$ZIPS_DIR"
mkdir -p "$UNZIPPED_DIR"

# --- Download All Files ---
echo -e "\n--- Downloading Human Prescription Files ---"
for i in "${!RX_FILES[@]}"; do
    filename="${RX_FILES[$i]}"
    url="${RX_URLS[$i]}"
    filepath="$ZIPS_DIR/$filename"

    if [ -f "$filepath" ]; then
        echo "  -> File '$filename' already exists. Skipping download."
    else
        echo "  -> Downloading '$filename'..."
        # Use wget with -P to specify the output directory.
        wget -q --show-progress -c -P "$ZIPS_DIR" "$url"
    fi
done

echo -e "\nAll files have been downloaded."


# --- Unzip All Files ---
echo -e "\n================================================="
echo "Unzipping all files into the '$UNZIPPED_DIR' directory..."
echo "================================================="

# Unzip all .zip files from the zips directory into the target directory.
unzip -q -o "$ZIPS_DIR/*.zip" -d "$UNZIPPED_DIR"

echo -e "\n================================================="
echo "✅ All tasks complete!"
echo "Your data is ready in the '$UNZIPPED_DIR' directory."
echo "================================================="
