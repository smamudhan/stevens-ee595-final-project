#!/bin/bash

# Directory containing images (set the path to your folder)
IMAGE_DIR="./AI"

# Initialize counters
total_images=0
non_256x256_count=0

# Check if the directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Directory $IMAGE_DIR does not exist."
    exit 1
fi

# Iterate through image files in the directory
for image in "$IMAGE_DIR"/*; do
    if [[ -f "$image" ]]; then
        # Increment total image count
        ((total_images++))
        
        # Get image dimensions
        dimensions=$(identify -format "%wx%h" "$image" 2>/dev/null)
        
        # Check if dimensions are 256x256
        if [[ "$dimensions" != "256x256" ]]; then
            ((non_256x256_count++))
        fi
    fi
done

# Calculate the percentage
if ((total_images > 0)); then
    percentage=$(echo "scale=2; ($non_256x256_count / $total_images) * 100" | bc)
    echo "Total images: $total_images"
    echo "Images not 256x256: $non_256x256_count"
    echo "Percentage of images not 256x256: $percentage%"
else
    echo "No images found in the directory."
fi
