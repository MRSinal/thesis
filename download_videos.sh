#!/bin/bash

# Configuration
SAVE_PATH="videos"
DIR_PATH="videos_to_download"
mkdir -p "$SAVE_PATH"

# Loop through every .txt file in the directory
for file in "$DIR_PATH"/*.txt; do
    # Read each line (URL) from the file
    while IFS= read -r url || [ -n "$url" ]; do
        # Skip empty lines
        [ -z "$url" ] && continue

        # Extract the Video ID (everything after the last slash)
        # This mimics: capid = _video.split("/")[-1]
        vid_id=$(basename "$url")

        # Check if a file starting with this ID already exists
        # This mimics: if capid in existing: continue
        if ls "$SAVE_PATH/$vid_id."* 1> /dev/null 2>&1; then
            echo "Skipping $vid_id (Already downloaded)"
            continue
        fi

        echo "Downloading $vid_id..."
        
        # Run yt-dlp for this specific video
        yt-dlp \
            --output "$SAVE_PATH/%(id)s.%(title)s.%(ext)s" \
            --cookies-from-browser firefox \
            --write-subs --sub-langs "en.*" --write-auto-subs \
            "$url"

    done < "$file"
done
