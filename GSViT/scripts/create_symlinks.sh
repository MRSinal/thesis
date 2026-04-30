# 1. Clean out the broken links
rm -rf /home/u941663/thesis/data/all_frames/*

# 2. Create absolute links for PitViS
# Note: realpath converts the folder to its full /home/u941663/... path
for d in /home/u941663/thesis/data/PitViS/frames/*/; do
    ln -s "$(realpath "$d")" "/home/u941663/thesis/data/all_frames/$(basename "$d")"
done

# 3. Create absolute links for CATARACTS
for d in /home/u941663/thesis/data/CATARACTS/frames/*/; do
    ln -s "$(realpath "$d")" "/home/u941663/thesis/data/all_frames/$(basename "$d")"
done

for d in /home/u941663/thesis/data/JIGSAWS/frames/*/; do
    ln -s "$(realpath "$d")" "/home/u941663/thesis/data/all_frames/$(basename "$d")"
done
