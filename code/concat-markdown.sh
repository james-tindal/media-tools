#!/usr/bin/env sh
# Usage: ./concat-md.sh output.md file1.md file2.md ...

output="concatenated.md"

for file in "$@"; do
  title="$(basename "${file}" .md)"
  echo "# ${title}" >> "$output"
  echo >> "$output"
  cat "$file" >> "$output"
  echo >> "$output"
done
