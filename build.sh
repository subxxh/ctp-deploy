#!/bin/bash
echo "Building Annoy models..."
cd annoy_similarity
python build_index.py
cd ..
echo "Build complete!"
