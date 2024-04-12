#!/bin/bash

# Compile and run all .cu files in the current directory except specified ones
for file in *.cu; do
    if [ -f "$file" ] && [ "$file" != "softmaxCuda_main_experiments.cu" ] && [ "$file" != "softmaxCuda_CuDNN.cu" ]; then
        filename="${file%.*}"  # Extract filename without extension
        echo "Compiling $file..."
        nvcc "$file" -o "$filename"
        if [ $? -eq 0 ]; then
            echo "successfully compiled $filename..."
            #./"$filename"
            echo "-------------------------------------"
        else
            echo "Compilation failed for $file"
        fi
    fi
done
