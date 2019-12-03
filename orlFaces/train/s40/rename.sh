for file in *.pgm; do 
	mv "$file" "${file%.pgm}_40.pgm"
done
