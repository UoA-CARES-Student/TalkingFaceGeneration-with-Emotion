#! /bin/bash


#
#
#
# NOT USED
#
#
#

mkdir -p voxceleb_preprocessed_6

files=`ls voxceleb_preprocessed_5`

echo $files

for file in $files; do 
	cp -r voxceleb_preprocessed_5/"$file"/* voxceleb_preprocessed_6 
	#echo -r dataset2/mp4/"$file"*/ dataset2/mp4/test; 
done