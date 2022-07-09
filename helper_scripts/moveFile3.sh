#! /bin/bash

#### 
#
# Used to restructure files for the full voxceleb dataset. (on the hard drive)
# Creates symbolic links
#
# Version 3: Need to split the dataset into multiple chunk, because someone keeps killing my process. 
#
####

mkdir -p "/mnt/DataSets/tish386/test_chunk_1"
mkdir -p "/mnt/DataSets/tish386/test_chunk_2"
mkdir -p "/mnt/DataSets/tish386/test_chunk_3"
mkdir -p "/mnt/DataSets/tish386/test_chunk_4"
mkdir -p "/mnt/DataSets/tish386/test_chunk_5"
mkdir -p "/mnt/DataSets/tish386/test_chunk_6"



files=`ls /mnt/DataSets/tish386/dev/mp4`
output_path=`ls /mnt/DataSets/tish386/test`


echo $files | wc 
echo $output_path | wc


# THERE IS TOTAL OF  5900-sh folders
counter=0
for file in $files; do 
	# ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test 

	let counter++  

	if [[ "$counter" -lt 1000 ]]; then
		# printf "lt 1000 : Counter : %d \n" $counter
		ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test_chunk_1 

	elif [[ "$counter" -lt 2000 ]]; then
		# printf "lt 2000 : Counter : %d \n" $counter
		ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test_chunk_2 

	elif [[ "$counter" -lt 3000 ]]; then
		# printf "lt 3000 : Counter : %d \n" $counter
		ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test_chunk_3 

	elif [[ "$counter" -lt 4000 ]]; then
		# printf "lt 4000 : Counter : %d \n" $counter
		ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test_chunk_4 

	elif [[ "$counter" -lt 5000 ]]; then
		# printf "lt 5000 : Counter : %d \n" $counter
		ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test_chunk_5 

	elif [[ "$counter" -lt 6000 ]]; then
		# printf "lt 6000 : Counter : %d \n" $counter
		ln -s --backup=numbered /mnt/DataSets/tish386/dev/mp4/"$file"/* /mnt/DataSets/tish386/test_chunk_6 

	else
		printf "Should not print : Counter : %d \n" $counter
	fi
	#echo -r dataset2/mp4/"$file"*/ dataset2/mp4/test; 
done

# ln -s /media/myuser1/Seagate\ Expansion\ Drive/p4p-g24-2022/VoxCeleb2-Dataset/raw_dataset/dev/mp4/id00012/* /media/myuser1/Seagate\ Expansion\ Drive/p4p-g24-2022/VoxCeleb2-Dataset/raw_dataset/test
# ln -s /media/myuser1/Seagate\ Expansion\ Drive/p4p-g24-2022/VoxCeleb2-Dataset/raw_dataset/dev/mp4/id00015/* /media/myuser1/Seagate\ Expansion\ Drive/p4p-g24-2022/VoxCeleb2-Dataset/raw_dataset/test

#cp -r id00817/* test
