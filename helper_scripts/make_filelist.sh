#! /bin/bash

train_filename="filelists/train.txt"
val_filename="filelists/val.txt"

touch $train_filename
touch $val_filename
> $train_filename
> $val_filename

folders=`find /root/takahiro/mini-preprocessed -mindepth 2 -maxdepth 2 -type d`
# echo $folders

num_folders=`find /root/takahiro/mini-preprocessed -mindepth 2 -maxdepth 2 -type d | wc -w`

test_val_ratio=80 #In percentage

test_num=$((num_folders * test_val_ratio / 100))


counter=0
for file in $folders; do
	counter=$(($counter+1))
	
	if ((counter <= test_num)); then
		echo "Test_sample: filename: ${file} | counter: ${counter}";
		echo ${file} >> $train_filename
	else
		echo "Val_sample:  filename: ${file} | counter: ${counter}";
		echo ${file} >> $val_filename
	fi
done


echo $num_folders

echo $test_num

# echo $counter
