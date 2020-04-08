#!/bin/bash/

src_dir=$1
des_dir=$2


echo "**************************************************"
echo "*  1st Stage: Crop test data to 512*512 patches  *"
echo "**************************************************"
python3 utils/resize_crop_shift.py --unprocessed_dir $src_dir --processed_dir crop_test_512


echo "**************************************************"
echo "*           2nd Stage: Dehazing.....             *"
echo "**************************************************"
python3 test.py --input_dir crop_test_512 --output_dir crop_output_512


echo "**************************************************"
echo "*      3rd Stage: Merging dehazed patches        *"
echo "**************************************************"
python3 utils/merge.py --unprocessed_dir crop_output_512 --processed_dir $des_dir


rm -rf crop_test_512
rm -rf crop_output_512




