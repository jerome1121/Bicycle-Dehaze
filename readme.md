How to run ?
========================

sh main.sh src_dir des_dir

`src_dir`: path to your test data dir which includes 51.png ~ 55.png

`des_dir`: destination directory where the dehazed images saved


code inside main.sh
-----------------------

`utils/resize_crop_shift.py`: crop the test data into 512\*512 patches

`test.py`: dehazing code which takes 512\*512 hazy patches as input

`utils/merge.py`: merge 512\*512 dehazed patches produced from previous step into full image
