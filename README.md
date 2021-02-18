# thermal-object-detection

## Overview
This project is about using a bunch of thermal images in a folder, to track/count the objects in them using SimpleBlobDetector from OpenCV and by using basic computer vision principles to extract valuable features.

## Authors

- [Vinayak Ravi](https://github.com/vinayakravi14)


## Run Instructions

- Clone the repository 
```
git clone git@github.com:vinayakravi14/thermal-object-detection.git
cd ~/object_tracking_CV/scripts
```

- Then launch the script file (Note: If you dont specify any args, it uses the sample_data  and file_ext)
```
python3 detector.py '/path/to/folder/' '.file_format'
For example: python3 detector.py '../data/footy2/' '.png'
```
## Example detector window, after running the script


<img src="https://github.com/vinayakravi14/thermal-object-detection/blob/main/object_tracking_CV/sample/output.png" alt="sample_output"/>



