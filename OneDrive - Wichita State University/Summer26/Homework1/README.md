EthanZahner-CS898BA-Project1

CS 898BA – Image Analysis and Computer Vision
Homework One

Project Description

This project uses Python and OpenCV to do basic image analysis and processsing on an image. The goal of the assignment is to take an original image, analyze it, convert it into different formats, apply transformations, and then apply Gaussian blur at different sigma levels.

The project also includes a basic Hello World script, an AI usage log, and this README file.

Programs and Libraries Used

Python
OpenCV
NumPy
VSCode
Git/GitHub

Project Files

hello_world.py
This is the first script for the project. It prints Hello World.

src/part2.py
This is the main script for Part 2. It loads the image, calculates statistics, creates different versions of the image, applies affine transformations, and applies Gaussian blur.

AI_Log.md
This file keeps track of AI usage for the project.

requirements.txt
This file lists the Python libraries needed to run the project.

images/input/
This folder stores the original image.

images/output/part2/
This folder stores all of the processed images and the statistics CSV file.

Setup Instructions

1. Open the project folder in VSCode.

2. Create a virtual environment.

py -m venv .venv

3. Activate the virtual environment.

..venv\Scripts\Activate.ps1

4. Install the required libraries.

pip install -r requirements.txt

Running the Hello World Script

To run the Hello World script:

python hello_world.py

The expected output is:

Hello World!

Running the Part 2 Script

The original image should be placed in this folder:

images/input/

The script currently expects the image to be named:

alien.png

If the image has a different name, the file path in src/part2_processing.py needs to be changed.

To run the Part 2 script:

python src/part2.py

The output images will be saved in:

images/output/part2/

Part 2 Explanation

For Part 2, the script first loads the original image. It then calculates statistics for each color channel of the original image. Since OpenCV readss images in BGR order, the channels are Blue, Green, and Red.

The statistics calculated for each channel are:

minimum
maximum
average
median
mode
skew
range
standard deviation
variance

These statistics are saved to a CSV file called:

original_image_channel_statistics.csv

The script then creates the 7 starting images:

original image
grayscale image
binary image
HSV image
CIELAB image
HLS image
HSV value-channel equalized image converted back to a normal color image

grayscale simplifies the image into brightness values instead of color. The binary image turns it into mostly black and white values. HSV, CIELAB, and HLS are different color spaces that separate color and brightness in different ways

For the HSV image, the script does histogram equalization on the V channel. The V channel shows value, or brightness. Eqaulizing this channel helps normalize the lighting and improve contrast without directly changing the hue and saturation channels.

After making the starting images, the script performs 2 affine transformations on each imageto make a total of 14 transformed images.

The transformations used are:

rotation with scaling
translation

The script uses random values with a seed so the results are random but repeatable

There are now:
7 starting images
14 transformed images

After that, Gaussian blur is applied to each of the 21 images.

The sigma values used are:

0.5
1.0
1.5
2.0
2.5
3.0
3.5

This creates 147 blurred images because 21 images are each blurred 7 times.

The total number of images after Part 2 is:

21 original/transformed images + 147 blurred images = 168 images

Gaussian Blur Discussion

Gaussian blur smooths an image. A lower sigma value causes less blur, and a higher sigma value causes more blur.

At sigma 0.5, the image is only slightly blurred and most details are still visible.

At sigma 1.0, the blur is more noticeable, but the image still keeps most of its shape and detail.

At sigma 1.5 and 2.0, small details start to disappear and edges become softer.

At sigma 2.5 and 3.0, the image becomes much smoother and fine details are harder to see.

At sigma 3.5, the blur is the strongest. The image loses a lot of sharpness, and object boundaries are less clear.

increasing sigma makes the image smoother, but it also removes detail. Lower sigma values are better for keeping detail and higher sigma values are better for stronger smoothing.

Expected Output

The Part 2 script should create:

7 base images
14 affine-transformed images
147 Gaussian-blurred images

This gives a total of 168 images.
The script also creates one CSV file with the original image statistics

Git Usage

This project uses Git and GitHub for version control. I forgot to make incremental commits with part 2, but they were made with part 3 as I added the project setup, code, image outputs, README updates, and AI log updates.

AI Usage

AI was used to help explain OpenCV tools, project setup, code structure, and README writing. All AI use is recorded in AI_Log.md with the prompt, date and time, tool used, response summary, and changes made because of the response.

