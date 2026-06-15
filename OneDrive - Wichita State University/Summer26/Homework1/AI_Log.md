## Entry 1

**Date and Time:** 06/14/2026, 1:12

**Prompt:**
Give me a basic rundown of OpenCV and it's tools with simple code examples that I can apply myself to the project in relation to part 2, please don't "give me the answer" and just provide me with examples of OpenCV keywords and examples of how they can be used in the project:

Part 2: You know you should at least do basic analysis to get started, so you perform the following on the image:

Find and print basic image statistics of the original image for each individual channel (min, max, average, median, mode, skew, range, standard deviation, variance)
Convert and save the image to greyscale, binary, and different color spaces (HSV, CIELAB, and HLS).
On the HSV converted image, normalize the lighting by performing histogram equalization across the V (value) channel.
Convert the normalized image back to RGB and save it.
You should now have 7 images.
Perform random affine transformations on each image (you should perform 14 total transformations - 2 for each image). Affine transformations can be translation, rotation, scaling, or shear as long as each is unique in either transformation type or transformation value (rotate 90 degrees vs rotate 186 degrees). No two images should be transformed in the exact same way. Save each of those images to new files.
You should now have 21 images.
Apply a Gaussian blur to each image using the levels of sigma: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. Discuss how the level of sigma changes the image. Save each of those images to new files.
You should now have 168 images.

Also generate an addition of this message for the ai log

**Tool:** ChatGPT

**Response Synopsis:**
ChatGPT explained the basic OpenCV tools needed for Part 2, including reading and saving images, converting color spaces, thresholding to binary, splitting and merging channels, histogram equalization, affine transformations, and Gaussian blur. It also provided Python example scripts for how to generate the required base images, transformed images, Gaussian-blurred images, and channel statistics.

**Change:**
Used the guidance to create part2.py, which loads the original image, calculates channel statistics, creates grayscale,binary,HSV,CIELAB,HLS,normalized images, performs random (seeded) affine transformations, applies Gaussian blur at multiple sigma levels, and saves the output files for Part 2.

## Entry 2

**Date and Time:** 06/14/2026, 4:40

**Prompt:**
I need some help with my code, I'm getting syntax errors with my affine transformation code. Can you take a look and help me trouble shoot? I also want to make sure that the image file is being loaded correctly if you could look at that

*attached current part2.py

**Tool:** ChatGPT

**Response Synopsis:**
ChatGPT corrected my syntax errors and showed me a cleaner way to display the code, and recommended adding a file check at the top of the program

**Change:**
Implemented the corrected syntax and file check

## Entry 3

**Date and Time:** 06/14/2026, 5:20

**Prompt:**
Generate a "skeleton" read me with just the basic structure for me to fill out with what I've done so far


**Tool:** ChatGPT

**Response Synopsis:**
Generated a simple readme structure with titles for the guassian blur section, project setup, etc

**Change:**
Copied it into my current readme and filled it out with my own information

## Entry 4

**Date and Time:** 06/14/2026, 6:00

**Prompt:**
How would I write a program to quickly sort the images into different folders and allocate them space accordingly? Here are the requirements:
Randomly create 4 equally sized subsets of the images from part 1.
Each subset should have 42 images.
Choose a subset to use in the remaining steps.
You should now have 42 images.


**Tool:** ChatGPT

**Response Synopsis:**
Generated code that sorts the images into files and confirms that the correct amount of images exist in each one beforehand
**Change:**
Added it to part3.py

## Entry 5

**Date and Time:** 06/14/2026, 6:28

**Prompt:**
Can you explain how to syntax for Sobel, Laplacian, Canny, Prewitt works and provide me with some simple examples? Once again, please don't "Provide me with the answer" I want to see examples


**Tool:** ChatGPT

**Response Synopsis:**
Generated code that showed the proper syntax for applying the different edge detection techniques
**Change:**
Plan on applying it to my own code

## Entry 6

**Date and Time:** 06/14/2026, 6:45

**Prompt:**
Look over the code I have written for the process_selected_subset and check it for errors or improvements to optimize it
*attached code*

**Tool:** ChatGPT

**Response Synopsis:**
Shorted parts of the code that were too long and added a section for the 5 image plot for a future function
**Change:**
Applyed it to my version of the code and updated the comments to be more in line with my understanding.