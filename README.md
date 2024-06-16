This file contains two sections:
1. Image alignment.
2. Target reconstruction.

The algorithm is crafted in Python and executed within Jupyter Notebook. It has minimal hardware requirements, and a standard computer configuration is sufficient to meet its needs during execution.

How to run the code:
Step 1: Change the colors in Cy5, Cy3 and Fam channels to R (255, 0, 0), B (0, 0, 255), and G (0, 255, 0). Then, the images obtained in time point 1, time point 2 and time point 3 are exported to tiff format in merged channels and named as m1, m2 and m3.
Step 2: Run Part 1 to align m1, m2 and m3.
Step 3: Run Part 2 to obtain the reconstructed image for each target.
Step 4: Merge the reconstructed images via zen software (Zeiss).

Dataset for testing: Test images.
