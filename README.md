# Generating abutting grating illusion samples

## Codes to generate datasets from "Abutting Grating Illusion: Cognitive Challenge to Neural Network Models"

This repository provides the code samples to generate abutting grating(AG) corruption samples with MNIST, high resolution MNIST and 160 silhouettes images.
### Example images of AG samples
#### AG-MNIST sample with horizontal gratings of interval 4
![ag_mnist_sample](https://user-images.githubusercontent.com/48897111/197470008-1db59fe8-010c-4b1e-8da4-ad15451f08ce.png)
#### high resolution AG-MNIST sample with horizontal gratings of interval 8
![high_resolution_ag_mnist_sample](https://user-images.githubusercontent.com/48897111/197470027-86c40762-b280-47f4-9176-caf974a9c1a2.png)
#### AG-silhouette sample with horizontal gratings of interval 8
![ag_silhouette_sample](https://user-images.githubusercontent.com/48897111/197470041-2f2de16b-c116-43ca-ab52-b9221b45430c.png)

The high resolution MNIST image samples are achieved by first interpolating the original MNIST samples from 1x28x28 to 3x224x224.
The 160 silhouette images are proposed by "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness":
https://github.com/rgeirhos/texture-vs-shape

Three python scripts are provided respectively. In each script, a function is implemented to accomplish the AG corruption.
You can change the parameter "interval" to alter the interval between abutting gratings.
You can change the parameter "direction" to alter the direction of the abutting gratings.
