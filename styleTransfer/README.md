# Style Transfer

This notebook deals with style transfer with the help of pretrained vgg16 model. Vgg16 is a convolution network for image classification tasks. Here we will not use the fully connected layers. Instead we will use the intermediate hidden layers to calculate content and style loss. Content loss is for the original image. Style loss is for the style image whose style we want to transfer to given image.

Code finds an image that simultaneously matches the content representation of the photograph and the style representation of the respective piece of art.

# Requirments
1. Keras
2. Scipy
3. Numpy
4. PIL