# Image Stylization by Edge-Aware Processing

data
----
* `statue.png` and `burger.jpg` are the original images
* `*_nc.png` refers to the "normalized convoluted" images
* `brush.jpg`, `water.jpg`, `stroke.jpg` are the stroke patterns used for style transfer on the burger image

code
----
* `src/NC.py` implements domain transformed edge-aware processing, specifically the normalized convolution. This is largely based on [Gastal et al. (2011)](https://www.inf.ufrgs.br/~eslgastal/DomainTransform/).
* `src/stylization.py` defines several functions:
    - `vanilla` smoothes the given image using domain transformed edge-aware processing
    - `border` extracts the edges given any image. Ideally it should be run on a smoothed image
    - `merge_border` merges the smoothed image with the edges. It can be used to merge a lightly smoothed image with edges extracted from a more heavily smoothed image or vice versa
    - `style_transfer` takes in an smoothed image and a brush pattern and apply style transfer using joint filtering
