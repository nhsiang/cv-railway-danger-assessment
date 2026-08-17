# cv-railway-danger-assessment
Computer vision research project using OpenCV and Ultralytics YOLO. Analyze railway tracks and return appropriate responses based on different situations.
## Steps
- ### YOLO object detection
  - Use the default YOLO26 nano model
  - Draw out bounding boxes, if overlaps with `rail-track`, `rail-raised`, or `rail-embedded` masks, a warning is displayed on screen
- ### YOLO semantic segmentation for rail curvature calculation
  - Use the YOLO26 nano semantic model
  - Train the model using `python train.py`, set the argument `epochs=50`
  - After the model yields sufficient results, isolate the `rail-track` mask
  - From top to bottom of the `rail-track` mask, scan the $y\text{-axis}$ to see if the mask exist at that $y$, if so, calculate the midpoint $\frac{x_\text{left_most}+x_\text{right_most}}{2}$ and store it in `coords`
  - Plot a second degree polynomial using `coords` derived from above, and calculate the curvature using the formula $κ(y)=\frac{|f''(y)|}{(1+[f'(y)]^2)^\frac{3}{2}}$
  - The final curvature used to determine the message on screen is $\kappa(y)_\text{median}$, if $\kappa(y)_\text{median}>0.001$, the railway is curved and a warning message is displayed

  **IMPORTANT: This should not be used as the determining factor of curvature on railways! For demonstration purposes, the formula is simplified. To achieve an industry-accepted curvature, more data of the vehicle is required.**
## Results
- The trained model was able to achieve the following mIoU results:

  | Overall | rail-track | rail-raised | rail-embedded |
  |---------|------------|-------------|---------------|
  | 0.74075 | 0.867      | 0.597       | 0.514         |
- Demonstration:
## Known issues
- Cannot identify which railway the vehicle is running on, to be added in the future
## Acknowledgement
This project utilizes code from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under AGPL-3.0.<br>
This project utilizes RailSem19 dataset provided by the AIT Austrian Institute of Technology GmbH as the training dataset.

[Zendel et al. "RailSem19: A Dataset for Semantic Rail Scene Understanding." IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2019.](https://mlanthology.org/cvprw/2019/zendel2019cvprw-railsem19/) doi:10.1109/CVPRW.2019.00161