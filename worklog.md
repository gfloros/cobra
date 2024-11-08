# Worklog

## Thursday, 7th November 2024

### Preamble
Before starting the experiments, I created a virtual environment using Python 3.11.10 (later Python versions were not compatible with [Open3D](https://www.open3d.org)), in which I tried to get the minimum amount of requirements needed in order to run the code successfully. To this end, I created a file called `requirements.minimal.txt` and I used the following command in order to install the necessary Python packages
```
python3 -m pip install -r requirements.minimal.txt
```

As a side note, I had to pin `numpy==1.26.4`, because the latest version was not [compatible](https://stackoverflow.com/questions/78778444/segmentation-fault-when-pass-a-numpy-array-to-open3d-utility-vector3dvector) with Open3D. 

### Scale problem debugging
Started doing debugging in the COBRA codebase because the [experiments](https://docs.google.com/document/d/1DaCrC-yPST5EM3ehiPs5iRLaVmTIbdvhyCMKaFlAoRw/edit?tab=t.0) have shown that scaling the 3D model with a scaling factor affects the shape representation and thus the model's ability to produce robust scores for pose estimates. To this end, I added some logs and I also save the virtual camera positions.

In the main branch and without having done any changes to the code, apart from the aforementioned ones (logs addition and saving of the virtual camera positions) I run the following script:
```
python3 preprocess.py --class-name planes
```

The result (normalized model + camera positions) is illustrated is the following figure

![Normalized model + camera positions](screenshots/model_with_cameras00.png "Normalized model + camera positions")

Afterwards, I run the script:
```
python3 preprocess.py --class-name planes --no-normalize
```

The result (unnormalized model + camera positions) is illustrated is the following figure

![Unnormalized model + camera positions](screenshots/model_with_cameras01.png "Unnormalized model + camera positions")

By comparing the two images, it's evident that the camera positions are not scaled with the same scaling factor as the model is. By inspecting the code, I found the following "suspicious" things:

1. When the model is scaled to the unit sphere, the [scaling factor](https://github.com/pansap99/COBRAv1.2/blob/ff906bf75ac71e339d72aada573b440f2350ee60/utils/model_3d.py#L290) is multiplied by an ad-hoc factor `1.03`.
2. When scaling the camera positions there is another adhoc [scaling factor](https://github.com/pansap99/COBRAv1.2/blob/ff906bf75ac71e339d72aada573b440f2350ee60/utils/model_3d.py#L153) `1.3` (should it have been `1.03`?).

In any case, the scaling of the model and the cameras should be 100% identical, if we want to have scale invariant results.

To solve this problem, I propose the folling change:

- We should use the function `get_model_size` to estimate the scaling factor of the cameras. In that case the cameras will always we scaled with the same scaling factor relative to the model.