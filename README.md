# Semi-Supervised Holistic Human Mesh Annotator
For undegraduate students, please follow the instruction below. Also, note that we will use `SMPL-X` not `SMPL`.\
Please check their [Project Page](https://smpl-x.is.tue.mpg.de/), [Paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf), and [Code](https://github.com/vchoutas/smplify-x) for more details information.\
**Note: If you use this application, means that you also agree with the third party's agreement that we use in this project**
## News <img src="https://user-images.githubusercontent.com/34101139/153993021-64e8fb1e-74fe-4100-b004-5c8d9a0072db.png" width="100" height="100">
## 1. Code Directory
Please download `data/` directory from [here](https://binusianorg-my.sharepoint.com/personal/joshua_santoso_binus_ac_id/Documents/undergrade/data?csf=1&web=1&e=wJdtVl) and the directory should be same as shown below.
```
commons/
  |- augmentation.py
  |- camera.py
  |- render.py
  |- ...
data/
  |- conf.yaml
  |- human_models/
    |-kids/
    |-smpl/
    |-smplx/
  |- body_model.h5
examples/
 |- test.jpg
human_models/
networks/
demo.py
```

## 2. Environment Preparation
**Note: We run our project on Linux.** \
Please download the anaconda environment in [here](https://binusianorg-my.sharepoint.com/personal/joshua_santoso_binus_ac_id/Documents/undergrade/human-app.yml?csf=1&web=1&e=uakmP4).\
If you are running on Windows or MacOS, we are not sure whether you can run smoothly or not. If you have any question or problem please contact me

## 2. Run Demo
Please run the code by using command prompt and type `python demo.py`.

