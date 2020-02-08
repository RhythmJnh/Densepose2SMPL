# Densepose2SMPL
  Densepose2SMPL is a implementation of converting densepose results to smpl.

# Install
  Create a virtual environment, and run:
  ```
    sh install.sh
  ```

# Usage
  Download the [smpl models](https://pan.baidu.com/s/1JTiyaxgocJgN2-6lSNE53A), unzip to models folder.  
  If setup properly, the folder structure should look like:
  ```
    .
    ├── models/
        ├── smpl/
            ├── %^%$^#%^$.pkl
            ├── %^&$^%$^%.pkl
            ├── ##^$$$^%&.pkl
            └── ...
  ```
  There are some modifications in ```process_densepose.py```.  
  Under the change_cloth project(environment), run```get_IUV.py```, get iuv.npy files.  
  Activate this environment, then run ```main.py```.  
  Please find more details in the code.
