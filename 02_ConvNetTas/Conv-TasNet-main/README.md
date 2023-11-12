# Conv-TasNet
tensorflow 2.x implementation of [Conv-TasNet, Y Luo et al.(2018)](https://github.com/naplab/Conv-TasNet)

`.py` structure is more appropriate for this project, but I used the jupyter notebook with the `import_ipynb` module, in order to make things more natural.

Main files are `CV_Tasnet_Model.ipynb`, `CV_Tasnet_CustomTraining.ipynb`, `CV_Tasnet_main.ipynb`.<br />
`CV_Tasnet_CustomTraining.ipynb` is for making custom loss, datasets and so on.<br />
`CV_Tasnet_Model.ipynb` is for making the main model.<br />
Finally, `CV_Tasnet_Model.ipynb` is for training the model.<br />
`CV_Tasnet_Unused.ipynb` is just for record.

As the WSJ0-2mix Dataset is NOT open source dataset, I used [LibriMix](https://github.com/JorisCos/LibriMix) dataset instead. 
