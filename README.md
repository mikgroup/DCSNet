# Direct Contrast Synthesis from MR Fingerprinting
This repository contains implementations of DCSNet in PyTorch. More information can be found [here](https://profs.etsmtl.ca/hlombaert/public/medneurips2019/107_CameraReadySubmission_NeurIPS_2019_DCS_CR.pdf).

## Installation
To use this package, install the required python packages (tested with python 3.6 on Ubuntu 16.04LTS):

```bash
pip install -r requirements.txt
```
## Training a model

Finally, to launch a training session, got to the models folder and run the following script:

```bash
python3 train.py --data-path <path to dataset folder> --exp-dir <path to summary folder> --device-num 0
```
