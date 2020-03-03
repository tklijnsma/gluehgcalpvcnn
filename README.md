# Temporary project in an effort to implement `pvccn` for HGCAL reconstruction

The functionality of this repo is to be implemented in https://github.com/lgray/hgcal_ldrd.


## Environment

The code is currently being developed in the following conda environment: [environment.yml](environment.yml).


## Install instructions

```
git clone --branch dev-pvcnn https://github.com/tklijnsma/hgcal_ldrd.git
git clone https://github.com/mit-han-lab/pvcnn.git
git clone https://github.com/tklijnsma/gluehgcalpvcnn.git
```

You will also need data samples. Contact me if you want to run all this stuff.


## Example run

Create a python file with the following contents and run it:

```
import os, sys, shutil
import os.path as osp

from gluehgcalpvcnn.trainingscript import LindseysTrainingScript

HGCAL_LDRD_PATH = osp.abspath('hgcal_ldrd/src')
PVCNN_PATH = osp.abspath('pvcnn')

sys.path.append(HGCAL_LDRD_PATH)
from datasets.hitgraphs import HitGraphDataset
# Make sure datasets import is from hgcal_ldrd, not from pvcnn
# Then add pvcnn to the path
sys.path.append(PVCNN_PATH)
# This is a very ugly fix

def main():
    script = LindseysTrainingScript(debug=False)
    if 'testsample' in script.dataset_path:
        # Force reprocessing for the test sample
        processed_path = osp.join(script.dataset_path, 'processed')
        print('Removing {0}'.format(processed_path))
        shutil.rmtree(processed_path)
    script.run_edgenet() # Ignore misnomer, will actually run pvcnn

if __name__ == '__main__':
    main()
```