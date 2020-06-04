# Deep neural nerwork for sound representation

This repo implements kell2018 model for auditory cortex as presented <a href="https://www.cell.com/neuron/fulltext/S0896-6273(18)30250-2">Kell et al., 2018</a>. 
* You can find a pytorch implementation of this model in`Kell model.py`.
* `Kelletal_2018.ipynb` shows a tensorflow implementation of this model and a reproduction of the results presented in the paper.
* For feature extraction or to reproduce the results you can find weights and stimuli  <a href="https://github.com/Ouasfi/kelletal2018">here</a>
## Dependencies
Most of the dependencies to run this are relatively standard. However, please note the following: 
- This notebook was tested and run with version 2.0 of `tensorflow`. It was not tested with other versions.
- `pycochleagram` is a module to generate cochleagrams to pass sounds into the network, which can be found <a href="https://github.com/mcdermottLab/pycochleagram">here</a>. To install it run the following commands:
    * `git clone https://github.com/mcdermottLab/pycochleagram`
    * `cd pycochleagram`
    * `python setup.py install`

- `PIL` is the Python Image Library.
