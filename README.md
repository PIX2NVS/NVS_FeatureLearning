# Graph-based Spatial-temporal Feature Learning for Neuromorphic Vision Sensing

## Summary
This is the implemtation code and proposed dataset(ASL-DVS) for the following paper. Please cite following paper if you use this code or dataset in your own work. The paper is available via: https://arxiv.org/abs/1910.03579.

MLA:

    Bi, Y., Chadha, A., Abbas, A., Bourtsoulatze, E., & Andreopoulos, Y. (2019). Graph-based Spatial-temporal Feature Learning for Neuromorphic Vision Sensing. IEEE Transactions on Image Processing, 9084-9008, 2020
    
BibTex:

    @article{bi2020graph,
     title={Graph-Based Spatio-Temporal Feature Learning for Neuromorphic Vision Sensing},
     author={Bi, Yin and Chadha, Aaron and Abbas, Alhabib and Bourtsoulatze, Eirina and Andreopoulos, Yiannis},
     journal={IEEE Transactions on Image Processing},
     volume={29},
     pages={9084--9098},
     year={2020},
     publisher={IEEE}
     }



## Dataset: UCF101-DVS, HMDB-DVS, ASLAN-DVS 
We release largest neuromorphic human activity datasets including UCF101-DVS, HMDB-DVS and ASLAN-DVS, and make them available to the research community at the link: https://www.dropbox.com/sh/ie75dn246cacf6n/AACoU-_zkGOAwj51lSCM0JhGa?dl=0


## Code Implementation
### Requirements:
     Python 2.7 
     Pytorch 1.0.1.post2
     pytorch_geometric 1.1.2
     
### Preparations:
    Training graphs are saved in '../traingraph' folder.
    Testing graphs are saved in '../testgraph' folder.
    Each sample should contains feature of nodes, edge, pseudo adresses and label.
    
### Running examples:
    cd code
    python main.py   # running file for RGCNN+Plain 3D 
    
    #The results can be found in the 'Results' folder.



## Contact 
For any questions or bug reports, please contact Yin Bi at yin.bi.16@ucl.ac.uk .
