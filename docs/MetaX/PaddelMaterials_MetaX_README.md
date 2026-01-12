# PaddleMaterials on GiteeAI (MetaX)

## Environment Setup on GiteeAI
1. Register and log in to [giteeAI](https://ai.gitee.com/).
2. Purchase computing resources and click **Rent Now**.  
   ![](./pic1.png)
3. Choose the **PaddleMaterials** image and click **Next** to create an instance.  
   ![](./pic2.png)
4. After the instance is created, click **Lab** to enter the container.  
   ![](./pic3.png)
5. In the Lab page, choose **Jupyter Lab**, then open a **Terminal**.  
   ![](./pic4.png)

## Training Process
- PaddleMaterials source directory: `/opt/PaddleMaterials`
- Reference documents:
- [MLIP - Machine Learning Interatomic Potential](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/interatomic_potentials/README.md)
- [MLES - Machine Learning Electronic Structure](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/electronic_structure/README.md)
- [PP - Property Prediction](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/property_prediction/README.md)
- [SG - Structure Generation](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/structure_generation/README.md)
- [SE - Spectrum Elucidation](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/spectrum_elucidation/README.md)

Below is the **Structure Generation / DiffCSP** example.

### 1) Train
```bash
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
```
![](./pic6.png)

### 2) Sample
```bash
python structure_generation/sample.py --model_name='diffcsp_mp20' --weights_name='latest.pdparams' --save_path='result_diffcsp_mp20-1/' --chemical_formula='LiMnO2'
```
![](./pic5.png)

Result example: `Li1-Mn1-O2_1.cif`  
![](./pic7.png)
