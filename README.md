# IAU
Code implementation for paper ***Strike the Shepherd and the Sheep will Scatter: Lightweight Few-shot Machine Unlearning via Inverse Adversarial Example***
## Environment
First create a new anaconda environment with python=3.9.12. Install all the requirements in the requirements.txt file.
```bash
conda create --name IAU python=3.9.12
conda activate IAU
pip install -r requirements.txt
```

## Quick Start
Run the demo with the following configuration, to unlearn an AllCNN model trained with MNIST. It forget the class with index 1, and limits the $D_f$ data amount to 1 and  $D_r$ data amount to 1 for each class. 
```bash
python main.py -method IAU -forget_class 1 -model_name AllCNN -data_name mnist -dataset_dir ./data -train load -epoch 30 -lr 0.01 -num_forget 1 -num_remain 1
```