#  Scratchpad for exploration and learning

## Looking into more details of feedforward 3DGS implementations

# setup

```
bash setup/project_setup.py

```

# notes

### compute  

For the default images, 1x A10G can process a 16 image jpg  
OOM on 60 image jpg  
Tensor size is [1, N_images, 3, 518, 518]  

### brainstorm project ideas

- finetune vggt-omega to generate 3DGS (non-commercial license)