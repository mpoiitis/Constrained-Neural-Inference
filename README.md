# Constrained-Neural-Inference

## Constrained Structure Generation
**Dataset**: ZINC
**Train-Test-Val split**: 220011-5000-24445. Predefined split from pytorch geometric

**Graph labels y**: logP_SA_cycle_normalized
**Graph node features x**: atom type
**Graph edge features edge_attr**: bond type

**Batch size**:32 from GraphVAE paper
**Epochs**:25 from GraphVAE paper