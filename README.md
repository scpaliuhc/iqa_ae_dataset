# iqa_ae_dataset

## Reference Images
We collect Ref images mainly from ImageNet and COCO. Detail requirements about the collection:
1. Clear
2. Rich texture
3. Resolution

## Adversarial Images
Target model:
1. xxx
2. xxx
3. ...

Methods:
1. Gradient-based: FGSM (0), MIM (1), PGD (2), NES (3)
2. Optimization-based: CW (4), AutoZOO (5), 
3. Discrete-noise-based: SimBA (6), SparseFool (7)(好像效率很低), Shouling (8),
4. NN-generation-based: AdvGAN (9), ATN (10)
5. Patch-based: Square (11), 
6. Universal-based: UAP (), GAP, FFF, GUAP
7. Spatially-transformed-based: xxx

*Fake label* $f(x) \neq f(x')$ or *real label* $f(x') \neq GT$?

Annotation:
* Ref Image (RI) ID
* Target Model (TM) ID (Table1)
* Method (ME) ID (Table2)
* Parameters (PA) ID (Table3)
* Fake label (FL) (or GT) ID
* Predicted (PR) ID (after attack)
* Objective Score (OS)

Examples:

```
{
    "RI":[{'TM':xx,
           'ME':xx,
           'PA':xx,
           'FL':xx,
           'PR':xx,
           'OS':xx},
          {'TM':xx,
           'ME':xx,
           'PA':xx,
           'FL':xx,
           'PR':xx,
           'OS':xx},
          ...
         ],
    "RI":[{'TM':xx,
           'ME':xx,
           'PA':xx,
           'FL':xx,
           'PR':xx,
           'OS':xx},
          {'TM':xx,
           'ME':xx,
           'PA':xx,
           'FL':xx,
           'PR':xx,
           'OS':xx},
          ...
         ],
    ...
}
```
