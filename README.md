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
1. Type 1:
2. Type 2:
3. Type 3:
4. Type 4:

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
