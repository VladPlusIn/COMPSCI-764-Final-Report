## Training

Run  `ctr_main.py`  for training

```
*/RLBid_EA/ctr$ python ctr_main.py --campaign_id=1458 --ctr_model=FM
```

Support models: `LR`, `FM`, `FFM`, `W&D`, `DeepFM`, `FNN`, `IPNN`, `OPNN`, `DCN`, `AFM`

Best model parameters are stored in `ctr/models/1458`, e.g. `FMbest.pth`
