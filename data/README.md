# Data

The C-MAPSS data are not redistributed here. Download them from the official NASA
PHM archive and extract the nested `CMAPSSData.zip`:

```bash
python code/download_cmapss.py
```

This creates `data/CMAPSSData/` with the 12 train/test/RUL files. The archive URL is
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
(12,429,152 bytes). The dataset is simulated turbofan data released for the PHM08
challenge; cite Saxena et al. (2008).
