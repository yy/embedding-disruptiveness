# Data folder

## Symlink

Symlink data folder. e.g.,

```sh
ln -s /path/to/the/rawdata ./raw
```

## Organization

- `raw`: raw data files (e.g., citation network data or embedding files).
- `derived`: any data derived from the raw data (e.g., preprocessed data). organize further using folder structure.
- `additional`: any additional (external) data that is not derived from the raw data (e.g., metadata).
