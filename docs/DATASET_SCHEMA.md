# Dataset Schema for Multimodal Crop-Stress Detection

## Manifest CSV Format

The system expects a CSV manifest file with paired image-sensor-label data:

```csv
image_id,image_path,crop,soil_moisture,temperature,humidity,rainfall,light_intensity,stress_label,source,timestamp
img_001,images/wheat_001.jpg,wheat,45.2,24.5,65.0,12.3,850,Healthy,real,2026-01-15T10:30:00Z
img_002,images/rice_002.jpg,rice,78.5,28.1,82.0,5.2,1200,Mild Stress,real,2026-01-15T11:00:00Z
img_003,images/maize_003.jpg,maize,22.1,35.2,45.0,0.0,1800,Severe Stress,synthetic,2026-01-15T12:00:00Z
```

### Required Columns

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `image_id` | string | Unique identifier | Required, unique |
| `image_path` | string | Relative path to image file | Required, file must exist |
| `crop` | string | Crop type | Required, from configured list |
| `soil_moisture` | float | Soil moisture percentage | 0-100 |
| `temperature` | float | Air temperature (°C) | -10 to 50 |
| `humidity` | float | Relative humidity (%) | 0-100 |
| `rainfall` | float | Rainfall (mm) | 0-500 |
| `light_intensity` | float | Light intensity (µmol/m²/s) | 0-2000 |
| `stress_label` | string | Stress category | Required, one of: Healthy, Mild Stress, Moderate Stress, Severe Stress |
| `source` | string | Data source | Optional: "real", "synthetic", "simulated" |
| `timestamp` | ISO datetime | Observation timestamp | Optional |

### Image Requirements

- **Formats**: JPG, JPEG, PNG, BMP, TIFF
- **Max size**: 10MB
- **Recommended resolution**: ≥224×224 pixels
- **Color space**: RGB (grayscale will be converted)

### Data Splits

The system creates three splits:
- **Train**: 70% (default)
- **Validation**: 15% (default)
- **Test**: 15% (default)

Stratified by `stress_label` to maintain class balance.

### Synthetic Data Handling

Synthetic/prototype data MUST be clearly labeled with `source: "synthetic"` or `source: "simulated"`. The system:
- Tracks synthetic vs real data separately
- Reports metrics for each source
- Never mixes them silently in evaluation
- Allows filtering by source

### Validation Checks

The validation pipeline checks:
1. **File existence**: All `image_path` files exist
2. **Image integrity**: Images are readable, not corrupted
3. **Format validation**: Supported image formats only
4. **Duplicate detection**: Duplicate `image_id` or identical images
5. **Sensor ranges**: All values within configured bounds
6. **Missing values**: No NaN in required columns
7. **Label validity**: All labels in configured classes
8. **Class balance**: Reports distribution, warns on severe imbalance
9. **Train/test leakage**: No overlapping `image_id` across splits
10. **Image-sensor pairing**: Each row has both image and sensor data