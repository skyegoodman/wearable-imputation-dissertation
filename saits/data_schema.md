# Data Schema

Protected Labfront/Garmin participant data are not included in this public version.
To run either pipeline, provide your own preprocessed wearable time-series file.

The default configs expect one row per timestamp with these columns:

| Column | Description |
| --- | --- |
| `timestamp` | Datetime column used as the time index. |
| `hr` | Heart rate. |
| `ibi` | Inter-beat interval. |
| `pulseOx` | Pulse oxygen saturation. |
| `steps` | Step count for the timestamp/bin. |
| `steps_rate` | Step rate or derived activity rate. |
| `device_stress` | Wearable/device stress metric. |
| `bodyBattery` | Wearable body battery metric. |
| `breathsPerMinute` | Respiration rate. |
| `sleep` | Encoded sleep state/stage. |

Missing values should remain blank/NaN. The realistic masking code builds empirical
missing-run libraries from the missingness pattern in the training split.

CSV files are supported directly. Parquet files are also supported if your Python
environment has a parquet engine such as `pyarrow` installed.
