# Data instructions

The Paderborn bearing dataset used by the study is not bundled with this repository.

Obtain the dataset from its official distribution channel and store it locally. Do not commit the raw dataset to this repository unless its redistribution terms explicitly permit it.

The public scripts support a configurable data root through the environment variable:

```text
BEARING_DATA_ROOT
```

For example:

```bash
export BEARING_DATA_ROOT=/path/to/paderborn_data
```

or on PowerShell:

```powershell
$env:BEARING_DATA_ROOT = "D:\path\to\paderborn_data"
```

Expected file organization can vary with the downloaded Paderborn package. Follow the loader assumptions documented in the individual scripts and do not change the evaluated vibration channel: `Y[6]` / `vibration_1`.
