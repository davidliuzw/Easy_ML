# Easy_ML
This repo concludes my work in LCICM and aims to help people train machine learning models even without prior knowledge

## Function Parameters

### filepath
- **Type**: `string`
- **Description**: The absolute path of the CSV file that contains the paths for all feature tables. There should be one path per cell, with as many cells as desired in the first column of the file. Each table pointed to by the paths must contain `id_column` in it for proper joining. All data in the tables pointed to by paths should be numeric. Categorical variables should already be one-hot encoded.

### merge_base_path
- **Type**: `string`
- **Description**: The absolute path of the CSV file that contains the cohort of samples you want to build on. Must contain the `id_column` in it for merging. Only pulls the `id_column` from the file.

### id_column
- **Type**: `string`
- **Description**: The name of the column that uniquely identifies each sample. Should be present in all tables being combined from `filepath` and `merge_base_path`. Defaults to `subjectId`, which is the column used in CENTER-TBI for this purpose.

### drop_columns
- **Type**: `list`
- **Description**: A list that contains the column names that we want to drop from the merged data.

### rename_columns
- **Type**: `dict`
- **Format**: `{'OldColumnName': 'NewColumnName'}`
- **Description**: A dictionary specifying the columns to rename, where the key is the old column name and the value is the new column name.

### drop_id
- **Type**: `list`
- **Description**: A list that contains the `subjectIds` that we want to drop from the merged data.

### keep_id
- **Type**: `list`
- **Description**: A list that contains the `subjectIds` that we want to keep in the merged data. If provided, all other `subjectIds` will be dropped.

### export_name
- **Type**: `string`
- **Description**: The name of the feature CSV file to export, e.g., `test.csv`.

### export_path
- **Type**: `string`
- **Description**: The path where the exported CSV file will be saved. Path format example: `/home/idies/workspace/Storage/kgong1`.
