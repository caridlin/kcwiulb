## First Step: Generate File Lists

The first step in the pipeline is to generate master file lists for each channel based on the directory structure.

Run:

```bash
python generate_master_filelists.py
```

This will create:

```text
master_filelist_blue.txt
master_filelist_red.txt
```

Each file contains the list of cube IDs grouped by field. For example:

```text
[offset2_a]
kb231211_00145
kb231211_00147
kb231211_00149

[offset3_a]
kb231211_00151
kb231211_00153

[offset2_b]
kb240208_00082
kb240208_00084

[offset3_b]
kb240208_00103
kb240208_00105
```

For the red channel, the structure is identical but uses `kr` cube IDs:

```text
[offset2_a]
kr231211_00145
kr231211_00147

[offset3_a]
kr231211_00151
kr231211_00153
```

These file lists define the input cubes and are used throughout the pipeline.