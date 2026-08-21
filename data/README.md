# Data

This directory contains local data used by the diploma project.

## Structure

### raw/

Original source data.

Do not modify files in this directory.

Examples:

- original ImagesOASIS data;
- original clinical metadata;
- original exported files.

### processed/

Derived and transformed data created by project scripts.

Examples:

- filtered cohorts;
- merged metadata;
- preprocessed subject tables;
- derived feature tables.

Files in this directory may be regenerated from raw data.

### metadata/

Metadata and small reference tables used to connect dataset subjects with clinical information.

Examples:

- subject identifiers;
- Age;
- Sex;
- CDR;
- MMSE.

## Important rule

Raw data must never be overwritten during preprocessing.

The project should always preserve a clear distinction between:

raw source data

and

processed or derived data.