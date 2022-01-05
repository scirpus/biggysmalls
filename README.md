# BiggySmalls

Numer.ai's large dataset takes about 16Gb of memory.  The aim of this repo is to download the parquet files and to run a linear transform to reduce the memory footprint to a more manageable size.

## Getting Started

These instructions will get you a copy of original files together with the reduced footprint files.

### Prerequisites

The things you need before installing the software.

* pip install requirement.txt

## Usage

Type python reducemem.py

```

The reduced files will all appear in the nmfdata directory tagged with "munged" alongside the originals for each round

```

## Additional

The reduction was performed using a linear transformation aided by a Kullback Leibler fitness metric to produce 128 parameters. 

