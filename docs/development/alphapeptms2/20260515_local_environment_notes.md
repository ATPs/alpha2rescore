# Local Environment Notes

This file is intentionally local and deployment-specific. It collects absolute
paths, hostnames, dataset locations, and lab-only commands that should not
appear in the public README files.

## Purpose

Use this note for:

- local editable-install commands tied to one server layout
- local benchmark datasets and output directories
- local PostgreSQL defaults
- links to older lab notes that contain server-specific commands

Do not copy these paths into `README.md` files intended for general users.

## Local Python Environments

- preferred local AlphaPept/DeepLC environment:
  - `/data/p/anaconda3/envs/alphabase/bin/python`
- previous local base Python used for PostgreSQL-only helper work:
  - `/data/p/anaconda3/bin/python`

## Local Source Trees

- bundled package repo:
  - `/data/p/ms2rescore/alpha2rescore`
- original standalone `alphapeptms2` source tree kept for reference:
  - `/data/p/xiaolong/xiaolongTools/XCLabServer/proteome/protcosmo/alphapeptdeep/alphapeptms2`

## Local Benchmark Inputs

- ProtInsight PIN parquet benchmark directory:
  - `/data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2pin.parquet`
- mzDuck parquet benchmark directory:
  - `/data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/mzDuck`
- raw mzML dataset mentioned in earlier notes:
  - `/data2/pub/proteome/PRIDE/mzML/2019/07/PXD010154`

## Local Outputs

- fast local output workspace:
  - `/XCLabServer002_fastIO/ms2rescore-test/`
- example previous output directory:
  - `/XCLabServer002_fastIO/ms2rescore-test/alpha2rescore-full-1554451-numba/`

## Local PostgreSQL Defaults

- host:
  - `10.110.120.2`
- port:
  - `5432`
- database:
  - `proteome`
- user:
  - `xlab`
- password file:
  - `/data/users/x/.ssh/20250505xcweb.server2.xlab.postgresql.passwd`
- schema:
  - `protein_hs`

## Related Local Notes

- implementation note:
  - `/data/p/ms2rescore/ms2rescore-test/notes/20260513_alpha2rescore_v1_implementation.md`
- bundled alphapeptms2 design note:
  - `20260511design.md`
- bundled alphapeptms2 benchmark fix note:
  - `20260511fix1.md`
