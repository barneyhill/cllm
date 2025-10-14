# Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq

## Authors

- Dylan Kotliar<sup>1</sup> ([ORCID: 0000-0002-7968-645X](https://orcid.org/0000-0002-7968-645X)) †
- Adrian Veres<sup>1</sup>
- M Aurel Nagy<sup>2</sup>
- Shervin Tabrizi<sup>3</sup>
- Eran Hodis<sup>2</sup>
- Douglas A Melton<sup>4</sup> ([ORCID: 0000-0002-1623-5504](https://orcid.org/0000-0002-1623-5504))
- Pardis C Sabeti<sup>1</sup>

### Affiliations

1. Department of Systems Biology Harvard Medical School Boston United States
2. Harvard-MIT Division of Health Sciences and Technology Massachusetts Institute of Technology Cambridge United States
3. Viral Computational Genomics Broad Institute of MIT and Harvard Cambridge United States
4. Department of Stem Cell and Regenerative Biology, Harvard Stem Cell Institute Harvard University Cambridge United States

† Corresponding author

## Abstract

Identifying gene expression programs underlying both cell-type identity and cellular activities (e.g. life-cycle processes, responses to environmental cues) is crucial for understanding the organization of cells and tissues. Although single-cell RNA-Seq (scRNA-Seq) can quantify transcripts in individual cells, each cell's expression profile may be a mixture of both types of programs, making them difficult to disentangle. Here we benchmark and enhance the use of matrix factorization to solve this problem. We show with simulations that a method we call consensus non-negative matrix factorization (cNMF) accurately infers identity and activity programs, including their relative contributions in each cell. To illustrate the insights this approach enables, we apply it to published brain organoid and visual cortex scRNA-Seq datasets; cNMF refines cell types and identifies both expected (e.g. cell cycle and hypoxia) and novel activity programs, including programs that may underlie a neurosecretory phenotype and synaptogenesis.
