# Association mapping from sequencing reads using k-mers

## Authors

- Atif Rahman<sup>1</sup> ([ORCID: 0000-0003-1805-3971](https://orcid.org/0000-0003-1805-3971)) †
- Ingileif Hallgrímsdóttir<sup>2</sup>
- Michael Eisen<sup>3</sup> ([ORCID: 0000-0002-7528-738X](https://orcid.org/0000-0002-7528-738X))
- Lior Pachter<sup>1</sup> †

### Affiliations

1. Department of Electrical Engineering and Computer Sciences University of California, Berkeley Berkeley United States
2. Department of Statistics University of California, Berkeley Berkeley United States
3. Department of Molecular and Cell Biology Howard Hughes Medical Institute, University of California, Berkeley Berkeley United States

† Corresponding author

## Abstract

Genome wide association studies (GWAS) rely on microarrays, or more recently mapping of sequencing reads, to genotype individuals. The reliance on prior sequencing of a reference genome limits the scope of association studies, and also precludes mapping associations outside of the reference. We present an alignment free method for association studies of categorical phenotypes based on counting k -mers in whole-genome sequencing reads, testing for associations directly between k -mers and the trait of interest, and local assembly of the statistically significant k -mers to identify sequence differences. An analysis of the 1000 genomes data show that sequences identified by our method largely agree with results obtained using the standard approach. However, unlike standard GWAS, our method identifies associations with structural variations and sites not present in the reference genome. We also demonstrate that population stratification can be inferred from k -mers. Finally, application to an E.coli dataset on ampicillin resistance validates the approach.
