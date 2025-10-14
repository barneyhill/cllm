# Peer review - Round 1

Editors:
- Alfonso Valencia, Barcelona Supercomputing Center - BSC Spain

Reviewers:
- Elisabetta Mereu, CRG Spain
- Berthold Göttgens, University of Cambridge United Kingdom

## Review text

DOI: [10.7554/eLife.43803.043](https://doi.org/10.7554/eLife.43803.043)

In the interests of transparency, eLife includes the editorial decision letter and accompanying author responses. A lightly edited version of the letter sent to the authors after peer review is shown, indicating the most substantive concerns; minor comments are not usually included.

Thank you for submitting your article "Identifying Gene Expression Programs of Cell-type Identity and Cellular Activity with Single-Cell RNA-Seq" for consideration by eLife. Your article has been reviewed by three peer reviewers, and the evaluation has been overseen by a Reviewing Editor and Naama Barkai as the Senior Editor. The following individuals involved in review of your submission have agreed to reveal their identity: Elisabetta Mereu (Reviewer #1); Berthold Göttgens (Reviewer #3).

The reviewers have discussed the reviews with one another and the Reviewing Editor has drafted this decision to help you prepare a revised submission.

Summary:

The non-negative matrix factorization method for the analysis of RNA-seq data presented in this paper, consensus-NMF (cNMF), addresses the identification cell types in the context of other non-cell type specific activities. and gene expression activity. A strength of the study is that results are not only compared with other methods, but also applied to both synthetic and real datasets.

Essential revisions:

A number of key elements of the algorithm are not described in sufficient detail to make it clear and reproducible, namely:

1) Clarify the comparison with alternative clustering approaches applied to the same problem, including the analysis of how it differs of other methods in the identification of genes that are part of cell activity programs, given the definition of cluster-specific marker is often done by a differentially expression analysis.

2) Provide an accurate description of the math of the method. Add equations to explain how to achieve the results and explain how parameters were selected.

2a) There is a large number of parameters used in simulation, data processing, running the method and methods comparison, many of which are not adequately explained or sufficiently justified (including but not limited to: uniform proportions of cell types in simulation, scaling non-log-transformed expression value in pre-processing, the choice of the number nearest neighbour for filtering outlier components, using 14 PCs for calculating cell distance for louvain clustering).

2b) The selection parameter K is non-trivial. It is a balance between stability and error. Currently, the choice of the number of factors, K, when applying the method to the simulated and the real datasets were not convincing. For example, k=13 seems more appropriate than k=14 for the simulated data judging from Figure 2—figure supplement 2A It's not clear which k should be chosen from Figure 3—figure supplement 1A. Showing how derived GEPs look under different choices of K would be essential. In addition, it is also unfair to compare with other unsupervised methods (such as PCA) if the K cannot be automatically selected. The authors can suggest an automatic solution to the K.

3) Pay more attention to performance according to the balance between cell types and samples.

3a) Because batch effects strongly affect RNAseq analysis, it is crucial the identification of a good set of highly variable genes before clustering for example or the application of a batch correction. How authors can be sure that one or more GEPs that come out from cNMF are not reflecting batch effects? How the method controls batch effects?

3b) There is not guarantee that one can find factors correlating with cell identity and activity in every dataset. It would be highly desirable to test the method in datasets with moderate batch effects and also in other tissues as currently tested real datasets are all in brain/neuronal tissues. It would be informative to show the variance explained by the overall model and each factor in both simulated and other real datasets preferably with a larger K to put the activity GEPs into perspective.
