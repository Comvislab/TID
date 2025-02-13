# TID
Talent Identification Project

In sports, revealing athletes with high potential to excel in their endeavors for sports schools
holds a pivotal role. In the literature, this process is called Talent Identification (TID) and is defined as
"to know the players participating in the sport with the potential to be perfect". 

The problem discussed in our new paper focuses on the early identification of an athlete’s talented sports 
branch before they are assigned to a specific branch. This determination process is based on the 
evaluation of general performance tests and assessments.

To address the solution for the TID challenge, a two-stage TID solution has been introduced. 

TID1: The first stage (TID1), the admitted athletes are determined. TID1 uses our Shallow Deep
learning (SDL) model to classify the admitted. In this stage, a remarkable performance was obtained with
98.85%.

TID2: In the second stage (TID2), athletes are classified into their talented branches (Football, basketball, volleyball, or athletics).  
In TID2, nine different feature selection methods (four RFE-related methods, three SelectKBest-
related methods, and Lasso and Boruta) are applied to reduce the number of features. After feature selection,
our novel SCM-DL deep learning classifier model (apart from the architectures in literature, this model is
constructed internally with parallel layers and carries a combinatorial layer) is applied and compared with
Random Forest, Decision Tree, Extra Tree, and Support Vector Classifiers.
