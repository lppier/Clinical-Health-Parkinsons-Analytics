# Multimodal Parkinson’s Disease Risk Assessment Application Based On Vocal and Improved Spiral Test

This analytics paper proposes a multi-modal approach combining voice and image test for early detection of Parkinson disease (PD). Research studies done earlier have used data related to either voice or spiral drawing to detect PD. However, different people experience different symptoms and different levels of severity of PD. Hence in this paper, we propose a multi-modal approach to enhance the reliability of identifying PD patient. Additionally, we propose to implement the multi-modal approach into a touch-enabled smartphone-based application to carry out preliminary PD tests conveniently, without the need of supervision of additional medical personnel or any specialized equipment. To substantiate our idea, we have evaluated both voice and spiral test data using various machine learning models. The results based on the two types of dataset demonstrate an excellent level of accuracy for PD identification.
Pairwise correlation and k-means clustering techniques are used to extract features from the vocal dataset. In this classification problem, the highest accuracy of 95.89% is obtained using an ensemble of 3 classification models.
The Pearson’s correlation is used to extract features from the image dataset. The best accuracy of 99.6% is achieved using the k-Nearest Neighbors classifier in the Dynamic Spiral Test (DST). An accuracy of 98.8% and 94.9% are achieved using the Logistic Regression classifier and the Adaptive Boosting classifier on the Static Spiral Test (SST) and Stability Test on Certain Point (STCP) respectively. A second ensemble making use of results from DST, SST, and STCP will provide the overall result of the spiral test.
The final ensemble for the application makes use of the results of the respective ensemble from the vocal and spiral test.

Paper Here :
https://github.com/lppier/Clinical-Health-Parkinsons-Analytics/blob/master/paper_Parkinson_v1.1.pdf

Datasets Used : 
https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings
https://archive.ics.uci.edu/ml/datasets/Parkinson+Disease+Spiral+Drawings+Using+Digitized+Graphics+Tablet

Report Here : https://github.com/lppier/Clinical_Health_Parkinsons_Analytics/blob/master/Group14-Multimodal%20Parkinson%20Risk%20Assessment%20Application%20Based%20On%20Vocal%20and%20Improved%20Spiral%20Test.pdf

## Group Members
* Chan Yi Jie Kelvin
* Gopa Sen
* Han Yuen Kwang Andy
* Lim Pier
* Teresa Cheng Siew Loon
