**Exploratory Data Analysis (EDA): Correlation analysis of biological data during aging**

#### Objective:

I am developing a startup based on the use of a device capable of analyzing blood biomarkers with just a few drops of blood. After obtaining the blood analysis results, our goal is to use an algorithm to decipher the user's age range and provide an estimation of their health status.

To effectively build the algorithm, we need data that correlates various factors such as age, blood markers, disease history, among others. The main idea is to conduct a comprehensive analysis of all documents provided by the database and establish correlations between age, health status, blood biomarkers, and other factors that can be useful for the subsequent construction of a predictive algorithm.

#### Datasets:

The datasets has been compiled through the Health and Retirement Study database (https://hrsdata.isr.umich.edu) and (https://hrsdata.isr.umich.edu/data-products/special-access-downloads). This portal hosts a longitudinal study that has surveyed a representative sample of approximately 20,000 people in the United States. This study is supported by the National Institute on Aging (NIA U01AG009740) and the U.S. Social Security Administration. The survey includes detailed data, such as complete blood analysis, various biomarkers, studies on diabetes, medication history, among others.

To access the data, completion of a document called "Restricted Data Agreement" is required. Access keys (username and password) can be provided to professors if necessary; however, I have only shared some lists from the database due to ethical and privacy regulations applicable to the collection and management of patient health data.

I will be using survey data from 2016, 2018, and 2020, and will cross-reference it with blood biomarker analyses (biomarker_2012, biomarker_2014, and biomarker_2016), 2016 venous blood analysis (VBS/VBS_subs/VBS_2), 2016 blood cell analysis (cells_2016), and a cognitive study (cognitive_2016).

I have added a /data_description folder with information acquired from each of the datasets. The column title descriptions (now with code) have not been uploaded due to privacy regulations.

#### Folders:

- /data: contains all the data used and generated along the EDA.
- /memoria: contains a detailed summary of all the steps followed during the EDA.
- /notebooks: contains preprocessing, cleaning and analysis jupyter notebooks using for this EDA.
- /utils: contains
- Presentation in PDF with the results of the EDA project.