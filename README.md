# README: Scripts and data usage

All scripts and publicly accessible data used in the paper by Brousse et al. (2024) are openly shared in this directory following open licences from data providers (UK Government, Google's Earth Engine or the European Union). Please contact Dr. Oscar Brousse (o.brousse@ucl.ac.uk) if there are any issues with the sharing of this data.

Some other data need to be granted access to be downloaded (MIDAS and Netatmo data in particular). Information on how to do this can be found in the two scripts used in this data analysis. Before treating the data, any user would have to collect the Netatmo weather stations' metadata using the Netatmo API. This can then be done using the _Collect_NetAtmo_Data_UK.py_ script. Once this step is achieved, users would have to download the MIDAS metadata from the CEDA archive website. Then, users can replicate the study by running the _Data_Treatment_and_Plots.py_ script. We used Python v3.9.15.
