# Project-FLOP
Using ML to predict the nature of PCAP flows

FLOw Predictor - Project FLOP - can be used to predict how encrypted internet traffic was created by analysing the time related features captured in any PCAP file.

The current version is capable of predicting:

- Whether flows were created using OpenVPN or not.

Future versions will be able to predict:

- The class of the traffic (e.g. Web Browsing, Video Streaming, File Transfer, P2P etc)
- The application used to create the traffic.

The model was tested using data from the ISCX VPN / non-VPN dataset and using the application CIC-FLOW-METER, both of which were created by the University of New Brunswick (https://www.unb.ca/cic/datasets/index.html)

CIC-FLOW-METER has been added here, but can also be downloaded from the original source at https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter, although updated versions have not been tested. 

****************************************

How to use:

Training Dataset:

Frist chose a training dataset. The 'all data' CSV is provided as a pre-labelled option. Alternatively create a new training dataset using the CIC-FLOW-METER application, noting that you will have to manually label the data.

In the python code, enter the location of the training dataset CSV where indicated.


Testset:

Use the CIC-FLOW-METER application to create a flow features CSV of the PCAP you wish to analyise.

Enter the location of the CSV as the testset in the python code.

Run the code:

The code will generate a Random Forest model (previous research shows that RF provides the best balance of speed and performance for the data). Check the metrics to validate how well the model has performed on the training data.  

The code will make predictions of the testset and add the predictions to the final column of the flow DataFrame. You can now either view the Dataframe directly, copy and paste the predictions, or write the DataFrame to a new CSV depending on what you want to achieve. 

   
