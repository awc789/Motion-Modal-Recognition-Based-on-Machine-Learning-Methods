# Motion-modal-recognition-based-on-machine-learning-methods
Southwest Jiaotong University Summer Project
（Undergraduate Graduation Project）

Tutor: Associate Prof Meng Hua

2019-10 ---– 2020-06


Data Collection
==================

## IMU Sensor
WitMotion website: [<https://www.wit-motion.com/>](https://www.wit-motion.com/)


**Base on IMU Sensor attached to the wrist and ankle**
- WitMotion Bluetooth 2.0 Mult-Connect BWT901CL **9 Axis IMU Sensor**
- WitMotion Bluetooth 2.0 Mult-Connect BWT901BCL **10 Axis IMU Sensor**

> Two sensors are attached to the wrist (9 axis) and the ankle (10 axis). And data Collection is acquired at frequencies greater than 50 Hz (200 Hz was used for this experiment)


Type of Motion
===

    ''' Bicycle  --------------------  1 ''' 
    ''' Downstairs  -----------------  2 ''' 
    ''' Heavy  ----------------------  3 '''
    ''' Jump  -----------------------  4 '''
    ''' Light  ----------------------  5 '''
    ''' Run  ------------------------  6 '''
    ''' Upstairs  -------------------  7 '''
    ''' Walk  -----------------------  8 '''

> **Heavy: Walking with weight in backpack**

> **Light: Walking without weight in backpack**


Methodology
===============

## pykalman are required, `$ easy_install pykalman`
pykalman: [<https://github.com/pykalman/pykalman>](https://github.com/pykalman/pykalman)

> **pykalman** depends on the following modules,

- numpy (for core functionality)
- scipy (for core functionality)
- Sphinx (for generating documentation)
- numpydoc (for generating documentation)
- nose (for running tests)

>All of these and **pykalman** can be installed using **easy_install**:

    $ easy_install numpy scipy Sphinx numpydoc nose pykalman



### 1. Feature Collection of Motion Data

> main .py

After setting up the path, this file automatically reads the data and detects whether the Kalman filter has been applied to the data and completes the identification and segmentation of the individual motion periods of the data.

The segmented data is then Fourier transformed to convert the time domain data into frequency domain data and feature engineering is carried out to obtain the characteristics of the data for identifying the type of motion.

The feature data will be save at
    
    './data/select_data_total/*'

> main .py will use the functions from:
    
    get_feature.py
    get_data.py
    wave_filter.py
    setting.py
    wave_detect.py

### 2. Individual Recognition of Motion

The following python file is used to classify the feature data obtained in the first part, so that the recognition of moving individuals can be completed.

>LDA_kf.py
    
    LDA（Linear Discriminant Analysis）

>SVM .py

    SVM（Support Vector Machines）

>PCA_kf.py

    PCA (Principal Component Analysis)

>LogisticRegression_RandomForest.py

    Logistic Regression
    Random Forest

>tf_LinerRegression.py

    Liner Regression

### 3. Recognition of Motion -- CNN

> train_cnn.py

The dimensionality of the data is further extended by fitting interpolate and differencing etc. This allows the data collected by the sensor to be extended from a $1 \times M$ data series to, $N \times M$ in the form of a data matrix.

The main part of the fit is based on a Legendre Polynomial Fit, which is shown in the figure below

    def f_fit(x, A, B, C, D, E, F, G, H, I):  # Using f_fit function to give the result of fitting
    ''' Legendre Polynomial Fit '''
    x0 = 1
    x1 = x
    x2 = (1/2) * (3 * x**2 - 1)
    x3 = (1/2) * (5 * x**3 - 3 * x)
    x4 = (1/8) * (35 * x**4 - 30 * x**2 + 3)
    x5 = (1/8) * (63 * x**5 - 70 * x**3 + 15 * x)
    x6 = (1/16) * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)
    x7 = (1/16) * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)
    x8 = (1/128) * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)
    return ((A * x8) + (B * x7) + (C * x6) + (D * x5) + (E * x4) + (F * x3) + (G * x2) + (H * x1) + (I * x0))

![fig_waves_6_1](https://i.imgur.com/gBnRxJP.png)

For more infomation can check the CurveFit_and_Interpolate.py file

> torch_cnn.py

This is the main part of the CNN model, which uses data in the form of a matrix generated from the previous file to simulate the image to train the model model and complete the classification of the motion.

![截图](https://i.imgur.com/4YtVf5h.png)


### 4. Motion Modal Prediction


#### RNN for Prediction
> rnn_prepare.py

Prepare the data format that suitable for the RNN model

> rnn_regression.py

Using RNN model with LSTM (Long-Short Time Memory) to predict the motion modal![Figure_6](https://i.imgur.com/7B2Nbj8.png)


#### Seasonal-ARIMA for Prediction

> cycle_prepare.py

Prepare the data format that suitable for the Seasonal-ARIMA

> cycle_predict.py

Using Seasonal-ARIMA for motion modal prediction

![预测](https://i.imgur.com/qcD1hIj.png)
