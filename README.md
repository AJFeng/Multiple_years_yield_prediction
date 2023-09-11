# Multiple_years_yield_prediction
Multiple years' yield prediction based on soil, weather and UAV image data.

This is the code related to our paper 'Prediction of cotton yield based on soil texture, weather conditions and UAV imagery using deep learning' authored by Aijing Feng, Jianfeng Zhou, Earl Vories and Kenneth Sudduth.

Cite this article
Feng, A., Zhou, J., Vories, E., & Sudduth, K. A. (2023). Prediction of cotton yield based on soil texture, weather conditions and UAV imagery using deep learning. Precision Agriculture, 1-24.

This study aimed to develop deep learning models for predicting cotton yield of different years using imagery data and environment (soil and weather conditions). The novelty of this study is to develop a method for predicting crop yield in future years based on the interaction of environment and management. Deep learning models were used to fuse data of different types and different resolutions. Yield prediction models based on deep learning were able to predict yield at good accuracy.

## Highlights:
- Developed methods to reduce the dimension and abstract key information of weather and soil.
- Developed deep learning models for fusing data of weather, soil and imagery.
- Explored the interaction effect of soil and weather on crop development.
- Developed deep learning models for predicting yield of future years.

## Models
![alt text](https://github.com/AJFeng/Multiple_years_yield_prediction/blob/main/figures/Picture1.png)
Fig. 3. Illustration of the convolutional neural network for soil (S_CNN). Soil clay content percentages from different depths were processed using a network with two convolution layers. Each coloured dot represents the result of a mathematical operation, ∑_(i=1)^2▒〖w_(i )×y_i 〗 where yi was the clay content percentage of each layer and the wi were the weights obtained by training from the data set provided. Filter: the input size of the kernels and used to detect spatial patterns. Stride: the number of steps that the filters move in the CNN.

![alt text](https://github.com/AJFeng/Multiple_years_yield_prediction/blob/main/figures/Picture2.png)
Fig. 4. Illustration of W_CNN. Weekly weather data from May 1 to October 29 were used. The first 13 weeks included dates from May 1 to July 30, week14-week18 included dates from July 31 to September 3, week19-week22 included dates from September 4 to October 1 and week 23 to week 26 included dates from October 2 to October 29. Weather feature abbreviations are: P- total precipitation (irrigation data included), Tmax- maximum air temperature, Tmin- minimum air temperature, SR- total solar radiation, VP- vapor pressure, and ETo- evapotranspiration from a reference crop. W1-W26 means week1-week26 after planting.

![alt text](https://github.com/AJFeng/Multiple_years_yield_prediction/blob/main/figures/Picture3.png)
Fig. 6. The architecture of the GRU network. SL is sequence length, which was set to 1 in this study. BZ is the batch size for the training procedure. FCL means a fully connected layer. IF is the image feature (i.e. NDVI). All the GRUs highlighted with yellow colour were the same loop processing unit in the network and had the same parameters. All FCL_1 units highlighted with green colour were using the same parameters.

## Results
![alt text](https://github.com/AJFeng/Multiple_years_yield_prediction/blob/main/figures/Picture4.png)
Fig. 9. Yield prediction results: (a) The data sets of 2019E and 2018W were used for training to predict the yield in 2017E; (b) The data sets of 2019E and 2017E were used for training to predict the yield in 2018W; (c) The data sets of 2018W and 2017E were used for training to predict the yield in 2019E; (d) The data sets of 2018W and 2017E were used for training to predict the yield in 2019W. Note, because no image NDVI was available in 2019W, the result in (d) is equal to the result of WO_TF_test.

![alt text](https://github.com/AJFeng/Multiple_years_yield_prediction/blob/main/figures/Picture5.png)
Fig. 10. Comparison of predicted and true yield of (a) 2017E and 2019E, and (b) 2018W and 2019W. The circles in all four maps in (a) and (b) mark the yield differences between years. The black circles mark those yield differences predicted well by the model, and the pink circles mark the yield differences of poor prediction. Legend is the same for all the yield maps. I: irrigation applied. S: regions with 55%-75% sand content.


