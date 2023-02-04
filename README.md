Practical Machine Learning project for Smartphone User Identification. 

The main goal of the competition was to develop a classifier that can group accelerometer signal recordings by user.

The dataset contains information from 20 users, each with 450 recordings. The signals are recorded for 1.5 seconds each, in the time in which a user taps the screen of his smartphone. The accelerometer records at 100 Hz and the values are in a 3D space, so they are expressed on $x, y, z$ axes.

For the project, I trained three machine learning models, in order to find the best performer for the classifying of smartphone users through accelerometer recordings. 

- The first approach I followed was with a Support Vector Machine (SVM) model. While training this classifier I tried different augmentations of the data, in order to improve the performance and the overall accuracy. The predictions made with the SVM model are used for my first submissions for the Kaggle competition, because they showed a good score. 
- My second approach was to train a Linear Discriminant Analysis. I used the same preprocessing techniques as for the SVM model, but the test scores where lower than the first method. Because of this, I didn't submit any predictions made with this model to the competition. 
- In search of a better performance, my third and last approach was using a Neural Network (NN) model for the task. With the NN model, I made many trials, using different parameters for the number of layers, epochs, batch sizes etc. I picked the best performers and submitted the predictions to the competition, with better results than the SVM model.


There are more details in the documentation
