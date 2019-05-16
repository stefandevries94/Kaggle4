# Libraries ---------------------------------------------------------------
library(tidyverse)
library(jpeg)
library(randomForest)
library(caret)
library(e1071)
library(pracma)



# Load in data ------------------------------------------------------------
skies = dir("cloudy_sky/", full.names = TRUE)
rivers = dir("rivers/", full.names = TRUE)
sunsets = dir("sunsets/", full.names = TRUE)
trees = dir("trees_and_forest/", full.names = TRUE)
test_set = dir("test_set/", full.names = TRUE)


# Functions ---------------------------------------------------------------
readJPEG_as_df <- function(path, featureExtractor = I) {
  img = readJPEG(path)
  d = dim(img)
  dimnames(img) = list(x = 1:d[1], y = 1:d[2], color = c('r', 'g', 'b'))
  df <- 
    as.table(img) %>% 
    as.data.frame(stringAsFactors = F) %>% 
    mutate(file = basename(path), x = as.numeric(x)-1, y = as.numeric(y)-1) %>%
    mutate(pixel_id = x + 28 * y) %>% 
    rename(pixel_value = Freq) %>%
    select(file, pixel_id, x, y, color, pixel_value)
  df %>%
    featureExtractor
  # euclidean distance between colors
  r = df$pixel_value[1:(length(df$pixel_value)/3)] # values of the red channel
  g = df$pixel_value[((length(df$pixel_value)/3)+1):(length(df$pixel_value)/1.5)] # values of the green channel
  b = df$pixel_value[((length(df$pixel_value)/1.5)+1):length(df$pixel_value)] # values of the blue channel
  vec_dist = sqrt(r^2 + g^2 + b^2) # compute the distance
  df$IMED = rep(vec_IMED,3) # put it into the data frame
  # extract features
  df %>%
    featureExtractor
}

nr = nc = 7
myFeatures  <- . %>% 
  group_by(file, X=cut(x, nr, labels = FALSE)-1, Y=cut(y, nc, labels=FALSE)-1, color) %>% 
  summarise(
    m = mean(pixel_value),
    s = sd(pixel_value),
    min = min(pixel_value),
    max = max(pixel_value),
    q25 = quantile(pixel_value, .25),
    q75 = quantile(pixel_value, .75),
    inner_product = dot(pixel_value, pixel_value),
    median = median(pixel_value),
    energy = sum(pixel_value^2) / length(pixel_value), 
    range = diff(range(pixel_value)),
    iqr1 = IQR(pixel_value), 
    m_IMED = mean(IMED),
    s_IMED = sd(IMED), 
    q25_IMED = quantile(IMED, .25), 
    q75_IMED = quantile(IMED, .75))


myImgDFReshape = . %>%
  gather(feature, value, -file, -X, -Y, -color) %>% 
  unite(feature, color, X, Y, feature) %>% 
  spread(feature, value)


# Create full data -------------------------------------------------------
# The method in the kernel gave me many NA's I assume this method was too fast
# First I specify the first row to get all correct column names and the right column length
Sunsets <- myImgDFReshape(myFeatures(map_df(sunsets[1], readJPEG_as_df)))
for (i in 1:length(sunsets)){
  Sunsets[i,] <- myImgDFReshape(myFeatures(map_df(sunsets[i], readJPEG_as_df)))
}
Sunsets$category = "sunsets"

Trees <- myImgDFReshape(myFeatures(map_df(trees[1], readJPEG_as_df)))
for (i in 1:length(trees)){
  Trees[i,] <- myImgDFReshape(myFeatures(map_df(trees[i], readJPEG_as_df)))
}
Trees$category = "trees_and_forest"

Rivers <- myImgDFReshape(myFeatures(map_df(rivers[1], readJPEG_as_df)))
for (i in 1:length(rivers)){
  Rivers[i,] <- myImgDFReshape(myFeatures(map_df(rivers[i], readJPEG_as_df)))
}
Rivers$category = "rivers"

Skies <- myImgDFReshape(myFeatures(map_df(skies[1], readJPEG_as_df)))
for (i in 1:length(skies)){
  Skies[i,] <- myImgDFReshape(myFeatures(map_df(skies[i], readJPEG_as_df)))
}
Skies$category = "cloudy_sky"
Train = bind_rows(Sunsets, Trees, Rivers, Skies) 



# Models ------------------------------------------------------------------
# Random forest
control = trainControl(method="repeatedcv", number=5, repeats=2)
metric <- "Accuracy"
rf_default <- train(category~. -file, data=Train, method="rf", metric=metric, 
                    trControl=control)

# mtry 69 was the best based on the cv
ranfor = randomForest(factor(category) ~ . - file, sub_train[1:550,], mtry = 69) 
predrf = predict(ranfor, sub_train[551:664,], type='class')


# SVM model
tune.out.radial=tune(svm, factor(category)~.-file, data=Train, kernel="radial", 
                     ranges=list(cost=c(0.1,1,10,100,1000)))
summary(tune.out.radial)

svmfit = svm(factor(category)~.-file, data=Train, kernel=("radial"),
             cost=10)

# Submission file ---------------------------------------------------------
Test <- myImgDFReshape(myFeatures(map_df(test_set[1], readJPEG_as_df)))
for (i in 1:length(test_set)){
  Test[i,] <- myImgDFReshape(myFeatures(map_df(test_set[i], readJPEG_as_df)))
}

Test %>% 
  ungroup %>% 
  transmute(file=file, category = predict(svmfit, ., type = "class")) %>% 
  write.csv(file = "predictions.csv", row.names = F)





# Neural Network ----------------------------------------------------------
## we tried to program a convolutional neural network
## the code works - but running it took more than 6 hours so we had to stop it and go for a svm ...


#loading keras library
library(keras)
# install_keras()

# Functions for the neural network------------------------------------------------------------------------------

# F1
readJPEG_as_df <- function(path, featureExtractor = I) {
  img = readJPEG(path)
  d = dim(img)
  dimnames(img) = list(x = 1:d[1], y = 1:d[2], color = c(1,2,3))
  df <-
    as.table(img) %>%
    as.data.frame(stringAsFactors = F) %>%
    mutate(file = basename(path), x = as.numeric(x), y = as.numeric(y)) %>%
    mutate(pixel_id = x + 28 * y) %>%
    rename(pixel_value = Freq) %>%
    select(file, pixel_id, x, y, color, pixel_value)
  df %>%
    featureExtractor
}

#F2
nr = nc = 32
myFeatures  <- . %>% # starting with '.' defines the pipe to be a function
  group_by(file, X=cut(x, nr, labels = FALSE), Y=cut(y, nc, labels=FALSE), color) %>%
  summarise(
    m = mean(pixel_value))

#create data
Sunsets = map_df(sunsets, readJPEG_as_df, featureExtractor = myFeatures) %>%
  mutate(category = 1)
Trees = map_df(trees, readJPEG_as_df, featureExtractor = myFeatures) %>%
  mutate(category = 2)
Rivers = map_df(rivers, readJPEG_as_df, featureExtractor = myFeatures) %>%
  mutate(category = 3)
Skies = map_df(skies, readJPEG_as_df, featureExtractor = myFeatures) %>% 
  mutate(category = 4)

Train = bind_rows(Sunsets, Trees, Rivers, Skies)

##### Create an 4D-Array for the neural network

# create empty arrays
Train_array <- array(
  data = 0,  
  dim = c(length(unique(Train$file)), 32,32, 3)
)

Target_array <- array(
  dim = c(length(unique(Train$file)))
)

# fill empty arrasy with the image data (each summarized in 32 x 32 pixels for each color channel)
previousFile = ""
cnt = 0
for (i in 1:nrow(Train)) {
  # checks if a new file is accessed
  if(!strcmp(previousFile, Train[i,]$file)){
    cnt = cnt + 1
    previousFile <-  Train[i,]$file
  }
  Train_array[cnt, (Train[i,]$X), (Train[i,]$Y), (Train[i,]$color)] <- Train[i,]$m
  Target_array[cnt] =  Train[i,]$category
}

# we found the following code in the internet and adjusted it for our purpose ----------------
# prepare data
train_y<-to_categorical(Target_array,num_classes = 4)
train_x<-Train_array

#TEST DATA
test_x<-Test_array
test_y<-to_categorical(TestTarget_array,num_classes=4) 

#checking the dimentions
dim(train_x)

#a linear stack of layers
model<-keras_model_sequential()

#configuring the Model

model %>%  
  #defining a 2-D convolution layer
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",                input_shape=c(32,32,3) ) %>%  
  layer_activation("relu") %>%  
  
  #another 2-D convolution layer
  
  layer_conv_2d(filter=32 ,kernel_size=c(3,3))  %>%  layer_activation("relu") %>%
  
  #Defining a Pooling layer which reduces the dimentions of the #features map and reduces the computational complexity of the model
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  
  #dropout layer to avoid overfitting
  
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter=32 , kernel_size=c(3,3),padding="same") %>% layer_activation("relu") %>%  layer_conv_2d(filter=32,kernel_size=c(3,3) ) %>%  layer_activation("relu") %>%  
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  
  #flatten the input  
  layer_flatten() %>%  
  
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  
  layer_dropout(0.5) %>%  
  
  #output layer-10 classes-10 units  
  layer_dense(4) %>%  
  
  #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy  
  
  layer_activation("softmax") 

#for computing Probabilities of classes-"logit(log probabilities)

#Model's Optimizer

#defining the type of optimizer-ADAM-Adaptive Momentum Estimation

opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )

#lr-learning rate , decay - learning rate decay over each update


model %>%
  compile(loss="categorical_crossentropy",
          optimizer=opt,metrics = "accuracy")

#Summary of the Model and its Architecture
summary(model)

#TRAINING PROCESS OF THE MODEL

data_augmentation <- FALSE  

if(!data_augmentation) {  
  model %>% fit(train_x,train_y ,batch_size=32,
                epochs=80,validation_data = list(test_x, test_y),
                shuffle=TRUE)
} else {  
  
  #Generating images
  
  gen_images <- image_data_generator(featurewise_center = TRUE,
                                     featurewise_std_normalization = TRUE,
                                     rotation_range = 20,
                                     width_shift_range = 0.30,
                                     height_shift_range = 0.30,
                                     horizontal_flip = TRUE  )
  
  #Fit image data generator internal statistics to some sample data
  gen_images %>% fit_image_data_generator(train_x)
  #Generates batches of augmented/normalized data from image data and #labels to visually see the generated images by the Model
  
  model %>% fit_generator(
    flow_images_from_data(train_x, train_y,gen_images,
                          batch_size=32,save_to_dir="CNNimages/"),
    steps_per_epoch=as.integer(50000/32),epochs = 80,
    validation_data = list(test_x, test_y) )
}

#use save_to_dir argument to specify the directory to save the #images generated by the Model and to visually check the Model's #output and ability to classify images.