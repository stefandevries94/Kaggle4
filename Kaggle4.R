# Libraries ---------------------------------------------------------------
library(tidyverse)
library(jpeg)
library(neuralnet)


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
}

nr = nc = 3
myFeatures  <- . %>% # starting with '.' defines the pipe to be a function 
  group_by(file, X=cut(x, nr, labels = FALSE)-1, Y=cut(y, nc, labels=FALSE)-1, color) %>% 
  summarise(
    m = mean(pixel_value),
    s = sd(pixel_value),
    min = min(pixel_value),
    max = max(pixel_value),
    q25 = quantile(pixel_value, .25),
    q75 = quantile(pixel_value, .75)
  ) 

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




# split the data to check how well it did
Sub <- Train[sample(nrow(Train)),]
Sub_train <- Sub[1:498,-1]
Sub_test <- Sub[499:664,-1]


# Models ------------------------------------------------------------------
nn_model = neuralnet(formula = category ~ ., data = Sub_train, hidden = 9)

# 
Predict = compute(nn_model,Sub_test)
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred

cbind(pred,Sub_test$category)
