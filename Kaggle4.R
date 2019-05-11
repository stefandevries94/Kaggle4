# Libraries ---------------------------------------------------------------
library(tidyverse)
library(jpeg)


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
Sunsets = map_df(sunsets, readJPEG_as_df) %>% 
  myFeatures() %>% 
  myImgDFReshape %>%
  mutate(category = "sunsets")
Trees = map_df(trees, readJPEG_as_df) %>%
  myFeatures() %>% 
  myImgDFReshape %>%
  mutate(category = "trees_and_forest")
Rivers = map_df(rivers[1:30], readJPEG_as_df) %>% 
  myFeatures() %>% 
  myImgDFReshape %>%
  mutate(category = "rivers")
Skies = map_df(skies[1:30], readJPEG_as_df) %>% 
  myFeatures() %>% 
  myImgDFReshape %>%
  mutate(category = "cloudy_sky")

Train = bind_rows(Sunsets, Trees, Rivers, Skies) 

