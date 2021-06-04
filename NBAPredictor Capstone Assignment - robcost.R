################################################################################
# HarvardX Professional Certificate in Data Science (PH125.9x)
# Project: CYO NBA Predictor Capstone Assignment
# Author: Rob Costello, June 2021
#
################################################################################

################################################################################
# SECTION ONE
#
# Create nba set, validation set (final hold-out test set)
#
################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(RCurl)

#===============================================================================
# 
# Download and prepare data
#
# Original NBA Games dataset source:
# https://www.kaggle.com/nathanlauga/nba-games
#
#===============================================================================

# As Kaggle requires users to be registered to download datasets, the NBA dataset
# has been placed in a seperate location to ensure this analysis can be re-run 
# by anyone.

# Download data-sets
games <- read_csv(file = getURL("https://dgnxm5yoxqvfm.cloudfront.net/NBAData/games.csv"), skip = 0, col_names = TRUE)
games_details <- read_csv(file = getURL("https://dgnxm5yoxqvfm.cloudfront.net/NBAData/games_details.csv"), skip = 0, col_names = TRUE)
teams <- read_csv(file = getURL("https://dgnxm5yoxqvfm.cloudfront.net/NBAData/teams.csv"), skip = 0, col_names = TRUE)
ranking <- read_csv(file = getURL("https://dgnxm5yoxqvfm.cloudfront.net/NBAData/ranking.csv"), skip = 0, col_names = TRUE)
players <- read_csv(file = getURL("https://dgnxm5yoxqvfm.cloudfront.net/NBAData/players.csv"), skip = 0, col_names = TRUE)

#===============================================================================
# 
# Construct data-set for analysis
#
#===============================================================================
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# switch to dataframe
# adding _spread columns for various fields we will analyze.
games_df <- as.data.frame(games) %>% 
  mutate(PTS_spread = PTS_home - PTS_away,
         FG_PCT_spread = FG_PCT_home - FG_PCT_away,
         FT_PCT_spread = FT_PCT_home - FT_PCT_away,
         FG3_PCT_spread = FG3_PCT_home - FG3_PCT_away,
         AST_spread = AST_home - AST_away,
         REB_spread = REB_home - REB_away)

# check for NA's in data
apply(is.na(games_df), 2, which)

# In the 2011/12 season there was a lockout, so the season was shortened to 66 games.
# Ref: https://en.wikipedia.org/wiki/2011%E2%80%9312_NBA_season
#
# As there are NA's in the dataset we need to address them. In this case we will remove
# these rows as they are all NA's (not just specific fields).
games_df <- drop_na(games_df)

# Add game details so we have more data about each game
game_details_df <- left_join(games_df, games_details, by = "GAME_ID")

# partition for train/test/validate
test_index <- createDataPartition(y = game_details_df$PTS_spread, times = 1, p = 0.1, list = FALSE)
nba <- game_details_df[-test_index,]
temp <- game_details_df[test_index,]

nba_test_index <- createDataPartition(y = nba$PTS_spread, times = 1, p = 0.1, list = FALSE)
train_set <- nba[-nba_test_index,]
test_set <- nba[nba_test_index,]

validation <- temp %>% 
  semi_join(nba, by = "HOME_TEAM_ID") %>%
  semi_join(nba, by = "VISITOR_TEAM_ID")


# Add rows removed from validation set back into NBA dataset
removed <- anti_join(temp, validation)
nba <- rbind(nba, removed)

# remove unused objects to free up memory
rm(game_details_df, test_index, temp, removed)


################################################################################
# SECTION TWO
#
# Explore Data
#
################################################################################

#===============================================================================
# 
# Basic details from dataset
#
#===============================================================================
nrow(nba)
nrow(train_set)
nrow(test_set)
nrow(validation)

# how many seasons are covered?
length(unique(nba$SEASON))

# how many teams are there?
length(unique(nba$HOME_TEAM_ID))

# what was the avg points spread and SD?
mean(train_set$PTS_spread) # 
sd(train_set$PTS_spread) # 


#===============================================================================
# 
# Construct visualisations
#
#===============================================================================


# Consider home court advantage over time
train_set %>%
  ggplot(aes(x = GAME_DATE_EST, y = HOME_TEAM_WINS)) +
  geom_smooth()

# check the distribution of point spread across all games
train_set %>% ggplot(aes(x = PTS_spread)) +
  geom_histogram(bins = 20, alpha = 0.75, col = "black", fill = "blue")
# interesting to note this looks like a normal distribution

# check distribution of points score for home teams
train_set %>% ggplot(aes(x = PTS_home)) +
  geom_histogram(bins = 20, alpha = 0.75, col = "black", fill = "blue")

# check distribution of points score for away teams
train_set %>% ggplot(aes(x = PTS_away)) +
  geom_histogram(bins = 20, alpha = 0.75, col = "black", fill = "blue")
# interesting to note that away team distribution is slightly lower.

# lets check the means to see if this is accurate
mean(train_set$PTS_home)
sd(train_set$PTS_home)
mean(train_set$PTS_away)
sd(train_set$PTS_away)
# confirms home teams have an advantage, perhaps this is a bias we should consider in our models?

# show pts spread per game over the years
train_set %>%
  ggplot(aes(x = GAME_DATE_EST, y = PTS_spread)) +
  geom_smooth()


# show Field Goal % per game over the years
train_set %>%
  ggplot(aes(x = GAME_DATE_EST, y = FG_PCT_spread)) +
  geom_smooth()

# check for FG bias
mean(train_set$FG_PCT_home)
sd(train_set$FG_PCT_home)
mean(train_set$FG_PCT_away)
sd(train_set$FG_PCT_away)
# we see home teams shoot better FG percentage

# check for 3pt bias
mean(train_set$FG3_PCT_home)
sd(train_set$FG3_PCT_home)
mean(train_set$FG3_PCT_away)
sd(train_set$FG3_PCT_away)
# again we see home teams shoot better 3pt percentage


# check for FT bias
mean(train_set$FT_PCT_home)
sd(train_set$FT_PCT_home)
mean(train_set$FT_PCT_away)
sd(train_set$FT_PCT_away)
# home teams shoot better free throws but only just


# check for REB bias
mean(train_set$REB_home)
sd(train_set$REB_home)
mean(train_set$REB_away)
sd(train_set$REB_away)
# again, home teams rebound better

#===============================================================================
# 
# Consider Individual Performance
#
#===============================================================================

# Find games where a single player scored more than 40 points.
nba %>%
  filter(PTS >= 40) %>%
  ggplot(aes(x = GAME_DATE_EST, y = PTS, col = HOME_TEAM_WINS)) +
  theme(legend.position = "none") +
  geom_point() +
  scale_colour_gradient(low="red",high="green")

nba %>%
  filter(PTS >= 40) %>%
  summarise(win = sum(HOME_TEAM_WINS), loss = length(HOME_TEAM_WINS) - win)

# Find games where a single player hit greater than 50% for 3pt shots.
nba %>%
  filter(FG3_PCT != 1, FG3_PCT > 0.5) %>%
  ggplot(aes(x = GAME_DATE_EST, y = FG3_PCT, col = HOME_TEAM_WINS)) +
  theme(legend.position = "none") +
  geom_point() +
  scale_colour_gradient(low="red",high="green")

# Find games where a single player hit greater than 50% for Free Throw shots.
nba %>%
  filter(FT_PCT != 1, FT_PCT > 0.5) %>%
  ggplot(aes(x = GAME_DATE_EST, y = FT_PCT, col = HOME_TEAM_WINS)) +
  theme(legend.position = "none") +
  geom_point() +
  scale_colour_gradient(low="red",high="green")

# In each plot there appears to be more green for wins, we should validate the 
# impact of this effect in our models

################################################################################
# SECTION THREE
#
# Naive analysis
#
################################################################################

# Define RMSE algorithm to be used for evaluating prediction performance
RMSE <- function(true_spread, predicted_spread){
  sqrt(mean((true_spread - predicted_spread)^2))
}

# Determine naive RMSE by comparing mean spread with the training set
mu <- mean(train_set$PTS_spread)
naive_rmse <- RMSE(test_set$PTS_spread, mu) 

# Create tibble to store results and capture naive rmse
rmse_results <- tibble(method = "Naive RMSE", RMSE = naive_rmse)


################################################################################
# SECTION FOUR
#
# Regression on different factors to find out which are important, measured by RMSE
#
################################################################################

#===============================================================================
# 
# Consider Home Team (b_i) Advantage
#
#===============================================================================

game_avgs <- train_set %>% 
  group_by(HOME_TEAM_WINS) %>% 
  summarize(b_i = mean(PTS_spread - mu))

predicted_spread <- mu + test_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  pull(b_i)

home_effect_rmse <- RMSE(test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Team Effect RMSE", RMSE = home_effect_rmse ))


#===============================================================================
#
# Consider team Field Goal Percentage (b_fg) Effect
#
#===============================================================================

# Define helper function to replace NAs with mu.
na.mu <- function (x) {
  x[is.na(x)] <- mu
  return(x)
}

na.zero <- function (x) {
  x[is.na(x)] <- 0
  return(x)
}

fg_avgs <- train_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  group_by(FG_PCT_spread) %>%
  summarize(b_fg = mean(PTS_spread - mu - b_i))

predicted_spread <- test_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  mutate(pred = mu + b_i + b_fg) %>%
  pull(pred)

predicted_spread <- na.mu(predicted_spread)

home_fg_effect_rmse <- RMSE(test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Game & FG Effect RMSE", RMSE = home_fg_effect_rmse ))

#===============================================================================
#
# Consider team 3 Point Percentage (b_fg3) Effect
#
#===============================================================================

fg3_avgs <- train_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  group_by(FG3_PCT_spread) %>%
  summarize(b_fg3 = mean(PTS_spread - mu - b_i - b_fg))

predicted_spread <- test_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  mutate(pred = mu + b_i + b_fg + b_fg3) %>%
  pull(pred)

predicted_spread <- na.mu(predicted_spread)

home_fg3_effect_rmse <- RMSE(test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Game & FG & FG3 Effect RMSE", RMSE = home_fg3_effect_rmse ))

# indicates 3pt percentage differences across teams is not a good predictor. Perhaps there are outliers that skew this result, we should regularize to deal with this

#===============================================================================
#
# Consider team Free Throw percentage difference (b_ft) Effect
#
#===============================================================================

ft_avgs <- train_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  group_by(FT_PCT_spread) %>%
  summarize(b_ft = mean(PTS_spread - mu - b_i - b_fg - b_fg3))

predicted_spread <- test_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  mutate(pred = mu + b_i + b_fg + b_fg3 + b_ft) %>%
  pull(pred)

predicted_spread <- na.mu(predicted_spread)

home_ft_effect_rmse <- RMSE(test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Game & FG & FG3 & FT Effect RMSE", RMSE = home_ft_effect_rmse ))

# indicates free throw percentage differences across teams is not a good predictor. Perhaps there are outliers that skew this result, we should regularize to deal with this

#===============================================================================
#
# Consider Assist difference (b_ast) Effect
#
#===============================================================================

ast_avgs <- train_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  group_by(AST_spread) %>%
  summarize(b_ast = mean(PTS_spread - mu - b_i - b_fg - b_fg3 - b_ft))

predicted_spread <- test_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  left_join(ast_avgs, by='AST_spread') %>%
  mutate(pred = mu + b_i + b_fg + b_fg3 + b_ft + b_ast) %>%
  pull(pred)

predicted_spread <- na.mu(predicted_spread)

ast_effect_rmse <- RMSE(test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Game & FG & FG3 & FT & AST Effect RMSE", RMSE = ast_effect_rmse ))

# indicates Assist differences across teams is not a good predictor. Perhaps there are outliers that skew this result, we should regularize to deal with this

#===============================================================================
#
# Consider Assist difference (b_ast) Effect
#
#===============================================================================

reb_avgs <- train_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  left_join(ast_avgs, by='AST_spread') %>%
  group_by(REB_spread) %>%
  summarize(b_reb = mean(PTS_spread - mu - b_i - b_fg - b_fg3 - b_ft - b_ast))

predicted_spread <- test_set %>% 
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  left_join(ast_avgs, by='AST_spread') %>%
  left_join(reb_avgs, by='REB_spread') %>%
  mutate(pred = mu + b_i + b_fg + b_fg3 + b_ft + b_ast + b_reb) %>%
  pull(pred)

predicted_spread <- na.mu(predicted_spread)

reb_effect_rmse <- RMSE(test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Game & FG & FG3 & FT & AST & REB Effect RMSE", RMSE = reb_effect_rmse ))

# indicates Assist differences across teams is not a good predictor. Perhaps there are outliers that skew this result, we should regularize to deal with this

#===============================================================================
#
# Consider Individual Field Goal Percentage (b_ifg) Effect
#
# Here we consider the impact of a home team individual score more than 40 points
#
#===============================================================================

ifg_avgs <- train_set %>%
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  left_join(ast_avgs, by='AST_spread') %>%
  group_by(PTS) %>%
  filter(PTS >= 40, HOME_TEAM_WINS == 1) %>%
  summarize(b_ifg = mean(PTS_spread - mu - b_i - b_fg - b_fg3 - b_ft - b_ast))

predicted_spread <- test_set %>%
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  left_join(ast_avgs, by='AST_spread') %>%
  left_join(reb_avgs, by='REB_spread') %>%
  left_join(ifg_avgs, by='PTS') %>%
  filter(PTS >= 40, HOME_TEAM_WINS == 1) %>%
  mutate(pred = mu + b_i + b_fg + b_fg3 + b_ft + b_ast + b_reb + b_ifg) %>%
  pull(pred)

#predicted_spread <- na.zero(predicted_spread)

# filter the test set so we only compare appropriate values
f_test_set <- test_set %>%
  filter(PTS >= 40, HOME_TEAM_WINS == 1)

home_ifg_effect_rmse <- RMSE(f_test_set$PTS_spread, predicted_spread)

# Update results tibble with   Effect RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Home Game & FG & FG3 & FT & AST & REB & Player > 40pts Effect RMSE", RMSE = home_ifg_effect_rmse ))

rmse_results

################################################################################
# SECTION FIVE
#
# Use Regularization to account for outliers in various effects
#
################################################################################

#===============================================================================
#
# Regularize all modeled Effects
#
#===============================================================================

lambdas <- seq(0, 3, 0.1)

rmses <- sapply(lambdas, function(l){
  
  game_avgs <- train_set %>% 
    group_by(HOME_TEAM_WINS) %>%
    summarize(b_i = sum(PTS_spread - mu)/(n()+l))
  
  fg_avgs <- train_set %>% 
    left_join(game_avgs, by="HOME_TEAM_WINS") %>%
    group_by(FG_PCT_spread) %>%
    summarize(b_u = sum(PTS_spread - b_i - mu)/(n()+l))
  
  fg3_avgs <- train_set %>% 
    left_join(game_avgs, by='HOME_TEAM_WINS') %>%
    left_join(fg_avgs, by='FG_PCT_spread') %>%
    group_by(FG3_PCT_spread) %>%
    summarize(b_fg3 = sum(PTS_spread - b_i - b_u - mu)/(n()+l))
  
  ft_avgs <- train_set %>% 
    left_join(game_avgs, by='HOME_TEAM_WINS') %>%
    left_join(fg_avgs, by='FG_PCT_spread') %>%
    left_join(fg3_avgs, by='FG3_PCT_spread') %>%
    group_by(FT_PCT_spread) %>%
    summarize(b_ft = sum(PTS_spread - b_i - b_u - b_fg3 - mu)/(n()+l))
  
  ast_avgs <- train_set %>% 
    left_join(game_avgs, by='HOME_TEAM_WINS') %>%
    left_join(fg_avgs, by='FG_PCT_spread') %>%
    left_join(fg3_avgs, by='FG3_PCT_spread') %>%
    left_join(ft_avgs, by='FT_PCT_spread') %>%
    group_by(AST_spread) %>%
    summarize(b_ast = sum(PTS_spread - b_i - b_u - b_fg3 - b_ft - mu)/(n()+l))
  
  reb_avgs <- train_set %>% 
    left_join(game_avgs, by='HOME_TEAM_WINS') %>%
    left_join(fg_avgs, by='FG_PCT_spread') %>%
    left_join(fg3_avgs, by='FG3_PCT_spread') %>%
    left_join(ft_avgs, by='FT_PCT_spread') %>%
    left_join(ast_avgs, by='AST_spread') %>%
    group_by(REB_spread) %>%
    summarize(b_reb = sum(PTS_spread - b_i - b_u - b_fg3 - b_ft - b_ast - mu)/(n()+l))
  
  
  ifg_avgs <- train_set %>%
    left_join(game_avgs, by='HOME_TEAM_WINS') %>%
    left_join(fg_avgs, by='FG_PCT_spread') %>%
    left_join(fg3_avgs, by='FG3_PCT_spread') %>%
    left_join(ft_avgs, by='FT_PCT_spread') %>%
    left_join(ast_avgs, by='AST_spread') %>%
    left_join(reb_avgs, by='REB_spread') %>%
    group_by(PTS) %>%
    filter(PTS >= 40, HOME_TEAM_WINS == 1) %>%
    summarize(b_ifg = sum(PTS_spread - b_i - b_u - b_fg3 - b_ft - b_ast - b_reb - mu)/(n()+l))
  
  predicted_spread <- test_set %>%
    left_join(game_avgs, by='HOME_TEAM_WINS') %>%
    left_join(fg_avgs, by='FG_PCT_spread') %>%
    left_join(fg3_avgs, by='FG3_PCT_spread') %>%
    left_join(ft_avgs, by='FT_PCT_spread') %>%
    left_join(ast_avgs, by='AST_spread') %>%
    left_join(reb_avgs, by='REB_spread') %>%
    left_join(ifg_avgs, by='PTS') %>%
    filter(PTS >= 40, HOME_TEAM_WINS == 1) %>%
    mutate(pred = mu + b_i + b_u + b_fg3 + b_ft + b_ast + b_reb + b_ifg) %>%
    pull(pred)
  
  #predicted_spread <- na.zero(predicted_spread)
  
  # filter the test set so we only compare appropriate values
  f_test_set <- test_set %>%
    filter(PTS >= 40, HOME_TEAM_WINS == 1)
  
  # predicted_spread <- na.mu(predicted_spread)
  
  return(RMSE(f_test_set$PTS_spread, predicted_spread))
})

qplot(lambdas, rmses)  

final_lambda <- lambdas[which.min(rmses)]

reg_effect_rmse <- min(rmses)

# Update results tibble with Regularization RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Regularization of all Effect RMSE", RMSE = reg_effect_rmse ))

################################################################################
# SECTION SIX
#
# Use Simple Linear Model to evaluate effectiveness
#
################################################################################

fit <- lm(PTS_spread ~ HOME_TEAM_WINS + FG_PCT_spread, data=games_df)

y_hat <- fit$coef[1] + fit$coef[2]*test_set$FG_PCT_spread

lm_rmse <- sqrt(mean((y_hat - test_set$PTS_spread)^2))

rmse_results <- bind_rows(rmse_results,tibble(method ="Linear Model RMSE", RMSE = lm_rmse ))


################################################################################
# SECTION SEVEN
#
# Test other built-in functions to construct models and evaluate RMSE
#
################################################################################

# GLM
fit <- train(PTS_spread ~ HOME_TEAM_WINS + FG_PCT_spread, method = "glm", data = games_df)

y_hat <- fit$finalModel$coef[1] + fit$finalModel$coef[2]*test_set$FG_PCT_spread

glm_rmse <- sqrt(mean((y_hat - test_set$PTS_spread)^2))

rmse_results <- bind_rows(rmse_results,tibble(method ="GLM RMSE", RMSE = glm_rmse ))


# SVM
fit <- train(PTS_spread ~ HOME_TEAM_WINS + FG_PCT_spread, method = "svmLinear", data = games_df)

svm_rmse <- fit$results$RMSE

rmse_results <- bind_rows(rmse_results,tibble(method ="SVM Linear RMSE", RMSE = svm_rmse ))


# KNN
fit <- train(PTS_spread ~ HOME_TEAM_WINS + FG_PCT_spread, method = "knn", data = games_df)

knn_rmse <- min(fit$results$RMSE)

rmse_results <- bind_rows(rmse_results,tibble(method ="kNN RMSE", RMSE = knn_rmse ))

# gamLoess
fit <- train(PTS_spread ~ HOME_TEAM_WINS + FG_PCT_spread, method = "gamLoess", data = games_df)

gam_rmse <- fit$results$RMSE

rmse_results <- bind_rows(rmse_results,tibble(method ="gamLoess RMSE", RMSE = gam_rmse ))

# Random Forest
fit <- train(PTS_spread ~ HOME_TEAM_WINS + FG_PCT_spread, method = "rf", data = games_df)

rf_rmse <- fit$results$RMSE

rmse_results <- bind_rows(rmse_results,tibble(method ="Random Forest RMSE", RMSE = rf_rmse ))


################################################################################
# SECTION EIGHT
#
# Run the best model on the Validation set to get final RMSE
#
################################################################################

predicted_spread <- validation %>%
  left_join(game_avgs, by='HOME_TEAM_WINS') %>%
  left_join(fg_avgs, by='FG_PCT_spread') %>%
  left_join(fg3_avgs, by='FG3_PCT_spread') %>%
  left_join(ft_avgs, by='FT_PCT_spread') %>%
  left_join(ast_avgs, by='AST_spread') %>%
  left_join(reb_avgs, by='REB_spread') %>%
  mutate(pred = mu + b_i + b_fg + b_fg3 + b_ft + b_ast + b_reb + final_lambda) %>%
  pull(pred)

predicted_spread <- na.mu(predicted_spread)

final_validation_rmse <- RMSE(validation$PTS_spread, predicted_spread)

# Update results tibble with Final RMSE
rmse_results <- bind_rows(rmse_results,tibble(method ="Final Test on Validation Set RMSE", RMSE = final_validation_rmse ))

rmse_results
