

# Load vocabulary and training data
voc = read.table("myvocab.txt", stringsAsFactors = FALSE,header = TRUE)
trial_data <- read.table("train.tsv", stringsAsFactors = FALSE,header = TRUE)


vectorizer = vocab_vectorizer(create_vocabulary(voc[[1]], ngram = c(1L, 2L)))
it_trial = itoken(trial_data$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
dtm_trial = create_dtm(it_trial, vectorizer)


# Logistic Regression
newfit = cv.glmnet(x = dtm_trial, 
                   y = trial_data$sentiment, 
                   alpha = 0.1,
                   type.measure = 'auc',
                   family='binomial')

# Load test data
test = read.table("test.tsv",stringsAsFactors = FALSE, header = TRUE)

# Regex operation to match html tags in a non-greedy manner
test$review = gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
dtm_test  = create_dtm(it_test, vectorizer)

# Prediction
preds = predict(newfit, dtm_test, s=newfit$lambda.min, type='response')


#print results
mysubmission = cbind(test$id, preds)
colnames(mysubmission) = c('id','prob')
write.table(mysubmission,"mysubmission.txt",sep = " ",row.names=FALSE)


### Evaluation

#test.y = read.table("test_y.tsv", stringsAsFactors = FALSE, header = TRUE)
#glmnet:::auc(test.y$sentiment, preds)

