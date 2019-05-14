a  = simulateHawkes(0.17454545, 0.05818182, 0.07757576, 40)
print(length(a[[1]]))
write.csv(a[[1]], file = "../practice/hsim.csv", row.names=FALSE)
likelihoodHawkes(0.5, 0.9, 2, a[[1]])
