import wget

#https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
car_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

#https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
breast_cancer_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data'

#https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
cmc_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'

#adult
adult_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

# wget.download(car_data_url)
# wget.download(breast_cancer_data_url)
# wget.download(cmc_data_url)
wget.download(adult_data_url)
