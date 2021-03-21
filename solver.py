import pandas as pd

car = pd.read_csv('./data/car.data',
                  names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

# cancer = pd.read_csv('./data/breast-cancer.data',
#                      names=['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])

# cancer = cancer[['age', 'menopause', 'tumor-size', 'inv-nodes',
#                  'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat', 'class']]

# cmc = pd.read_csv('./data/cmc.data',
#                   names=['age', 'education', 'husbandEducation', 'noChildren', 'religion', 'isWorking', 'husbandOccupation', 'livingStandard', 'mediaExposure', 'class'])

# data shuffling
car = car.sample(frac=1).reset_index(drop=True)
# cancer = cancer.sample(frac=1).reset_index(drop=True)
# cmc = cmc.sample(frac=1).reset_index(drop=True)

print(car)
# print(cancer)
# print(cmc)