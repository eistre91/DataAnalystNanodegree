import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import ttest_ind

titanic = pd.read_csv('titanic_data.csv')
colors = ['r', 'g']

#overall survival rate in the dataset

titanic.groupby('Embarked')['Embarked'].count()

#fares = titanic['Fare']

#for key, group in titanic.groupby('Survived'):
	#plt.scatter(group['Fare'], np.zeros_like(group['Fare']), c=colors[key])
#sns.lmplot('carat', 'price', data=df, hue='color', fit_reg=False)

#plt.show()

titanic.corr(method='pearson')

titanic.groupby('Pclass')['Survived'].mean()

titanic.groupby('Survived').mean()

grouped = titanic.groupby('Survived')

#highly statistically significant, p < .001
result_pclass = ttest_ind(grouped.get_group(0)['Pclass'], grouped.get_group(1)['Pclass'], equal_var=False)
result_fare = ttest_ind(grouped.get_group(0)['Fare'], grouped.get_group(1)['Fare'], equal_var=False, nan_policy='omit')

#statistically significant, p < .05
result_age = ttest_ind(grouped.get_group(0)['Age'], grouped.get_group(1)['Age'], equal_var=False, nan_policy='omit')
result_parch = ttest_ind(grouped.get_group(0)['Parch'], grouped.get_group(1)['Parch'], equal_var=False, nan_policy='omit')

#not statistically significant, p > .05
result_sibsp = ttest_ind(grouped.get_group(0)['SibSp'], grouped.get_group(1)['SibSp'], equal_var=False, nan_policy='omit')

titanic['last_name'] = titanic['Name'].apply(lambda x: x.split(',')[0])
titanic['last_name'].unique()
last_name_groups = titanic.groupby('last_name').count()[titanic.groupby('last_name')['last_name'].count() > 1]
last_name_groups = last_name_groups.index
last_name_groups = titanic[titanic['last_name'].isin(last_name_groups)]

((last_name_groups['SibSp'] == 0) & (last_name_groups['Parch'] == 0)).sum()
last_name_groups = last_name_groups[~((last_name_groups['SibSp'] == 0) & (last_name_groups['Parch'] == 0))]
last_name_groups['group'] = True

titanic[~titanic.index.isin(last_name_groups.index)]
titanic.loc[~titanic.index.isin(last_name_groups.index), 'group'] = False
titanic.loc[titanic.index.isin(last_name_groups.index), 'group'] = True

#survival rate of those in groups vs those not
titanic.loc[(titanic['Parch'] > 0) | (titanic['SibSp'] > 0), 'group'] = True
titanic.loc[~((titanic['Parch'] > 0) | (titanic['SibSp'] > 0)), 'group'] = False

titanic.groupby('group').mean()
 
#mean age of those who survived and were in a group is lower
titanic.groupby(['group', 'Survived']).mean()
 
titanic['group'] = titanic['group'].apply(lambda x: int(x))

#some correlation between survived and whether you were in a group or not, .2
titanic.corr()

#disappears on group size
titanic['group_size'] = titanic['SibSp'] + titanic['Parch']

#H-test on group-size and survival rate
#from scipy.stats import kruskal 
#I don't think h-test will work since it computes medians
from scipy.stats import f_oneway
titanic_restricted_group_size = titanic.loc[titanic['group_size'] < 4]
grouped = titanic_restricted_group_size.groupby('group_size')[['group_size', 'Survived']]
f_oneway(grouped.get_group(0)['Survived'], grouped.get_group(1)['Survived'], grouped.get_group(2)['Survived'], grouped.get_group(3)['Survived'])

from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
mod = MultiComparison(titanic['Survived'], titanic['group_size'])
print(mod.tukeyhsd())
mod.groupsunique
#group_size 0 is significantly different from sizes of 1, 2, 3
#but 1, 2, 3 are not pairwise different

#multicollinearity between group size and P class?
#.06 correlation between group size and P class

#what about gender?
titanic.groupby('Sex').mean()
grouped = titanic.groupby('Sex')
results_gender = ttest_ind(grouped.get_group('female')['Survived'], grouped.get_group('male')['Survived'], equal_var=False, nan_policy='omit')
#significant difference

#what about under 18? 16?
titanic.groupby(lambda x: "Under_18" if titanic['Age'].loc[x] < 18 else "Over_18").mean()
grouped = titanic.groupby(lambda x: "Under_18" if titanic['Age'].loc[x] < 18 else "Over_18")
results_children = ttest_ind(grouped.get_group('Under_18')['Survived'], grouped.get_group('Over_18')['Survived'], equal_var=False, nan_policy='omit')
#significant difference

#what about under 18 and by gender?
grouped = titanic.groupby(['Sex', lambda x: "Under_18" if titanic['Age'].loc[x] < 18 else "Over_18"])
grouped.mean()

def sex_age_pair(row):
	if row['Sex'] == 'male' and row['Age'] < 18:
		return "M-C"
	elif row['Sex'] == 'male' and row['Age'] >= 18:
		return "M-A"
	elif row['Sex'] == 'female' and row['Age'] < 18:
		return "F-C"
	else:
		return "F-A"
		
titanic['sex_age_pair'] = titanic.apply(lambda x: sex_age_pair(x), axis=1)
mod = MultiComparison(titanic['Survived'], titanic['sex_age_pair'])
print(mod.tukeyhsd())

#visualizations to use
#stacked bar graph

#how many women were alone on the boat
#looks like significantly fewer than men
#much higher mean of SibSp and Parch for women
#survival rate of those alone vs not
def sex_group_pair(row):
	if row['Sex'] == 'male' and row['group'] == True:
		return "M-G"
	elif row['Sex'] == 'male' and row['group'] == False:
		return "M-S"
	elif row['Sex'] == 'female' and row['group'] == True:
		return "F-G"
	else:
		return "F-S"
		
titanic['sex_group_pair'] = titanic.apply(lambda x: sex_group_pair(x), axis=1)
grouped = titanic.groupby('sex_group_pair')
f_oneway(grouped.get_group('M-G')['Survived'], grouped.get_group('M-S')['Survived'], grouped.get_group('F-G')['Survived'], grouped.get_group('F-S')['Survived'])
mod = MultiComparison(titanic['Survived'], titanic['sex_group_pair'])
print(mod.tukeyhsd())
#no difference between F-G F-S, but there was a difference between M-G M-S
#i.e. more grouped men survived

#does the pclass difference persist after sex and age stratification?
#what about the proportions of sex and age in each pclass, 
#does that explain differences?
(titanic.groupby(['Pclass', 'Sex']).count()/titanic.groupby('Pclass').count())['Survived']
grouped = titanic.groupby(['Pclass', lambda x: "Under_18" if titanic['Age'].loc[x] < 18 else "Over_18"])
grouped.mean()['Survived']

groups = ['-'.join(map(str, index)).strip() for index in gender_age_pclass_group.indices.keys()]
print(groups)

#        % Survived of Total  Total
#Sex                               
#female             0.742038    314
#male               0.188908    577

titanic_safe.groupby(['Sex','Under_18'])['Age'].count()

sr_male = .188908
sr_female = .742038
#what we would expect if under_18 had no influence
print( sr_male * (395/601) + sr_female * (206/601) )
print( sr_male * (58/113)  + sr_female * (55/113) )

titanic_safe.corr()

titanic_safe['Sex_number'] = titanic_safe['Sex'].astype('category').cat.codes
titanic_safe['Under_18_number'] = titanic_safe['Under_18'].astype('int')
import statsmodels.api as sm

logit = sm.Logit(titanic_safe['Survived'], titanic_safe['Under_18'].astype('int'))
result = logit.fit()
print(result.summary())

logit = sm.Logit(titanic_safe['Survived'], titanic_safe['Sex'].astype('category').cat.codes)
result = logit.fit()
print(result.summary())

logit = sm.Logit(titanic_safe['Survived'], titanic_safe[['Under_18_number','Sex_number']])
result = logit.fit()
print(result.summary())

titanic_safe['Survival'] = titanic_safe.Survived.map({0 : 'Died', 1 : 'Survived'})
titanic_safe.rename(columns={'sex_age_pair':'Sex and Age'})
t = titanic_safe.groupby(['Pclass', 'Sex and Age', 'Survival'], as_index = False)['Name'].count()
p = sns.factorplot(data = t, x = 'Pclass', y = 'Name', col = 'Survival', hue = 'Sex and Age', kind = 'bar')
silent = (p.set_axis_labels('', 'Passenger Count')
.set_xticklabels(['First Class', 'Second Class', 'Third Class'])
.set_titles("{col_name}"))

titanic_safe = titanic_safe.rename(columns={'sex_age_pair' : 'Sex and Age'})
t = titanic_safe.groupby(['Pclass', 'Sex and Age'], as_index = False)['Survived'].mean()
p = sns.barplot(data = t, x = 'Pclass', y = 'Survived', hue = 'Sex and Age')
silent = p.set(title = 'Survival by Passenger Class, Gender, and Age', 
        xlabel = 'Passnger Class', 
        ylabel = 'Survival Proportion', 
        xticklabels = ['1st Class', 'Second Class', 'Third Class'])

