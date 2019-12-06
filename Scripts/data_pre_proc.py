import pandas as pd
import auxiliary

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Change path according to directory used
dataset_path = '../Data/HR_DS.csv'
df = pd.read_csv(dataset_path)

# Remove non object variables with no variability
auxiliary.drop_var_nonobj(df)

# Remove object variables with no variability
auxiliary.drop_var_obj(df)

objects = df.describe(include='O').columns.tolist()
non_objs = df.describe().columns.tolist()

# Remove patterns
patterns = [' ', 'Travel_', '-', '&']
for p in patterns:
    df[objects] = df[objects].apply(lambda x: x.str.replace(p, ''))
    print('\tPattern "{}" cleared.'.format(p))

# Score that evaluates the number of years per companies
# The lower the score, the less stable/unexperienced the employee is
# The higher the score, the most stable/experienced the employee it
df['StayScore']=((df.TotalWorkingYears-df.YearsAtCompany)/(df.NumCompaniesWorked+1))

df_ = df.drop(columns=['JobLevel', 'EmployeeNumber', 'TotalWorkingYears', 'NumCompaniesWorked', 'YearsAtCompany'])
objects = df_.describe(include='O').columns.tolist()

# Making some dummies

finaldf = pd.get_dummies(df_, drop_first=True)

n_features = len(finaldf.columns)

print(
    '--------------------------------------------------------------------------\n',
    'Pre-processing concluded on ',
    dataset_path,
    'with success.\n',
    'Dataframes produced: finaldf\n',
    'Other instances created: n_features\n',
    '___________________________________________________________________________'
)
