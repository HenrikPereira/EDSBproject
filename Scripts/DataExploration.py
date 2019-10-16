import pandas as pd
import seaborn as sb

data_path = r'Data\HR_DS.csv'
#root_path = r'C:\Users\henri\OneDrive\Documentos\EDSA\EDS Bootcamp\Project\'
data = pd.read_csv(r'C:\Users\henri\OneDrive\Documentos\EDSA\EDS Bootcamp\Project\Data\HR_DS.csv')

data.head()

sb.pairplot(data)

# no null values

