

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("gdp_dataset/gdp_dataset.csv")

print(df.head())
print(df.columns[4:len(df.columns)-1])


x = df.columns[4:len(df.columns)-1]

# Find row of Aruba and Angola
angola_row = df.loc[df['Country Name'] == 'Angola']
aruba_row  = df.loc[df['Country Name'] == 'Aruba']

# Extract y-values
angola_gdp = angola_row.iloc[0, 4:]
aruba_gdp  = aruba_row.iloc[0, 4:]

# clean data so that we keep only years with actual values
angola_gdp_clean = []
aruba_gdp_clean  = []
x_clean          = []

for idx in range(len(x)):
    if pd.notna(angola_gdp[idx]) and pd.notna(aruba_gdp[idx]):
        x_clean.append(x[idx])
        angola_gdp_clean.append(angola_gdp[idx])
        aruba_gdp_clean.append(aruba_gdp[idx])

# plot data
plt.plot(x_clean, angola_gdp_clean, label="Angola")
plt.plot(x_clean, aruba_gdp_clean, label="Aruba")

plt.xlabel("Year")
plt.ylabel("GDP")
plt.title("GDP over the years of Angola vs Aruba")

plt.legend()
plt.show()



