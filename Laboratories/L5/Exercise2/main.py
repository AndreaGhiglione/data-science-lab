import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

# 1
df = pd.read_csv('831394006_T_ONTIME.csv',parse_dates=[0],date_parser=lambda date: datetime.strptime(date,'%Y-%m-%d'))

# 2
df.info()
print(f'Total missing values: {df.isnull().sum(axis=0).sum()}')
print(f'Unique carriers: {len(df.groupby("UNIQUE_CARRIER").count())}')
unique_airports_origin = df.groupby('ORIGIN').count().index
unique_airports_dest = df.groupby('DEST').count().index
print(f'Unique airports: {unique_airports_origin.intersection(unique_airports_dest)}')
print(f'Data were collected since {min(df["FL_DATE"])}')

# 3
df = df[df['CANCELLATION_CODE'].isnull()]  # Filtering cancelled flights

# 4
print('\nFlights per carrier: ')
print(df.groupby('UNIQUE_CARRIER').count()['AIRLINE_ID'])

print('\nMean delay per carrier, considering all the reasons: ')
delays = ['DEP_DELAY','ARR_DELAY','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
print(round(df.groupby('UNIQUE_CARRIER')[delays].sum().mean(axis=1),2))

# 5
df['weekday'] = df['FL_DATE'].dt.dayofweek
df['delaydelta'] = df['ARR_DELAY']-df['DEP_DELAY']

# 6
df.groupby('weekday')['ARR_DELAY'].mean().plot(kind='bar',rot=0)
plt.show()

# 7
print('\nMean arrival delay per carrier during the weekend:')
mask = (df['weekday'] >= 5)
print(df[mask].groupby('UNIQUE_CARRIER')['ARR_DELAY'].mean())
print('\nMean arrival delay per carrier during the week:')
mask = (df['weekday'] != 5) & (df['weekday'] != 6)
print(df[mask].groupby('UNIQUE_CARRIER')['ARR_DELAY'].mean())

df_companies_delays_grouped = df.groupby('AIRLINE_ID')
del_only_week = []
del_only_weekend = []
for key,group in df_companies_delays_grouped:
    weekend_mask = (group['weekday'] >= 5)
    week_mask = (group['weekday'] != 5) & (group['weekday'] != 6)
    condition = ((group[weekend_mask]['ARR_DELAY'] <= 0).eq(True).all()) & ((group[week_mask]['ARR_DELAY'] > 0).eq(True).all())
    if condition:
        del_only_week.append(key)
    else:
        condition = ((group[week_mask]['ARR_DELAY'] <= 0).eq(True).all()) & ((group[weekend_mask]['ARR_DELAY'] > 0).eq(True).all())
        if condition:
            del_only_weekend.append(key)
print(f'Companies which are delayed only in week: {del_only_week}')
print(f'Companies which are delayed only in weekend: {del_only_weekend}')

# 8
df_multi_index = df.set_index(['UNIQUE_CARRIER','ORIGIN','DEST','FL_DATE'])

# 9
print(df_multi_index.loc[['AA','DL'],['LAX'],:,:][['DEP_TIME','DEP_DELAY']])

# 10
first_week_mask = df_multi_index.index.get_level_values('FL_DATE').isocalendar().week == 1
print(f'Mean arrival delay: {df_multi_index.loc[:,:,"LAX",first_week_mask]["ARR_DELAY"].mean()}')

# 11
piv_table = df.pivot_table('FL_DATE',index='weekday',columns='UNIQUE_CARRIER',aggfunc='count')  # all FL_DATE are not null
print('Pivot table with number of departed flights for each carrier and for each weekday')
print(piv_table)
pairwise_corr = piv_table.corr()
sns.heatmap(pairwise_corr,xticklabels=pairwise_corr.columns,yticklabels=pairwise_corr.columns)
plt.show()

# 12
piv_table = df.pivot_table('ARR_DELAY',index='weekday',columns='UNIQUE_CARRIER',aggfunc='mean')
print('Pivot table with average arrival delay for each carrier and for each weekday')
print(piv_table)
pairwise_corr = piv_table.corr()
sns.heatmap(pairwise_corr,xticklabels=pairwise_corr.columns,yticklabels=pairwise_corr.columns)
plt.show()

# 13
piv_table = df.pivot_table('delaydelta',index='weekday',columns='UNIQUE_CARRIER',aggfunc='mean')
piv_table = piv_table[['HA','DL','AA','AS']]
print(piv_table)
sns.set()  # Using Seaborn styles
piv_table.plot()
plt.ylabel('delay delta')  # Adding label to yaxis
plt.show()
