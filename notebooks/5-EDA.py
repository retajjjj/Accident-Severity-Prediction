import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

accidents = pd.read_pickle("../data/interim/cleaned_data.pkl")

accidents['Hour'] = accidents['Time'].str[0:2]
accidents['Hour'] = pd.to_numeric(accidents['Hour'])
accidents['Hour'] = accidents['Hour'].astype('int')

def when_was_it(hour):
    if hour >= 5 and hour < 10:
        return "morning rush (5-10)"
    elif hour >= 10 and hour < 15:
        return "office hours (10-15)"
    elif hour >= 15 and hour < 19:
        return "afternoon rush (15-19)"
    elif hour >= 19 and hour < 23:
        return "evening (19-23)"
    else:
        return "night (23-5)"
    
accidents['Daytime'] = accidents['Hour'].apply(when_was_it)
accidents[['Time', 'Hour', 'Daytime']].head(8)

sns.set_style('white')
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1
accidents.set_index('Date').resample('ME').size().plot(color='grey', ax=axes[0,0], label='Total per Month')
accidents.set_index('Date').resample('ME').size().rolling(window=10).mean().plot(color='darkorange', linewidth=4, ax=axes[0,0], label='10-MA')
axes[0,0].set_title('Accidents per Month', fontweight='bold')
axes[0,0].legend(frameon=False)

# Plot 2
yearly_count = accidents['Date'].dt.year.value_counts().sort_index(ascending=False)
axes[0,1].bar(yearly_count.index, yearly_count.values, color='lightsteelblue')
axes[0,1].plot(yearly_count, linestyle=':', color='black')
axes[0,1].set_title('Accidents per Year', fontweight='bold')

# Plot3
weekday_counts = accidents.set_index('Date').resample('1d')['Accident_Index'].size().reset_index()
weekday_counts.columns = ['Date', 'Count']
weekday_counts['Weekday'] = weekday_counts['Date'].dt.day_name()
weekday_averages = weekday_counts.groupby('Weekday')['Count'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
weekday_averages.plot(kind='barh', ax=axes[1,0], color='steelblue')
axes[1,0].set_title('Average Accidents per Weekday', fontweight='bold')

# Plot 4
severity = accidents.Accident_Severity.value_counts()
axes[1,1].pie(x=severity.values, labels=severity.index, colors=['silver', 'darkorange', 'red'], 
              autopct='%1.2f%%', pctdistance=0.8, textprops=dict(fontweight='bold'),
              wedgeprops={'linewidth':7, 'edgecolor':'white'})
axes[1,1].add_artist(plt.Circle((0,0), 0.6, color='white'))
axes[1,1].set_title('Accident Severity: Share in %', fontweight='bold')

# Plot 5
criteria = accidents['Accident_Severity']=='Fatal'
weekly_fatal = accidents.loc[criteria].set_index('Date').sort_index().resample('W').size()
weekly_fatal.plot(color='grey', ax=axes[2,0])
axes[2,0].fill_between(weekly_fatal.index, weekly_fatal.values, color='grey', alpha=0.3)
weekly_fatal.rolling(window=10).mean().plot(color='darkorange', linewidth=4, ax=axes[2,0])
axes[2,0].set_title('Fatalities (Weekly)', fontweight='bold')

# Plot 6
accidents.Hour.hist(bins=24, ax=axes[2,1], color='lightsteelblue', grid=False)
axes[2,1].set_title('Accidents by Time of Day', fontweight='bold')
axes[2,1].set_xlabel('Hour of the Day')

# Save and Show
plt.savefig('../reports/accident_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

