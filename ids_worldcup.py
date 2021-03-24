import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\results.csv")
df = df[(df['tournament'] == 'FIFA World Cup')]

Country = "Germany"  # select a national football team that played in the World Cup


def total_wins(Country):
    home = df[df['home_team'] == Country]
    away = df[df['away_team'] == Country]

    home_wins = home[home['home_score'] > home['away_score']]
    away_wins = away[away['home_score'] < away['away_score']]

    return len(home_wins.index) + len(away_wins.index)


def total_loss(Country):
    home = df[df['home_team'] == Country]
    away = df[df['away_team'] == Country]

    home_loss = home[home['home_score'] < home['away_score']]
    away_loss = away[away['home_score'] > away['away_score']]

    return len(home_loss.index) + len(away_loss.index)


def total_draws(Country):
    home = df[df['home_team'] == Country]
    away = df[df['away_team'] == Country]

    home_draw = home[home['home_score'] == home['away_score']]
    away_draw = away[away['home_score'] == away['away_score']]

    return len(home_draw.index) + len(away_draw.index)


def total_games_played(Country):
    w = total_wins(Country)
    d = total_draws(Country)
    l = total_loss(Country)
    return w + d + l


def total_scored_goals(Country):
    home = df[df['home_team'] == Country]
    away = df[df['away_team'] == Country]

    home_scored_goals = home['home_score']
    away_scored_goals = away['away_score']

    return home_scored_goals.sum(axis=0) + away_scored_goals.sum(axis=0)


def total_conceded_goals(Country):
    home = df[df['home_team'] == Country]
    away = df[df['away_team'] == Country]

    home_conceded_goals = home['away_score']
    away_conceded_goals = away['home_score']

    return home_conceded_goals.sum(axis=0) + away_conceded_goals.sum(axis=0)


def avg_scored_conceded(Country):
    avg_conceded = total_conceded_goals(Country) / total_games_played(Country)
    avg_scored = total_scored_goals(Country) / total_games_played(Country)
    return avg_scored, avg_conceded


def consecutive_wins(s, Country):
    num = 0
    maxn = 0
    for c in s:
        if c == Country:
            num += 1
            maxn = max(num, maxn)
        else:
            num = 0
    return maxn


home = df[(df['home_team'] == Country)]
away = df[(df['away_team'] == Country)]

games_played = pd.concat([home, away]).sort_values('date')
con = [(games_played['home_score'] == games_played['away_score']),
       (games_played['home_score'] > games_played['away_score']),
       (games_played['home_score'] < games_played['away_score'])]
result = ['draw', games_played['home_team'], games_played['away_team']]
games_played['Result'] = np.select(con, result)

print(games_played)

print(Country)
print('Wins: ', total_wins(Country))
print('Draws: ', total_draws(Country))
print('Loss: ', total_loss(Country))
print('Total played games: ', total_games_played(Country))
print('Average Scored Goals, Average Conceded Goals', avg_scored_conceded(Country))
print('Longest win-streak: ', consecutive_wins(games_played['Result'], Country))

team_home = pd.DataFrame({'Date': home['date'],
                          'Scored': home['home_score'],
                          'Conceded': home['away_score']
                          })

team_away = pd.DataFrame({'Date': away['date'],
                          'Scored': away['away_score'],
                          'Conceded': away['home_score']
                          })

team = pd.concat([team_home, team_away])
team = team.sort_values('Date')

fig, ax = plt.subplots()
ax.plot(team['Date'], team['Scored'])
ax.plot(team['Date'], team['Conceded'])

plt.legend(["Scored", "Conceded"])
plt.title("Trend for played game ")
plt.xticks(rotation='vertical')
plt.tick_params(axis='x', which='major', labelsize=7)
#plt.savefig("result.png", bbox_inches='tight')
plt.show()

# creating a new df for every national teams's statistics
team_stats = pd.DataFrame(
    {'Team': pd.unique(df[["home_team", "away_team"]].values.ravel()),  # list of national teams's names
     })
# apply function on every team to get statistics
team_stats['Win'] = team_stats['Team'].map(lambda x: total_wins(x))  # epicissimo
team_stats['Draws'] = team_stats['Team'].map(lambda x: total_draws(x))
team_stats['Loss'] = team_stats['Team'].map(lambda x: total_loss(x))
team_stats['Avg_Scored'] = team_stats['Team'].map(lambda x: avg_scored_conceded(x)[0])  # epico
team_stats['Avg_Conceded'] = team_stats['Team'].map(lambda x: avg_scored_conceded(x)[1])

# sorting team to get rankings , (team at [0] is better than team at [>0] )
team_stats = team_stats.sort_values(['Win', 'Draws', 'Loss', 'Avg_Scored', 'Avg_Conceded'],
                                    ascending=[False, False, True, False, True], ignore_index=True)

# print(team_stats)

# get index of the selected team to compare it with the others
index = int(team_stats[team_stats['Team'] == Country].index.values)
if index == 0: # you've selected the best team... la Seleçao :)
    print(Country, 'is #1 team for statistics ')
    result = team_stats.iloc[0]
else:
    print('List of teams with better statistics: ')
    result = team_stats.iloc[0:index]
    print(result)


