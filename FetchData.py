from get_nba_data import advanced_stats
import nba_py.game
import nba_py.team
import nba_py.constants

def get_raw_reg_season_data():
    teams = nba_py.constants.TEAMS.keys()
    recorded_games = []
    # [home_team, away_team, home_win]
    regular_season_raw_data = []
    for t in teams:
        print(t)
        team_id = nba_py.constants.TEAMS[t]['id']
        team_season_log = nba_py.team.TeamGameLogs(team_id, season='2015-16',
                                                   season_type="Regular Season").info().values
        for game in team_season_log:
            game_id = game[1]
            if game_id not in recorded_games:
                recorded_games.append(game_id)
                match_up = game[3]
                home_team = None
                away_team = None
                home_win = 'L'
                if match_up[4] == '@':
                    home_team = match_up[-3:]
                    away_team = t
                    if game[4] == 'L':
                        home_win = 'W'
                else:
                    home_team = t
                    away_team = match_up[-3:]
                    if game[4] == 'W':
                        home_win = 'W'
                regular_season_raw_data.append([home_team, away_team, home_win])
    return regular_season_raw_data


if __name__ == '__main__':
    a = get_raw_reg_season_data()
    f = open('2015-16_reg.txt', 'w')
    for data in a:
        f.write(data[0] + ' ' + data[1] + ' ' + data[2] + '\n')
    f.close()
