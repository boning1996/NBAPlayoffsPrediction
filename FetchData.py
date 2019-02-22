from get_nba_data import advanced_stats
import nba_py.game
import nba_py.team
import nba_py.constants
import time

def get_raw_reg_season_data():
    teams = nba_py.constants.TEAMS.keys()
    # [home_team, away_team, home_team_score, away_team_score]
    regular_season_data = {}
    for t in teams:
        # time.sleep(1)
        print(t)
        team_id = nba_py.constants.TEAMS[t]['id']
        team_season_log = nba_py.team.TeamGameLogs(team_id, season='2017-18',
                                                   season_type="Regular Season").info().values
        for game in team_season_log:
            game_id = game[1]
            if game_id not in regular_season_data:
                record = [None, None, None, None]
                regular_season_data[game_id] = [None, None, None, None]
                match_up = game[3]
                if match_up[4] == '@':
                    record[0] = match_up[-3:]
                    record[1] = t
                    record[3] = game[-1]
                else:
                    record[0] = t
                    record[1] = match_up[-3:]
                    record[2] = game[-1]
                regular_season_data[game_id] = record
            else:
                record = regular_season_data[game_id]
                if record[2] is None:
                    record[2] = game[-1]
                else:
                    record[3] = game[-1]
    return regular_season_data


if __name__ == '__main__':
    a = get_raw_reg_season_data()
    f = open('2017-18_reg.txt', 'w')
    for game in a:
        record = a[game]
        f.write(record[0] + ' ' + record[1] + ' ' + str(record[2]) + ' ' + str(record[3]) + '\n')
    f.close()
