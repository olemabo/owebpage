import json
from statistic.models import FPLPlayersModel


def read_data_from_json():
    with open('JSON_DATA/Element_Summary/Salah.json') as json_static:
        player_info = json.load(json_static)
    return player_info


def fill_player_database():
    player_info = read_data_from_json()
    fill_model = FPLPlayersModel()
    """
    print(df)
    number_of_teams = len(names)
    for i in range(number_of_teams):
        oppTeamNameList, oppTeamHomeAwayList, oppTeamDifficultyScore, gw = [], [], [], []
        fill_model = AddPlTeamsToDB(team_name=names[i], team_id=ids[i], team_short_name=short_names[i])
        team_info = df.loc[i]
        for j in range(38):
            gw_info_TEAM_HA_SCORE_GW = team_info.iloc[j + 1]
            oppTeamNameList.append(gw_info_TEAM_HA_SCORE_GW[0])
            oppTeamHomeAwayList.append(gw_info_TEAM_HA_SCORE_GW[1])
            oppTeamDifficultyScore.append(gw_info_TEAM_HA_SCORE_GW[2])
            gw.append(gw_info_TEAM_HA_SCORE_GW[3])
        fill_model = AddPlTeamsToDB(team_name=names[i], team_id=ids[i], team_short_name=short_names[i],
                                    oppTeamDifficultyScore=oppTeamDifficultyScore,
                                    oppTeamHomeAwayList=oppTeamHomeAwayList,
                                    oppTeamNameList=oppTeamNameList,
                                    gw=gw)
        fill_model.save()
    kickofftime_info = read_data.return_kickofftime()
    for gw_info in kickofftime_info:
        fill_model = KickOffTime(gameweek=gw_info[0], kickoff_time=gw_info[1], day_month=gw_info[2])
        fill_model.save()
    """
    return 0
