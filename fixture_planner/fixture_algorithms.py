import json
import numpy as np
import pandas as pd
import math
from itertools import combinations


class Team:
    """
        Class of a team
    """

    def __init__(self, team_id, players_df, fixtures_df):
        """
            Initialize object
        :param team_number: Based on alphabetical order -> Arsenal = 1, ..., Wolves = 20
        :param players_df: DataFrame containing information about the players on the team
        :param fixtures_df: DataFrame providing information about gameweeks, opponents, difficulty and H/A.
        """
        self.team_id = team_id
        self.team = team_id
        self.players_df = players_df
        self.fixtures_df = fixtures_df


class Fixtures:
    """
        Class of Fixtures
    """

    def __init__(self, fixtures_information):
        """
            Initialize object
        :param fixtures_information: list of dictionaries with fixture information
        """
        df = pd.DataFrame(fixtures_information)
        features = ['event', 'team_h', 'team_h_difficulty', 'team_a', 'team_a_difficulty']
        final_features = ['gameweek', 'opponent_team', 'difficulty', 'H/A']
        new_name = {'event': 'gameweek'}
        df = df[features]
        df = df.rename(columns=new_name)
        self.list_teams = []
        # loop through each team and get fixture information
        for i in range(df['team_h'].min(), df['team_h'].max() + 1):
            temp = pd.DataFrame()
            home_matches = df[df['team_h'] == i].copy()
            home_matches.loc[:, 'difficulty'] = home_matches['team_h_difficulty']
            home_matches.loc[:, 'opponent_team'] = home_matches['team_a']
            home_matches.loc[:, 'H/A'] = ['H'] * 19
            away_matches = df[df['team_a'] == i].copy()
            away_matches.loc[:, 'difficulty'] = away_matches['team_a_difficulty']
            away_matches.loc[:, 'opponent_team'] = away_matches['team_h']
            away_matches.loc[:, 'H/A'] = ['A'] * 19
            # this is for postponed matches. Set postponed match difficulty to 0
            for j in range(len(home_matches)):
                home, away = home_matches.iloc[j]['gameweek'], away_matches.iloc[j]['gameweek']
                if math.isnan(home):
                    home_matches.at[home_matches.iloc[j].name, ['gameweek', 'difficulty']] = min(
                        home_matches.iloc[j + 1]['gameweek'], away) - 1, 0

                if math.isnan(away):
                    away_matches.at[away_matches.iloc[j].name, ['gameweek', 'difficulty']] = min(
                        away_matches.iloc[j + 1]['gameweek'], home) - 1, 0

            temp = temp.append(home_matches[final_features])
            temp = temp.append(away_matches[final_features])
            temp = temp.sort_values('gameweek', ascending=True)
            self.list_teams.append(temp)


class Players:
    """
        Class of FPL players.
    """

    def __init__(self, information_players, fixtures_information):
        """
            Initialize object
        :param information_players: list of dictionaries with information about players
        :param fixtures_information: list of dictionaries with fixture information
        """
        df = pd.DataFrame(information_players)
        float_columns = ['form', 'points_per_game', 'selected_by_percent']  # ensure float from strings
        df[float_columns] = df[float_columns].astype(float)
        length_news = df['news'].str.len()
        df.loc[:, 'fitness'] = length_news.values == 0  # create boolean for fitness/available players
        new_names = {'web_name': 'name', 'now_cost': 'price', 'selected_by_percent': 'selection',
                     'element_type': 'position'}  # new and better names for columns
        df = df.rename(columns=new_names)
        # decide attributes to keep
        attributes_to_use = ['name', 'price', 'team', 'total_points', 'points_per_game', 'minutes', 'form',
                             'clean_sheets', 'assists', 'goals_scored', 'selection', 'position', 'fitness', 'id']
        self.df = df[attributes_to_use]
        self.fixtures = Fixtures(fixtures_information)

    def get_attribute_from_all_players(self, attribute):
        if attribute not in self.df.keys():
            print(f'Attribute {attribute} does not exist for a player!')
            exit(1)
        return self.df[attribute]

    def create_all_teams(self):
        list_of_teams_df = [x for _, x in self.df.groupby('team')]
        team_list = []
        for i in range(len(list_of_teams_df)):
            team = Team(list_of_teams_df[i]['team'].values[0], list_of_teams_df[i], self.fixtures.list_teams[i])
            team_list.append(team)
        return team_list

    def create_squad(self, list_of_player_names, list_of_player_teams):
        if len(list_of_player_names) != 15:
            print('Not enough players..')
            exit(1)
        df = pd.DataFrame()
        for i in range(len(list_of_player_names)):
            player_name, team_number = list_of_player_names[i], list_of_player_teams[i]
            player_df = self.df.loc[(self.df['name'] == player_name) & (self.df['team'] == team_number)].copy()
            df = df.append(player_df)
        return df


def get_json():
    with open('JSON_DATA/static.json') as json_static:
        static = json.load(json_static)
    with open('JSON_DATA/fixture.json') as json_fixture:
        fixture = json.load(json_fixture)
    return static, fixture


def create_dict_with_team_ids_to_team_name(fixtures, name_or_short_name="name"):
    # fpl_info = pd.DataFrame(DataFetch().get_current_fpl_info()['teams'])
    fpl_info = pd.DataFrame(fixtures)
    return dict(zip(fpl_info['id'], fpl_info[name_or_short_name]))

def create_dict_with_team_name_to_team_ids(fixtures, name_or_short_name="name"):
    # fpl_info = pd.DataFrame(DataFetch().get_current_fpl_info()['teams'])
    fpl_info = pd.DataFrame(fixtures)
    return dict(zip(fpl_info[name_or_short_name], fpl_info['id']))

def create_team_list():
    static, fixture = get_json()
    # a = DataFetch()
    # b = a.get_current_fpl_info()
    b = static
    player_info = b['elements']
    # fixture_info = a.get_current_fixtures()
    fixture_info = fixture
    players = Players(player_info, fixture_info)
    team_list = players.create_all_teams()  # list teams, where Arsenal = team_list[0], ... Wolves = team_list[-1]
    dict_team_id_to_team_name = create_dict_with_team_ids_to_team_name(static['teams'], "name")
    for team_id in range(len(team_list)):
        team_list[team_id].team = dict_team_id_to_team_name[team_id + 1]
    # the list index is also the same as the team id minus 1. Arsenal = id=1 - 1, ...
    return team_list


def create_data_frame():
    """
    create dataframe with fixture information.
    Rows: Team_id - 1
    Columns: Team (team name) 1 (GW number) 2 3 4 5
    :return:             Team            1            2   ...           38
            0          Arsenal  [FUL, A, 2]  [WHU, H, 2]  ...  [BHA, H, 2]
            1      Aston Villa  [MCI, A, 0]  [SHU, H, 3]  ...  [CHE, H, 3]

    """
    static, fixture = get_json()

    team_list = create_team_list()
    number_of_PL_teams = len(team_list)
    columns = [str(i) for i in range(0, len(team_list[0].fixtures_df.index) + 1)]
    columns[0] = 'Team'
    data = []
    temp_team = []
    dict_team_id_to_name = create_dict_with_team_ids_to_team_name(static['teams'], name_or_short_name="short_name")
    for team in range(number_of_PL_teams):
        temp_team.append(team_list[team].team)
    for gameweek in range(number_of_PL_teams):
        team_info = team_list[gameweek]
        temp_data = [team_info.team]
        for team_idx in range(len(columns) - 1):
            index_opp = team_info.fixtures_df['opponent_team'].index[team_idx]
            index_ah = team_info.fixtures_df['H/A'].index[team_idx]
            index_diff = team_info.fixtures_df['difficulty'].index[team_idx]
            opp = team_info.fixtures_df['opponent_team'][index_opp]
            ah = team_info.fixtures_df['H/A'][index_ah]
            diff = team_info.fixtures_df['difficulty'][index_diff]

            temp_data.append([dict_team_id_to_name[opp], ah, diff])
        data.append(temp_data)
    return pd.DataFrame(data=data, columns=columns)


def return_fixture_names_shortnames():
    df = create_data_frame()
    static, fixture = get_json()
    names = pd.DataFrame(static['teams'])['name']
    short_names = pd.DataFrame(static['teams'])['short_name']
    ids = pd.DataFrame(static['teams'])['id']
    return df, names, short_names, ids


def fixture_score_one_team(df, team_idx, GW_start, GW_end):
    """
    return fixture score for one team with team_id = team_idx from GW_start to GW_end.
    :param df: dataframe with fixture info (create_data_frame())
    :param team_idx: team_id + 1
    :param GW_start: first GW
    :param GW_end: last GW
    :return: np.array with length 8. [score, team name, opponents, fixture difficulty, GW_start, GW_end, score / number_of_fixtures]
    [31 'Arsenal' array(['ful', 'WHU', 'liv', 'SHU', 'mci', 'LEI', 'mun', 'AVL', 'lee', 'WOL'], dtype=object)
    array([2, 2, 5, 3, 5, 3, 4, 2, 2, 3]) 1 10 3.1]
    """
    score = 0
    team_idx = team_idx - 1
    team = df.loc[team_idx][0]
    number_of_fixtures = GW_end - GW_start + 1
    upcoming_fixtures = np.empty(number_of_fixtures, dtype=object)
    home_away = np.empty(number_of_fixtures, dtype=int)
    upcoming_fixtures_score = np.empty(number_of_fixtures, dtype=float)
    blanks = []
    if GW_start < 1:
        print("GW_start must be larger than 0")
        return -1
    if GW_end > 38:
        print("GW_end must be smaller than 38")
        return -1
    for i in range(GW_start - 1, GW_end):
        add_score = float(df.loc[team_idx][1:][i][2])
        if add_score == 0:
            number_of_fixtures -= 1
            blanks.append(i + 1)
        score += add_score
        if df.loc[team_idx][1:][i][1] == 'A':
            upcoming_fixtures[i - GW_start + 1] = df.loc[team_idx][1:][i][0].lower()
            home_away[i - GW_start + 1] = 0
        if df.loc[team_idx][1:][i][1] == 'H':
            upcoming_fixtures[i - GW_start + 1] = df.loc[team_idx][1:][i][0].upper()
            home_away[i - GW_start + 1] = 1
        upcoming_fixtures_score[i - GW_start + 1] = float(df.loc[team_idx][1:][i][2])
    return np.array([round(score, 2), team, upcoming_fixtures, upcoming_fixtures_score, GW_start, GW_end, blanks, round(score / number_of_fixtures, 3), home_away], dtype=object)

def create_list_with_team_ids_from_list_with_team_names(f, team_names_list):
    team_id_list = []
    dict = create_dict_with_team_name_to_team_ids(f, name_or_short_name="name")
    for team_name in team_names_list:
        team_id_list.append(dict[team_name])
    return team_id_list

def insertion_sort(A, size, element_to_sort, min_max="min"):
    """
    Sort a 2D list with respect to one of the element_idx
    :param A:
    :param size:
    :param element_to_sort: 0 / -1 (score / score divided by fixtures
    :return:
    """
    # sort an array according to its fixture score
    if min_max == "min":
        for i in range(size - 1):
            score = A[i + 1][element_to_sort]
            team_array = A[i + 1]
            while i >= 0 and A[i][element_to_sort] > score:
                A[i + 1] = A[i]
                i = i - 1
            A[i + 1] = team_array
    else:
        for i in range(size - 1):
            score = A[i + 1][element_to_sort]
            team_array = A[i + 1]
            while i >= 0 and A[i][element_to_sort] < score:
                A[i + 1] = A[i]
                i = i - 1
            A[i + 1] = team_array
    return A

def find_best_rotation_combos(GW_start, GW_end, teams_to_check=5, teams_to_play=3, team_names=[-1],
                              teams_in_solution=[], teams_not_in_solution=[],
                              top_teams_adjustment=False, one_double_up=False, home_away_adjustment=True,
                              include_extra_good_games=False, num_to_print=20):
    """
    Find the best rotation combo for "teams_to_check" number of team where "team_to_play" number of them must play each gameweek.
    :param GW_start: start count from this gameweek
    :param GW_end: end counting in this gameweek
    :param teams_to_check: how many teams to check (1-5)
    :param teams_to_play: how many of the teams_to_check you want to play in each gameweek. ( team_to_play < teams_to_check)
    teams_to_check = 5 and teams_to_play = 3 will give the best 5 teams to use in your team if you must use at least 3 each round
    :param team_names: which teams to check from. ([-1]: all teams, ["Arsenal", "Spurs"]: check only some teams)
    :param teams_in_solution: ["Liverpool"]: liverpool must be in the optimal solution. []: no extra dependencies.
    :param home_away_adjustment: wheter to give home/away match score -/+ 0.1 points. Default=true
    :param num_to_print: how many of the best results to print to screen
    :return: combos_with_score [[score, [team_ids], [team_names]], ... ]   ([22.2, [1, 4, 11], ['Arsenal', 'Burnley', 'Liverpool']])
    """
    print("Teams to check: ", teams_to_check)
    print("Teams to play: ", teams_to_play)
    print("Double up from one team: ", one_double_up)
    if team_names[0] != -1:
        if teams_to_check > len(team_names):
            print("Teams to check must be >= to number of input teams")
            return -1
    if teams_to_play > teams_to_check:
        print("Teams to play must be smaller than teams_to_check.")
        return -1
    if len(teams_in_solution) > teams_to_check:
        print("Teams_in_solution must be smaller than teams_to_play.")
        return -1
    if one_double_up:
        if teams_to_check < 2:
            print("Teams to check must be >= 1")
            return -1

    # create fixture dataframe. Each element: ['ARS', 'H', 3]
    df = create_data_frame()
    # adjust the fixture difficulty
    if home_away_adjustment > 0:
        l = 0
        #df = adjust_df_for_home_away(df, home_advantage=home_away_adjustment)

    if top_teams_adjustment:
        l = 0
        #df = adjust_df_for_difficult_teams(df)

    static, fixture = get_json()
    dict_with_team_name_to_team_ids = create_dict_with_team_name_to_team_ids(static['teams'])

    team_ids = []
    for team_name in team_names:
        if team_name == -1:
            number_of_teams = df.shape[0]
            team_ids = np.arange(1, number_of_teams + 1)
            break

        team_id = dict_with_team_name_to_team_ids(team_name)
        team_ids.append(team_id)

    number_of_GW = GW_end - GW_start + 1

    dict_team_id_to_fixtures = {}
    dict_team_id_to_home_away = {}
    for idx, team_id in enumerate(team_ids):
        info = fixture_score_one_team(df, team_id, GW_start, GW_end)[3]
        info[info == 0] = 10
        dict_team_id_to_fixtures[team_id] = info
        dict_team_id_to_home_away[team_id] = fixture_score_one_team(df, team_id, GW_start, GW_end)[8]


    dict_with_team_ids_to_team_name = create_dict_with_team_ids_to_team_name(static['teams'])


    if len(teams_in_solution) > 0:
        print("This/these team(s) must be in the solution: ", teams_in_solution, "\n")
    if len(teams_not_in_solution) > 0:
        print("This/these team(s) can not be in the solution: ", teams_not_in_solution, "\n")

    ids_must_be_in_solution = create_list_with_team_ids_from_list_with_team_names(static['teams'], teams_in_solution)
    ids_must_not_be_in_solution = create_list_with_team_ids_from_list_with_team_names(static['teams'], teams_not_in_solution)

    #if one_double_up:
        # allow combinations with one double up from one team
    #    unique_team_ids = [[*comb] for comb in combinations(team_ids, teams_to_check - 1) if
    #                       all(elem in [*comb] for elem in ids_must_be_in_solution)]
    #    temp_unique_team_ids = []
    #    for unique_team_id in unique_team_ids:
    #        for id in team_ids:
    #            temp_unique_team_ids.append(unique_team_id + [id])
    #    unique_team_ids = temp_unique_team_ids

    if not one_double_up:
        # create unique team combos
        unique_team_ids = [[*comb] for comb in combinations(team_ids, teams_to_check) if
                           all(elem in [*comb] for elem in ids_must_be_in_solution)]

    # remove all combinations where teams from teams_not_in_solution exists
    temp_unique_team_ids = []
    for comb in unique_team_ids:
        if len([i for i in comb if i in ids_must_not_be_in_solution]) == 0:
            temp_unique_team_ids.append(comb)
    unique_team_ids = temp_unique_team_ids

    combos_with_score = []
    for team_combos in unique_team_ids:
        team_total_score = 0
        extra_fixtures = 0
        home_games = 0
        for GW in range(number_of_GW):
            GW_home_scores = []
            for team_id in team_combos:
                GW_home_scores.append([dict_team_id_to_fixtures[team_id][GW], dict_team_id_to_home_away[team_id][GW]])

            # team_total_score += np.sum(sorted(GW_scores, key=float)[:teams_to_play])

            sorted_scores = np.array(sorted(GW_home_scores, key=lambda l: l[0], reverse=False))
            home_games += np.sum(sorted_scores[:teams_to_play, 1])
            team_total_score += np.sum(sorted_scores[:teams_to_play, 0])

            # if there are more good games than
            if include_extra_good_games:
                if teams_to_play < len(team_combos):
                    extra_scores = sorted_scores[teams_to_play:, 0]
                    for extra_score in extra_scores:
                        if extra_score <= 2.9:
                            team_total_score -= 0.1
                            extra_fixtures += 1

        combo_names = []
        for team_id in team_combos:
            combo_names.append(dict_with_team_ids_to_team_name[team_id])
        combos_with_score.append(
            [round(team_total_score / number_of_GW / teams_to_play, 4), team_combos, combo_names, extra_fixtures,
             home_games])
    # sort all the combos by the team_total_score. Best fixture will be first element and so on.
    insertion_sort(combos_with_score, len(combos_with_score), element_to_sort=0, min_max="min")
    print("\n\nRank \t Avg difficulty/game \t\t Teams")
    for idx, i in enumerate(combos_with_score[:num_to_print]):
        print(str(idx + 1) + "\t\t" + str(i[0]) + "\t\t" + str(', '.join(i[2])))
    print("\n")
    return combos_with_score


class FDR_team:
    """
        Class of a team
    """
    def __init__(self, team_name, opponent_team_name_short, FDR, H_A, Use_Not_Use):
        """
            Initialize object
        """
        self.team_name = team_name
        self.opponent_team_name = opponent_team_name_short
        self.difficulty_score = int(FDR)
        self.H_A = self.convert_H_A_to_String(H_A)
        self.Use_Not_Use = Use_Not_Use

    def convert_H_A_to_String(self, H_A):
        if H_A == 1:
            return 'H'
        return 'A'


def two_D_list(row, col):
    twod_list = []
    for i in range (0, row):
        new = []
        for j in range (0, col):
            new.append(0)
        twod_list.append(new)
    return twod_list

def find_best_rotation_combos2(GW_start, GW_end, teams_to_check=5, teams_to_play=3, team_names=[-1],
                              teams_in_solution=[], teams_not_in_solution=[],
                              top_teams_adjustment=False, one_double_up=False, home_away_adjustment=True,
                              include_extra_good_games=False, num_to_print=20):
    """
    Find the best rotation combo for "teams_to_check" number of team where "team_to_play" number of them must play each gameweek.
    :param GW_start: start count from this gameweek
    :param GW_end: end counting in this gameweek
    :param teams_to_check: how many teams to check (1-5)
    :param teams_to_play: how many of the teams_to_check you want to play in each gameweek. ( team_to_play < teams_to_check)
    teams_to_check = 5 and teams_to_play = 3 will give the best 5 teams to use in your team if you must use at least 3 each round
    :param team_names: which teams to check from. ([-1]: all teams, ["Arsenal", "Spurs"]: check only some teams)
    :param teams_in_solution: ["Liverpool"]: liverpool must be in the optimal solution. []: no extra dependencies.
    :param home_away_adjustment: wheter to give home/away match score -/+ 0.1 points. Default=true
    :param num_to_print: how many of the best results to print to screen
    :return: combos_with_score [[score, [team_ids], [team_names]], ... ]   ([22.2, [1, 4, 11], ['Arsenal', 'Burnley', 'Liverpool']])
    """
    print("Teams to check: ", teams_to_check)
    print("Teams to play: ", teams_to_play)
    print("Double up from one team: ", one_double_up)
    if team_names[0] != -1:
        if teams_to_check > len(team_names):
            print("Teams to check must be >= to number of input teams")
            return -1
    if teams_to_play > teams_to_check:
        print("Teams to play must be smaller than teams_to_check.")
        return -1
    if len(teams_in_solution) > teams_to_check:
        print("Teams_in_solution must be smaller than teams_to_play.")
        return -1
    if one_double_up:
        if teams_to_check < 2:
            print("Teams to check must be >= 1")
            return -1

    # create fixture dataframe. Each element: ['ARS', 'H', 3]
    # df = create_data_frame()
    df, names, short_names, ids = return_fixture_names_shortnames()

    # adjust the fixture difficulty
    if home_away_adjustment > 0:
        l = 0
        #df = adjust_df_for_home_away(df, home_advantage=home_away_adjustment)

    if top_teams_adjustment:
        l = 0
        #df = adjust_df_for_difficult_teams(df)

    static, fixture = get_json()
    dict_with_team_name_to_team_ids = create_dict_with_team_name_to_team_ids(static['teams'])

    team_ids = []
    for team_name in team_names:
        if team_name == -1:
            number_of_teams = df.shape[0]
            team_ids = np.arange(1, number_of_teams + 1)
            break
        team_id = dict_with_team_name_to_team_ids[team_name]
        team_ids.append(team_id)

    number_of_GW = GW_end - GW_start + 1

    dict_team_id_to_fixtures = {}
    dict_team_id_to_home_away = {}
    dict_team_id_to_opponent = {}

    for idx, team_id in enumerate(team_ids):
        info = fixture_score_one_team(df, team_id, GW_start, GW_end)[3]
        info[info == 0] = 10
        dict_team_id_to_fixtures[team_id] = info
        dict_team_id_to_home_away[team_id] = fixture_score_one_team(df, team_id, GW_start, GW_end)[8]
        dict_team_id_to_opponent[team_id] = fixture_score_one_team(df, team_id, GW_start, GW_end)[2]

    dict_with_team_ids_to_team_name = create_dict_with_team_ids_to_team_name(static['teams'])

    dict_with_team_ids_to_team_short_name = create_dict_with_team_ids_to_team_name(static['teams'], "short_name")


    if len(teams_in_solution) > 0:
        print("This/these team(s) must be in the solution: ", teams_in_solution, "\n")
    if len(teams_not_in_solution) > 0:
        print("This/these team(s) can not be in the solution: ", teams_not_in_solution, "\n")

    ids_must_be_in_solution = create_list_with_team_ids_from_list_with_team_names(static['teams'], teams_in_solution)
    ids_must_not_be_in_solution = create_list_with_team_ids_from_list_with_team_names(static['teams'], teams_not_in_solution)

    #if one_double_up:
        # allow combinations with one double up from one team
    #    unique_team_ids = [[*comb] for comb in combinations(team_ids, teams_to_check - 1) if
    #                       all(elem in [*comb] for elem in ids_must_be_in_solution)]
    #    temp_unique_team_ids = []
    #    for unique_team_id in unique_team_ids:
    #        for id in team_ids:
    #            temp_unique_team_ids.append(unique_team_id + [id])
    #    unique_team_ids = temp_unique_team_ids

    if not one_double_up:
        # create unique team combos
        unique_team_ids = [[*comb] for comb in combinations(team_ids, teams_to_check) if
                           all(elem in [*comb] for elem in ids_must_be_in_solution)]

    # remove all combinations where teams from teams_not_in_solution exists
    temp_unique_team_ids = []
    for comb in unique_team_ids:
        if len([i for i in comb if i in ids_must_not_be_in_solution]) == 0:
            temp_unique_team_ids.append(comb)
    unique_team_ids = temp_unique_team_ids


    combos_with_score = []
    for team_combos in unique_team_ids:
        # team_combos = [1, 3]
        team_total_score = 0
        extra_fixtures = 0
        home_games = 0
        two_D_lis = two_D_list(len(team_combos), number_of_GW)
        for GW_idx, GW in enumerate(range(number_of_GW)):
            GW_home_scores = []
            GW_scores = []
            for team_idx, team_id in enumerate(team_combos):
                # team_id = 6
                FDR = dict_team_id_to_fixtures[team_id][GW]
                H_A = dict_team_id_to_home_away[team_id][GW]
                Opponent = dict_team_id_to_opponent[team_id][GW]
                team_name = dict_with_team_ids_to_team_name[team_id]
                GW_home_scores.append([FDR, H_A])
                GW_scores.append(FDR)
                team_object = FDR_team(team_name, Opponent.upper(),
                                       FDR, H_A, 0)
                two_D_lis[team_idx][GW_idx] = team_object

            Use_Not_Use_idx = np.array(GW_scores).argsort()[:teams_to_play]
            for k in Use_Not_Use_idx:
                two_D_lis[k][GW_idx].Use_Not_Use = 1
            #print("Argsort: ", np.array(GW_scores).argsort()[:teams_to_play], GW_scores)

            # team_total_score += np.sum(sorted(GW_scores, key=float)[:teams_to_play])
            sorted_scores = np.array(sorted(GW_home_scores, key=lambda l: l[0], reverse=False))
            home_games += np.sum(sorted_scores[:teams_to_play, 1])
            team_total_score += np.sum(sorted_scores[:teams_to_play, 0])
            #print(team_combos, team_total_score, home_games, GW_home_scores, teams_to_play)
            # if there are more good games than
            if include_extra_good_games:
                if teams_to_play < len(team_combos):
                    extra_scores = sorted_scores[teams_to_play:, 0]
                    for extra_score in extra_scores:
                        if extra_score <= 2.9:
                            team_total_score -= 0.1
                            extra_fixtures += 1

        combo_names = []
        for team_id in team_combos:
            combo_names.append(dict_with_team_ids_to_team_name[team_id]
)

        combos_with_score.append(
            [round(team_total_score / number_of_GW / teams_to_play, 4), team_combos, combo_names, extra_fixtures,
             home_games, two_D_lis])
        #print("HEHEH", combo_names, team_total_score)
    #print(combos_with_score)
    # sort all the combos by the team_total_score. Best fixture will be first element and so on.
    insertion_sort(combos_with_score, len(combos_with_score), element_to_sort=0, min_max="min")
    print("\n\nRank \t Avg difficulty/game \t\t Teams")
    for idx, i in enumerate(combos_with_score[:num_to_print]):
        print(str(idx + 1) + "\t\t" + str(i[0]) + "\t\t" + str(', '.join(i[2])))
    print("\n")
    #print(combos_with_score[0])
    #for GW in range(number_of_GW):
    #    for TEAM in range(teams_to_check):
    #        ob = combos_with_score[0][-1][TEAM][GW]
    #        print("GW: ", GW_start + GW, ob.team_name, ob.FDR, ob.opponent_team_name_short, ob.Use_Not_Use)
    return combos_with_score

def return_number_of_premier_league_teams():
    # get number of premier league teams
    return 20

def compute_best_fixtures_one_team(df, GW_start, GW_end, team_idx, min_length):
    """
    Find best gameweek region with respect to fixture values between GW_start and GW_end with a lenght
    >= min_length.
    :param df: dataframe with fixture data. create_data_frame()
    :param GW_start: first GW to count. GW_start > 0. (1)
    :param GW_end: last GW to count. GW_start <= GW_end <= 38 (38)
    :param team_idx:
    :param min_length: must be smaller than GW_end - GW_start + 1
    :return:
    """
    if min_length > (GW_end-GW_start+1):
        print('min_length: must be smaller than GW_end - GW_start + 1')
        return -1
    max_info = fixture_score_one_team(df, team_idx, GW_start, GW_end)
    ii, jj, length = GW_start, GW_end, len(max_info[2])
    max_score = max_info[0] / (GW_end - GW_start + 1)
    for i in range(GW_end - GW_start + 1):
        for j in range(i + min_length - 1, GW_end - GW_start + 1):
            temp_score = np.sum(max_info[3][i:j+1]) / (j - i + 1)
            temp_len = j - i + 1
            if temp_score <= max_score:
                if temp_score == max_score and temp_len > length:
                    ii, jj, length = i+1, j+1, temp_len
                    max_score = temp_score
                if temp_score != max_score:
                    ii, jj, length = i+1, j+1, temp_len
                    max_score = temp_score
    return fixture_score_one_team(df, team_idx, ii, jj)


def find_best_fixture_with_min_length_each_team(data, GW_start, GW_end, min_length=5):
    df = create_data_frame()
    #df = adjust_df_for_blanks(df, adjust_blank_to=6)
    #df = adjust_df_for_home_away(df)
    #df = adjust_df_for_difficult_teams(df)
    best_fixtures_min_length = []
    for team_id in range(return_number_of_premier_league_teams()):
        info = compute_best_fixtures_one_team_db_data(data, GW_start, GW_end, team_id + 1, min_length)
        best_fixtures_min_length.append(info)
    return best_fixtures_min_length


def create_FDR_dict(fdr_data, blank_score=10):
    num_gws = len(fdr_data.gw)
    new_dict = {new_list: [] for new_list in range(num_gws + 1)}

    for gw, H_A, opponent, FDR in zip(fdr_data.gw, fdr_data.oppTeamHomeAwayList, fdr_data.oppTeamNameList, fdr_data.oppTeamDifficultyScore):
        new_dict[gw].append([opponent, H_A, FDR])
    for i in range(1, num_gws + 1):
        if not new_dict[i]:
            new_dict[i] = [['-', ' ', blank_score]]
    return new_dict

def calc_score(fdr_dict, gw_start, gw_end):
    score = 0
    for gw in range(gw_start, gw_end + 1):
        data = fdr_dict[gw]
        gws_this_round = len(data)
        temp_score = 0
        if gws_this_round > 1:
            for i in range(gws_this_round):
                temp_score += data[i][2]
        else:
            temp_score += data[0][2]
        score += temp_score / gws_this_round ** 2
    return score

def compute_best_fixtures_one_team_db_data(data, GW_start, GW_end, team_idx, min_length):
    """
    Find best gameweek region with respect to fixture values between GW_start and GW_end with a lenght
    >= min_length.
    :param df: dataframe with fixture data. create_data_frame()
    :param GW_start: first GW to count. GW_start > 0. (1)
    :param GW_end: last GW to count. GW_start <= GW_end <= 38 (38)
    :param team_idx:
    :param min_length: must be smaller than GW_end - GW_start + 1
    :return:
    """
    if min_length > (GW_end-GW_start+1):
        print('min_length: must be smaller than GW_end - GW_start + 1')
        return -1
    fdr_dict = create_FDR_dict(data[team_idx - 1])
    number_of_gameweeks = GW_end - GW_start + 1
    ii, jj, length = GW_start, GW_end, number_of_gameweeks
    max_score = calc_score(fdr_dict, GW_start, GW_end) / (GW_end - GW_start + 1)
    for i in range(GW_start, GW_end + 1):
        for j in range(i + min_length - 1, GW_end + 1):
            #temp_score = np.sum(max_info[3][i:j+1]) / (j - i + 1)
            temp_len = j - i + 1
            temp_score = calc_score(fdr_dict, gw_start=i, gw_end=j) / temp_len
            if temp_score <= max_score:
                if temp_score == max_score and temp_len > length:
                    ii, jj, length = i, j, temp_len
                    max_score = temp_score
                if temp_score != max_score:
                    ii, jj, length = i, j, temp_len
                    max_score = temp_score
    fixture_list = []
    for gw in range(GW_start, GW_end + 1):
        gw_fixtures = fdr_dict[gw]
        temp_list = []
        good_gw = 0
        if gw >= ii and gw <= jj:
            good_gw = 1
        for gw_fixture in gw_fixtures:
            temp_list.append([FDR_team(data[team_idx - 1].team_name, gw_fixture[0].upper(),
                                       gw_fixture[2], gw_fixture[1], good_gw)])
        fixture_list.append(temp_list)
    return fixture_list



def find_best_fixture_with_min_length_each_team2(GW_start, GW_end, min_length=5):
    df = create_data_frame()
    #df = adjust_df_for_blanks(df, adjust_blank_to=6)
    #df = adjust_df_for_home_away(df)
    #df = adjust_df_for_difficult_teams(df)
    best_fixtures_min_length = []
    for team_id in range(return_number_of_premier_league_teams()):
        info = compute_best_fixtures_one_team(df, GW_start, GW_end, team_id + 1, min_length)
        print(str(info[1]) + " GW: " + str(info[4]) + " to " + str(info[5]) + " (avg: " + str(info[7]) + ") " + str(
            info[2]))
        best_fixtures_min_length.append(info)
    return best_fixtures_min_length

#find_best_rotation_combos2(15, 26, teams_to_check=2, teams_to_play=1, team_names=[-1],
#                             teams_in_solution=[], teams_not_in_solution=[],
#                              top_teams_adjustment=False, one_double_up=False, home_away_adjustment=True,
#                              include_extra_good_games=False, num_to_print=20)