from django.shortcuts import render
from fixture_planner.models import AddPlTeamsToDB, KickOffTime
from django.http import HttpResponse
from django.views import generic
import fixture_planner.read_data as read_data
import fixture_planner.fixture_algorithms as alg
from datetime import date
import statistic.calculate_statistics as stat

class FixturePlannerView(generic.ListView):
    model = AddPlTeamsToDB


def fill_data_base(request):
    df, names, short_names, ids = read_data.return_fixture_names_shortnames()
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
    stat.fill_database_all_players()
    return HttpResponse("Filled Database")


class team_info:
    def __init__(self, opponent_team_name, difficulty_score, H_A, team_name, FDR_score):
        ...
        self.opponent_team_name = opponent_team_name
        self.difficulty_score = difficulty_score
        self.H_A = H_A
        self.team_name = team_name
        self.FDR_score = FDR_score
        self.Use_Not_Use = 0


def get_current_gw():
    # find current gw
    today_date = date.today()
    kickofftime_db = KickOffTime.objects.filter(gameweek__range=(0, 38))
    for i in range(len(kickofftime_db)):
        current_gw = i + 1
        dates = kickofftime_db[i].kickoff_time.split("T")[0].split("-")
        gw_i_date = date(int(dates[0]), int(dates[1]), int(dates[2]))
        if gw_i_date > today_date:
            return current_gw
    return 0


def get_max_gw():
    return 38


class which_team_to_check:
    def __init__(self, team_name, checked, checked_must_be_in_solution=''):
        ...
        self.team_name = team_name
        self.checked = checked
        self.checked_must_be_in_solution = checked_must_be_in_solution



def fixture_planner(request, start_gw=get_current_gw(), end_gw=get_current_gw()+5, combinations="FDR", teams_to_check=2, teams_to_play=1, min_num_fixtures=4):
    """View function for home page of site."""
    # Generate counts of some of the main objects
    if end_gw > get_max_gw():
        end_gw = get_max_gw()

    # collect data from database [[AddPlTeamsToDB], [AddPlTeamsToDB], ... ]
    fixture_list_db = AddPlTeamsToDB.objects.all()
    team_name_list = []
    team_dict = {}
    number_of_teams = len(fixture_list_db)
    for i in range(number_of_teams):
        team_dict[fixture_list_db[i].team_name] = which_team_to_check(fixture_list_db[i].team_name, 'checked')

    fixture_list = [fixture_list_db[i] for i in range(0, number_of_teams)]
    fpl_teams = [-1]
    if request.method == 'POST':
        for i in range(number_of_teams):
            team_dict[fixture_list[i].team_name] = which_team_to_check(fixture_list[i].team_name, '')

        fpl_teams = request.POST.getlist('fpl-teams')
        for fpl_team in fpl_teams:
            team_dict[fpl_team] = which_team_to_check(team_dict[fpl_team].team_name, 'checked')

        # gameweek info
        gw_info = request.POST.getlist('gw-info')
        start_gw = int(gw_info[0])
        end_gw = int(gw_info[1])

        combinations = request.POST.getlist('combination')[0]
        min_num_fixtures = int(request.POST.getlist('min_num_fixtures')[0])

        teams_to_check = int(request.POST.getlist('teams_to_check')[0])
        teams_to_play = int(request.POST.getlist('teams_to_play')[0])

    if end_gw < start_gw:
          end_gw = start_gw + 1

    gws = end_gw - start_gw + 1
    gw_numbers = [i for i in range(start_gw, end_gw + 1)]
    kickofftime_db = KickOffTime.objects.filter(gameweek__range=(start_gw, end_gw))

    fixture_list = []
    for i in range(number_of_teams):
        temp_object = team_dict[fixture_list_db[i].team_name]
        team_name_list.append(team_dict[fixture_list_db[i].team_name])

        if temp_object.checked == 'checked':
            fixture_list.append(fixture_list_db[i])
    teams = len(fixture_list)

    fdr_fixture_data = []
    if combinations == 'FDR':
        FDR_scores = []
        for idx, i in enumerate(fixture_list):
            fdr_dict = alg.create_FDR_dict(i)
            sum = alg.calc_score(fdr_dict, start_gw, end_gw)
            FDR_scores.append([i, sum])
        FDR_scores = sorted(FDR_scores, key=lambda x: x[1], reverse=False)

        for i in range(teams):
            temp_list2 = [[] for i in range(gws)]
            team_i = FDR_scores[i][0]
            FDR_score = FDR_scores[i][1]
            temp_gws = team_i.gw
            for j in range(len(team_i.gw)):
                temp_gw = temp_gws[j]
                if temp_gw in gw_numbers:
                    temp_list2[gw_numbers.index(temp_gw)].append([
                        team_info(team_i.oppTeamNameList[j],
                                                   team_i.oppTeamDifficultyScore[j],
                                                   team_i.oppTeamHomeAwayList[j],
                                                   team_i.team_name,
                                                   FDR_score)
                    ])
            for k in range(len(temp_list2)):
                if not temp_list2[k]:
                    temp_list2[k] = [[team_info("-", 0, " ", team_i.team_name, 0)]]

            fdr_fixture_data.append(temp_list2)


    rotation_data = []
    if combinations == 'Rotation':
        teams_in_solution = []
        if request.method == 'POST':
            teams_in_solution = request.POST.getlist('fpl-teams-in-solution')
        remove_these_teams = []
        for team_sol in teams_in_solution:
            if team_sol not in fpl_teams:
                remove_these_teams.append(team_sol)
        for remove_team in remove_these_teams:
            teams_in_solution.remove(remove_team)
        for i in team_name_list:
            if i.team_name in teams_in_solution:
                i.checked_must_be_in_solution = 'checked'
        rotation_data = alg.find_best_rotation_combos2(fixture_list_db, start_gw, end_gw,
                    teams_to_check=teams_to_check, teams_to_play=teams_to_play,
                    team_names=fpl_teams, teams_in_solution=teams_in_solution, teams_not_in_solution=[],
                    top_teams_adjustment=False, one_double_up=False,
                    home_away_adjustment=True, include_extra_good_games=False,
                                    num_to_print=0)
        if rotation_data == -1:
            rotation_data = [['Wrong input', [], [], 0, 0, [[]]]]
        else:
            rotation_data = rotation_data[:(min(len(rotation_data), 50))]

    if abs(min_num_fixtures) > (end_gw-start_gw):
            min_num_fixtures = abs(end_gw-start_gw)
            if min_num_fixtures == 0:
                min_num_fixtures = 1
                end_gw = start_gw + 1
    if combinations == 'FDR-best':
        fdr_fixture_data = alg.find_best_fixture_with_min_length_each_team(fixture_list, GW_start=start_gw, GW_end=end_gw, min_length=min_num_fixtures)


    context = {
        'teams': teams,
        'gws': gws,
        'gw_numbers': kickofftime_db,
        'gw_start': start_gw,
        'gw_end': end_gw,
        'combinations': combinations,
        'rotation_data': rotation_data,
        'teams_to_play': teams_to_play,
        'teams_to_check': teams_to_check,
        'fdr_fixture_data': fdr_fixture_data,
        'min_num_fixtures': min_num_fixtures,
        'team_name_list': team_name_list,
    }
    # Render the HTML template index_catalog.html with the data in the context variable
    return render(request, 'fixture_planner_main.html', context=context)




"""
    list = []
    for i in range(teams):
        temp_list = []
        team_i = FDR_scores[i][0]
        FDR_score = FDR_scores[i][1]
        for j in range(start_gw-1, end_gw):
            gw = team_i.gw
            if gw == 0:
                temp_list.append(team_info(team_i.oppTeamNameList[j],
                                           team_i.oppTeamDifficultyScore[j],
                                           team_i.oppTeamHomeAwayList[j],
                                           team_i.team_name,
                                           FDR_score))
            else:
                temp_list.append(team_info(team_i.oppTeamNameList[j],
                                   team_i.oppTeamDifficultyScore[j],
                                   team_i.oppTeamHomeAwayList[j],
                                    team_i.team_name,
                                       FDR_score ))


        list.append(temp_list)
    """