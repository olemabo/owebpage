from django.shortcuts import render
from fixture_planner.models import AddPlTeamsToDB
from django.http import HttpResponse
from django.views import generic
import numpy as np


class FixturePlannerView(generic.ListView):
    model = AddPlTeamsToDB


def show_statistics(request):
    return HttpResponse("Hello World2")

class team_info:
    def __init__(self, opponent_team_name, difficulty_score, H_A, team_name, FDR_score):
        ...
        self.opponent_team_name = opponent_team_name
        self.difficulty_score = difficulty_score
        self.H_A = H_A
        self.team_name = team_name
        self.FDR_score = FDR_score


def show_statistics(request, ownership_vs_nationality="nationality", top_x=10000, gw=38):
    """View function for home page of site."""
    # Generate counts of some of the main objects
    context = {
        'ownership_vs_nationality': ownership_vs_nationality,
        'top_x': top_x,
        'gw': gw,
    }
    # Render the HTML template index_catalog.html with the data in the context variable
    return render(request, 'statistics.html', context=context)

