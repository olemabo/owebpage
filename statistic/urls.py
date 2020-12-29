from . import views
from django.urls import path

#
urlpatterns = [
    path("", views.show_statistics, name='statistic'),
    path("<str:ownership_vs_nationality>/<int:top_x>/<int:gw>", views.show_statistics, name='show_statistics-ownership_vs_nationality-top_x-gw'),
]