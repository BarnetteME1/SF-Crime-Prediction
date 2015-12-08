from django.shortcuts import render, render_to_response

# Create your views here.
from django.views.generic import TemplateView


class IndexView(TemplateView):
    template_name = 'base.html'

class MapView(TemplateView):
    template_name = 'maps.html'

class GraphView(TemplateView):
    template_name = 'graphs.html'