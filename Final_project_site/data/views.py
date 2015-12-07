from django.shortcuts import render, render_to_response

# Create your views here.
from django.views.generic import TemplateView


def index_view(request):
    context = {}
    return render_to_response(template_name='base.html', context=context)