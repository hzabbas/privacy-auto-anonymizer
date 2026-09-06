from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/analyze/', views.analyze_image_api, name='analyze_image_api'),
]
