from django.urls import path
from . import views

urlpatterns = [
    path('', views.customer_list, name='customer_list'),
    path('analisis/', views.analysis, name='analysis'),
    path('segmentasi/', views.segmentasi, name='segmentasi'),
    path('fis/', views.fis_pelanggan, name='fis'),
]
