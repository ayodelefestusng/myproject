
from django.urls import path
from . import views

# app_name = 'myapp'

urlpatterns = [
    path('post', views.post_list, name='post_list'),
    path('home', views.home, name='home'),
    path('', views.chat_home, name='chat_home'),
    path('send/', views.send_message, name='send_message'),
    path('history/', views.chat_history, name='chat_history'),
    path('conversation/<int:conversation_id>/', views.view_conversation, name='view_conversation'),
    
    path('edit_currency/', views.edit_currency, name='edit_currency'),
    path('oya/', views.oya, name='oya'),
    path('oya1/<str:param_name>/', views.send_message1, name='send_message1'),
    path("dashboard/", views.dashboard, name="dashboard"),
]