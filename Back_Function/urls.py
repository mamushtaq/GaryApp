"""Back_Function URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from myapp import views
from django.conf import settings
from django.conf.urls.static import static
# hello
urlpatterns = [
    # define a route for home
    path('', views.homepage, name='homepage'),
    path('api/upload/', views.upload_pdf, name='upload_pdf'),
    path('chat2/', views.chat2, name='chat2'),
    path('api/getpdf/', views.getPDF, name='getpdf'),
    path('home/', views.home, name='home'),
    path('txt_upload/', views.upload, name='index'),
    path('webhook/', views.webhook, name='webhook'),
    path('chat/', views.contact, name='contact'),
    path('chatapp/', views.chatapp, name='chatapp'),
    path('chatting/', views.chattingView, name='chatting'),
    path('api/chatting/', views.chatting, name='chattingapi'),
    path('api/chatting_tuned/', views.chatting_tuned, name='chatting'),
    path('chatting_tuned/', views.chatting_tunedView, name='chatting'),
    path('api/grammar/', views.grammar, name='chatting'),
    path('grammar/', views.grammarView, name='chatting'),
    path('check_availability/', views.check_availability, name='check'),
    path('get_caller_id/', views.set_caller_id, name='get_caller_id'),
    path('sessionID/', views.get_SessionId, name='get_session_id'),
    path('sessionID_Suc/', views.get_SessionId_Suc, name='get_session_id_suc')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
